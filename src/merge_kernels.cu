// merge_kernels.cu -- Hybrid CPU+GPU merge pipeline kernels
//
// Post-expansion merge (RTSS 2024, Eq. 24-29):
//   Condition 1: D(N1) = D(N2)  (guaranteed by D-key grouping)
//   Condition 2: Availability overlap for all cores
//   Condition 3: Finish-time overlap for all tracked jobs
//
// Merge ops: X=AND, A_min=min, A_max=max, F_min=min, F_max=max, F_mask=OR, ovf=OR
//
// OPTIMIZATIONS:
//   Phase 1b: ComputeGroupSizesKernel/FusedRangeMergeKernel read num_groups from device ptr
//   Phase 2:  Warp-cooperative FusedRangeMergeKernelWarp (1 warp per D-group)
//   Phase 4:  Fused ExtractDKeysAndIotaKernel

#include "sag_config.h"
#include "sag_types.h"
#include <cstdio>

using namespace sag;
using namespace sag::config;

// ===========================================================================
// Device helper: copy all fields of one state
// ===========================================================================
__device__ void dev_copy_full_state(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayout layout, int n, int m, int W)
{
    const uint64_t* sD = layout.D(src_buf, src_idx);
    uint64_t*       dD = layout.D(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dD[w] = sD[w];

    const uint64_t* sX = layout.X(src_buf, src_idx);
    uint64_t*       dX = layout.X(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dX[w] = sX[w];

    const uint64_t* sF = layout.F_mask(src_buf, src_idx);
    uint64_t*       dF = layout.F_mask(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dF[w] = sF[w];

    const int32_t* sAmin = layout.A_min(src_buf, src_idx);
    int32_t*       dAmin = layout.A_min(dst_buf, dst_idx);
    for (int i = 0; i < m; i++) dAmin[i] = sAmin[i];

    const int32_t* sAmax = layout.A_max(src_buf, src_idx);
    int32_t*       dAmax = layout.A_max(dst_buf, dst_idx);
    for (int i = 0; i < m; i++) dAmax[i] = sAmax[i];

    const int32_t* sFmin = layout.F_min(src_buf, src_idx);
    int32_t*       dFmin = layout.F_min(dst_buf, dst_idx);
    for (int k = 0; k < n; k++) dFmin[k] = sFmin[k];

    const int32_t* sFmax = layout.F_max(src_buf, src_idx);
    int32_t*       dFmax = layout.F_max(dst_buf, dst_idx);
    for (int k = 0; k < n; k++) dFmax[k] = sFmax[k];

    *layout.ovf(dst_buf, dst_idx) = *layout.ovf(src_buf, src_idx);
}

// ===========================================================================
// Device helper: check range compatibility (conditions 2+3)
// ===========================================================================
__device__ bool dev_check_range_compatible(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayout layout, int n, int m, int W,
    bool useJobFinishTimes = false)
{
    // Condition 2: availability intervals overlap for ALL cores
    const int32_t* a_Amin = layout.A_min(buf_a, idx_a);
    const int32_t* a_Amax = layout.A_max(buf_a, idx_a);
    const int32_t* b_Amin = layout.A_min(buf_b, idx_b);
    const int32_t* b_Amax = layout.A_max(buf_b, idx_b);

    for (int i = 0; i < m; i++) {
        if (a_Amin[i] > b_Amax[i] + 1) return false;
        if (b_Amin[i] > a_Amax[i] + 1) return false;
    }

    // Condition 3: finish-time intervals overlap for ALL tracked jobs
    // (disabled by default to match nptest's default behavior)
    if (useJobFinishTimes) {
        const uint64_t* fm = layout.F_mask(buf_b, idx_b);
        const int32_t* a_Fmin = layout.F_min(buf_a, idx_a);
        const int32_t* a_Fmax = layout.F_max(buf_a, idx_a);
        const int32_t* b_Fmin = layout.F_min(buf_b, idx_b);
        const int32_t* b_Fmax = layout.F_max(buf_b, idx_b);

        for (int w = 0; w < W; w++) {
            uint64_t bits = fm[w];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int J = w * 64 + bit;
                if (J >= n) break;
                if (a_Fmin[J] > b_Fmax[J] + 1) return false;
                if (b_Fmin[J] > a_Fmax[J] + 1) return false;
                bits &= bits - 1;
            }
        }
    }
    return true;
}

// ===========================================================================
// Device helper: merge state into existing slot
// ===========================================================================
__device__ void dev_merge_into_slot(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayout layout, int n, int m, int W)
{
    // X: AND (intersection)
    const uint64_t* sX = layout.X(src_buf, src_idx);
    uint64_t*       dX = layout.X(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dX[w] &= sX[w];

    // A_min: component-wise min
    const int32_t* sAmin = layout.A_min(src_buf, src_idx);
    int32_t*       dAmin = layout.A_min(dst_buf, dst_idx);
    for (int i = 0; i < m; i++)
        if (sAmin[i] < dAmin[i]) dAmin[i] = sAmin[i];

    // A_max: component-wise max
    const int32_t* sAmax = layout.A_max(src_buf, src_idx);
    int32_t*       dAmax = layout.A_max(dst_buf, dst_idx);
    for (int i = 0; i < m; i++)
        if (sAmax[i] > dAmax[i]) dAmax[i] = sAmax[i];

    // F_min/F_max: iterate over (src.F_mask UNION dst.F_mask). Bits only in
    // src must be copied (their dst slot is stale); bits in both get min/max
    // merged. Iterating only dst.F_mask drops src-only witnesses → WCRT
    // under-estimates and unsoundness.
    const uint64_t* sFmask = layout.F_mask(src_buf, src_idx);
    const uint64_t* dFmask = layout.F_mask(dst_buf, dst_idx);
    const int32_t* sFmin = layout.F_min(src_buf, src_idx);
    int32_t*       dFmin = layout.F_min(dst_buf, dst_idx);
    const int32_t* sFmax = layout.F_max(src_buf, src_idx);
    int32_t*       dFmax = layout.F_max(dst_buf, dst_idx);

    for (int w = 0; w < W; w++) {
        uint64_t bits = sFmask[w] | dFmask[w];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int J = w * 64 + bit;
            if (J >= n) break;
            bool in_src = (sFmask[w] >> bit) & 1;
            bool in_dst = (dFmask[w] >> bit) & 1;
            if (in_src && in_dst) {
                if (sFmin[J] < dFmin[J]) dFmin[J] = sFmin[J];
                if (sFmax[J] > dFmax[J]) dFmax[J] = sFmax[J];
            } else if (in_src) {
                dFmin[J] = sFmin[J];
                dFmax[J] = sFmax[J];
            }
            bits &= bits - 1;
        }
    }

    // F_mask: OR (union)
    uint64_t* dF = layout.F_mask(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dF[w] |= sFmask[w];

    // ovf: OR
    *layout.ovf(dst_buf, dst_idx) |= *layout.ovf(src_buf, src_idx);
}

// ===========================================================================
// Phase 4: Fused ExtractDKeys + Iota kernel
// ===========================================================================
__global__ void ExtractDKeysAndIotaKernel(
    const char* __restrict__ d_output,
    SAGStateLayout layout,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys,
    int* __restrict__ d_indices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    const uint64_t* D = layout.D(d_output, tid);
    for (int w = 0; w < W; w++)
        d_D_keys[tid * W + w] = D[w];
    d_indices[tid] = tid;
}

// ===========================================================================
// Legacy kernels kept for backward compatibility
// ===========================================================================
__global__ void ExtractDKeysKernel(
    const char* __restrict__ d_output,
    SAGStateLayout layout,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    const uint64_t* D = layout.D(d_output, tid);
    for (int w = 0; w < W; w++)
        d_D_keys[tid * W + w] = D[w];
}

__global__ void IotaKernel(int* d_indices, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) d_indices[tid] = tid;
}

// ===========================================================================
// Kernel: Gather D-keys in sorted order (for D2H)
// ===========================================================================
__global__ void GatherDKeysKernel(
    const uint64_t* __restrict__ d_D_keys_unsorted,
    const int*      __restrict__ d_sorted_indices,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys_sorted)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    int orig = d_sorted_indices[tid];
    for (int w = 0; w < W; w++)
        d_D_keys_sorted[tid * W + w] = d_D_keys_unsorted[orig * W + w];
}

// ===========================================================================
// Kernel: Detect group boundaries in sorted D-keys (GPU-only merge path)
// ===========================================================================
__global__ void DetectBoundariesKernel(
    const uint64_t* __restrict__ d_D_keys_sorted,
    int num_states, int W,
    int* __restrict__ d_is_start)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    if (tid == 0) {
        d_is_start[tid] = 1;
        return;
    }

    bool same = true;
    for (int w = 0; w < W; w++) {
        if (d_D_keys_sorted[tid * W + w] != d_D_keys_sorted[(tid - 1) * W + w]) {
            same = false;
            break;
        }
    }
    d_is_start[tid] = same ? 0 : 1;
}

// ===========================================================================
// Kernel: Compact group starts from boundary flags + prefix sum
// ===========================================================================
__global__ void CompactGroupStartsKernel(
    const int* __restrict__ d_is_start,
    const int* __restrict__ d_group_id,
    int num_states,
    int* __restrict__ d_group_starts,
    int* __restrict__ d_num_groups)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    if (d_is_start[tid]) {
        d_group_starts[d_group_id[tid]] = tid;
    }

    if (tid == num_states - 1) {
        *d_num_groups = d_group_id[tid] + d_is_start[tid];
    }
}

// ===========================================================================
// Phase 1b: ComputeGroupSizesKernel - reads num_groups from device pointer
// ===========================================================================
__global__ void ComputeGroupSizesKernel(
    const int* __restrict__ d_group_starts,
    int num_groups, int num_states,
    int* __restrict__ d_group_sizes)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    int start = d_group_starts[gid];
    int end = (gid + 1 < num_groups) ? d_group_starts[gid + 1] : num_states;
    d_group_sizes[gid] = end - start;
}

// Overload that reads num_groups from device pointer (Phase 1b: no CPU sync needed)
__global__ void ComputeGroupSizesKernelDev(
    const int* __restrict__ d_group_starts,
    const int* __restrict__ d_num_groups_ptr,
    int num_states,
    int* __restrict__ d_group_sizes)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int ng = *d_num_groups_ptr;
    if (gid >= ng) return;

    int start = d_group_starts[gid];
    int end = (gid + 1 < ng) ? d_group_starts[gid + 1] : num_states;
    d_group_sizes[gid] = end - start;
}

// ===========================================================================
// Legacy: FusedRangeMergeKernel (1 thread per D-group, kept for fallback)
// ===========================================================================
__global__ void FusedRangeMergeKernel(
    const char* __restrict__ d_expanded,
    const int*  __restrict__ d_sorted_idx,
    const int*  __restrict__ d_group_starts,
    const int*  __restrict__ d_group_sizes,
    int         num_groups,
    SAGStateLayout layout,
    int n, int m, int W,
    char*       d_merge_buf,
    int*        d_merge_count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    int g_start = d_group_starts[gid];
    int g_size  = d_group_sizes[gid];

    int local_slots[MAX_SLOTS_PER_GROUP];
    int num_slots = 0;

    for (int s = 0; s < g_size; s++) {
        int orig_idx = d_sorted_idx[g_start + s];
        bool merged = false;

        for (int sl = 0; sl < num_slots; sl++) {
            if (dev_check_range_compatible(d_expanded, orig_idx,
                                           d_merge_buf, local_slots[sl],
                                           layout, n, m, W, true)) {
                dev_merge_into_slot(d_expanded, orig_idx,
                                    d_merge_buf, local_slots[sl],
                                    layout, n, m, W);
                merged = true;
                break;
            }
        }

        if (!merged) {
            int slot_idx = atomicAdd(d_merge_count, 1);
            dev_copy_full_state(d_expanded, orig_idx,
                               d_merge_buf, slot_idx,
                               layout, n, m, W);
            if (num_slots < MAX_SLOTS_PER_GROUP) {
                local_slots[num_slots++] = slot_idx;
            }
        }
    }
}

// ===========================================================================
// Phase 2: Warp-Cooperative FusedRangeMergeKernel
// 1 warp (32 threads) per D-group
// Grid: ceil(num_groups / MERGE_WARPS_PER_BLOCK)
// Block: MERGE_WARPS_PER_BLOCK * 32 threads
// ===========================================================================
__global__ void FusedRangeMergeKernelWarp(
    const char* __restrict__ d_expanded,
    const int*  __restrict__ d_sorted_idx,
    const int*  __restrict__ d_group_starts,
    const int*  __restrict__ d_num_groups_ptr,  // Phase 1b: read from device
    int         num_states,                      // for group size computation
    SAGStateLayout layout,
    int n, int m, int W,
    char*       d_merge_buf,
    int*        d_merge_count)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warpInBlock = threadIdx.x / WARP_SIZE;
    const int globalWarp = blockIdx.x * MERGE_WARPS_PER_BLOCK + warpInBlock;

    // Phase 1b: read num_groups from device pointer
    int num_groups = *d_num_groups_ptr;
    if (globalWarp >= num_groups) return;

    // Load group info (lane 0 loads, broadcast)
    int g_start, g_size;
    if (lane == 0) {
        g_start = d_group_starts[globalWarp];
        int g_end = (globalWarp + 1 < num_groups)
                    ? d_group_starts[globalWarp + 1] : num_states;
        g_size = g_end - g_start;
    }
    g_start = __shfl_sync(0xFFFFFFFF, g_start, 0);
    g_size  = __shfl_sync(0xFFFFFFFF, g_size, 0);

    if (g_size <= 0) return;

    // Shared memory for slot indices (per warp)
    extern __shared__ int smem_merge[];
    int* my_slots = smem_merge + warpInBlock * MAX_SLOTS_PER_GROUP;
    int num_slots = 0;

    // Process each state in this D-group sequentially
    for (int s = 0; s < g_size; s++) {
        int orig_idx = d_sorted_idx[g_start + s];
        bool merged = false;
        int merge_slot = -1;

        // --- Parallel slot search: each lane checks a different slot ---
        // Round 1: lanes 0-31 check slots 0-31
        if (lane < num_slots && lane < 32) {
            int sl_idx = my_slots[lane];
            if (dev_check_range_compatible(d_expanded, orig_idx,
                                           d_merge_buf, sl_idx,
                                           layout, n, m, W, true)) {
                merge_slot = lane;
            }
        }
        // Collect results: ballot gives bitmask of lanes that found compatible slot
        unsigned ballot1 = __ballot_sync(0xFFFFFFFF, merge_slot >= 0);
        if (ballot1 != 0) {
            int first_lane = __ffs(ballot1) - 1;  // lowest-numbered = deterministic
            if (lane == first_lane) {
                dev_merge_into_slot(d_expanded, orig_idx,
                                    d_merge_buf, my_slots[merge_slot],
                                    layout, n, m, W);
            }
            merged = true;
        }

        // Round 2: if >32 slots and not yet merged, check slots 32-63
        if (!merged && num_slots > 32) {
            merge_slot = -1;
            int sl2 = 32 + lane;
            if (sl2 < num_slots) {
                int sl_idx = my_slots[sl2];
                if (dev_check_range_compatible(d_expanded, orig_idx,
                                               d_merge_buf, sl_idx,
                                               layout, n, m, W, true)) {
                    merge_slot = sl2;
                }
            }
            unsigned ballot2 = __ballot_sync(0xFFFFFFFF, merge_slot >= 0);
            if (ballot2 != 0) {
                int first_lane = __ffs(ballot2) - 1;
                if (lane == first_lane) {
                    dev_merge_into_slot(d_expanded, orig_idx,
                                        d_merge_buf, my_slots[merge_slot],
                                        layout, n, m, W);
                }
                merged = true;
            }
        }

        // Broadcast merged status
        merged = (__ballot_sync(0xFFFFFFFF, merged) != 0);

        // If no compatible slot found, create new slot
        if (!merged) {
            int slot_idx;
            if (lane == 0) {
                slot_idx = atomicAdd(d_merge_count, 1);
            }
            slot_idx = __shfl_sync(0xFFFFFFFF, slot_idx, 0);

            // Cooperative state copy (all lanes participate)
            // Copy bitmask fields
            {
                const uint64_t* sD = layout.D(d_expanded, orig_idx);
                uint64_t*       dD = layout.D(d_merge_buf, slot_idx);
                if (lane < W) dD[lane] = sD[lane];

                const uint64_t* sX = layout.X(d_expanded, orig_idx);
                uint64_t*       dX = layout.X(d_merge_buf, slot_idx);
                if (lane < W) dX[lane] = sX[lane];

                const uint64_t* sF = layout.F_mask(d_expanded, orig_idx);
                uint64_t*       dF = layout.F_mask(d_merge_buf, slot_idx);
                if (lane < W) dF[lane] = sF[lane];
            }

            // Copy A_min/A_max
            {
                const int32_t* sAmin = layout.A_min(d_expanded, orig_idx);
                int32_t*       dAmin = layout.A_min(d_merge_buf, slot_idx);
                const int32_t* sAmax = layout.A_max(d_expanded, orig_idx);
                int32_t*       dAmax = layout.A_max(d_merge_buf, slot_idx);
                for (int i = lane; i < m; i += WARP_SIZE) {
                    dAmin[i] = sAmin[i];
                    dAmax[i] = sAmax[i];
                }
            }

            // Copy F_min/F_max cooperatively
            {
                const int32_t* sFmin = layout.F_min(d_expanded, orig_idx);
                int32_t*       dFmin = layout.F_min(d_merge_buf, slot_idx);
                const int32_t* sFmax = layout.F_max(d_expanded, orig_idx);
                int32_t*       dFmax = layout.F_max(d_merge_buf, slot_idx);
                for (int k = lane; k < n; k += WARP_SIZE) {
                    dFmin[k] = sFmin[k];
                    dFmax[k] = sFmax[k];
                }
            }

            // Copy ovf
            if (lane == 0) {
                *layout.ovf(d_merge_buf, slot_idx) = *layout.ovf(d_expanded, orig_idx);
            }

            // Add to slot list
            if (lane == 0 && num_slots < MAX_SLOTS_PER_GROUP) {
                my_slots[num_slots] = slot_idx;
            }
            __syncwarp(0xFFFFFFFF);
            if (num_slots < MAX_SLOTS_PER_GROUP) num_slots++;
        }

        __syncwarp(0xFFFFFFFF);
    }
}

// ===========================================================================
// Kernel: Update BCRT/WCRT from merged states' F_min/F_max
// ===========================================================================
__global__ void UpdateBCRTWCRTFromMergedKernel(
    const char* __restrict__ d_merged_buf,
    SAGStateLayout layout,
    int num_merged,
    int n, int W,
    const int32_t* __restrict__ d_r_min,
    int32_t* d_BCRT,
    int32_t* d_WCRT)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_merged) return;

    const uint64_t* fm = layout.F_mask(d_merged_buf, sid);
    const int32_t* fmin = layout.F_min(d_merged_buf, sid);
    const int32_t* fmax = layout.F_max(d_merged_buf, sid);

    for (int w = 0; w < W; w++) {
        uint64_t bits = fm[w];
        while (bits) {
            int bit = __ffsll(bits) - 1;
            int j = w * 64 + bit;
            if (j >= n) break;
            int32_t bcrt = fmin[j] - d_r_min[j];
            int32_t wcrt = fmax[j] - d_r_min[j];
            atomicMin(&d_BCRT[j], bcrt);
            atomicMax(&d_WCRT[j], wcrt);
            bits &= bits - 1;
        }
    }
}

// ===========================================================================
// MaxPopcountKernel: compute max popcount(D) across all states
// Used by Phase 5 GPU persistence to avoid D2H for candidate reduction
// ===========================================================================
__global__ void MaxPopcountKernel(
    const char* __restrict__ d_states,
    SAGStateLayout layout,
    int num_states, int W,
    int* __restrict__ d_max_popcount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    const uint64_t* D = layout.D(d_states, tid);
    int pc = 0;
    for (int w = 0; w < W; w++) pc += __popcll(D[w]);

    atomicMax(d_max_popcount, pc);
}
