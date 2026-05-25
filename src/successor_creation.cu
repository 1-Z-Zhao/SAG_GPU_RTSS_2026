// successor_creation.cu -- Successor creation kernel (Phase 3)
//
// Each thread processes one ValidPair and produces one successor state.
// Grid: ceil(num_valid / SUCCESSOR_BLOCK_SIZE) blocks
// Block: SUCCESSOR_BLOCK_SIZE threads
//
// Multi-word bitmask version: supports arbitrary job counts via W-word bitsets.

#include "sag_config.h"
#include "sag_types.h"

using namespace sag;
using namespace sag::config;

// ---------------------------------------------------------------------------
// Multi-word bitmask helpers (local to this TU)
// ---------------------------------------------------------------------------

__device__ __forceinline__ int ctz64_s(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ int popcnt64_s(uint64_t v) {
    return __popcll(v);
}

// Test if bit b is set in a W-word bitset
__device__ __forceinline__ bool bit_test_s(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

// Set bit b in a W-word bitset
__device__ __forceinline__ void bit_set_s(uint64_t* bitset, int b) {
    bitset[b / 64] |= (1ULL << (b % 64));
}

// Clear bit b in a W-word bitset
__device__ __forceinline__ void bit_clear_s(uint64_t* bitset, int b) {
    bitset[b / 64] &= ~(1ULL << (b % 64));
}

// Test if (mask & ~set) == 0 over W words (mask is subset of set)
__device__ __forceinline__ bool is_subset_s(const uint64_t* mask,
                                            const uint64_t* set, int W) {
    for (int w = 0; w < W; w++) {
        if (mask[w] & ~set[w]) return false;
    }
    return true;
}

// Test if bitset is all-zero over W words
__device__ __forceinline__ bool is_zero_s(const uint64_t* bitset, int W) {
    for (int w = 0; w < W; w++) {
        if (bitset[w]) return false;
    }
    return true;
}

// popcount across W words of (a & b)
__device__ __forceinline__ int popcount_and_s(const uint64_t* a,
                                              const uint64_t* b, int W) {
    int count = 0;
    for (int w = 0; w < W; w++) {
        count += __popcll(a[w] & b[w]);
    }
    return count;
}


// ===========================================================================
// Warp-cooperative successor creation kernel
// 1 warp (32 threads) processes ONE ValidPair cooperatively.
// Block: K2_WARPS_PER_BLOCK warps = K2_BLOCK_SIZE threads
// Grid: ceil(num_valid / K2_WARPS_PER_BLOCK) blocks
// ===========================================================================

__global__ void CreateSuccessorsKernel(
    const char*      d_input,
    SAGStateLayout   layout,
    const ValidPair* d_valid_pairs,
    const int*       d_num_valid_ptr,
    const uint64_t*  d_Pred,
    const uint64_t*  d_Succ,
    const int32_t*   d_C_min,
    const int32_t*   d_C_max,
    const int32_t*   d_deadline,
    int n, int m, int W,
    char*            d_output,
    int*             d_output_count,
    int*             d_unschedulable_flag,
    int max_output_states,
    int32_t*         d_BCRT,       // [n], atomicMin: BCRT[j] = min(f_min_j - r_min_j)
    int32_t*         d_WCRT,       // [n], atomicMax: WCRT[j] = max(f_max_j - r_max_j)
    const int32_t*   d_r_min,      // [n], release times for response time calc
    int*             d_trunc_flag) // set to 1 on output buffer overflow
{
    const int lane        = threadIdx.x % WARP_SIZE;
    const int warpInBlock = threadIdx.x / WARP_SIZE;
    const int globalWarp  = blockIdx.x * K2_WARPS_PER_BLOCK + warpInBlock;

    const int num_valid = *d_num_valid_ptr;
    if (globalWarp >= num_valid) return;

    // --- Shared memory layout per warp ---
    // D_new[W] | X_new[W] | F_mask_new[W] | A_min_new[m] | A_max_new[m]
    extern __shared__ char smem_k2[];
    const int smem_per_warp = 3 * W * (int)sizeof(uint64_t) + 2 * m * (int)sizeof(int32_t);
    char* my_smem = smem_k2 + warpInBlock * smem_per_warp;

    uint64_t* s_D_new      = reinterpret_cast<uint64_t*>(my_smem);
    uint64_t* s_X_new      = s_D_new + W;
    uint64_t* s_F_mask_new = s_X_new + W;
    int32_t*  s_A_min_new  = reinterpret_cast<int32_t*>(s_F_mask_new + W);
    int32_t*  s_A_max_new  = s_A_min_new + m;

    // === STEP 1: Load ValidPair (lane 0 loads, broadcast via shfl) ===
    int si, j;
    int32_t s_min_val, s_max_val;
    if (lane == 0) {
        ValidPair vp = d_valid_pairs[globalWarp];
        si        = vp.state_idx;
        j         = vp.job_j;
        s_min_val = vp.s_min;
        s_max_val = vp.s_max;
    }
    si        = __shfl_sync(0xFFFFFFFF, si, 0);
    j         = __shfl_sync(0xFFFFFFFF, j, 0);
    s_min_val = __shfl_sync(0xFFFFFFFF, s_min_val, 0);
    s_max_val = __shfl_sync(0xFFFFFFFF, s_max_val, 0);

    // Parent state field pointers
    const uint64_t* par_D      = layout.D(d_input, si);
    const uint64_t* par_X      = layout.X(d_input, si);
    const uint64_t* par_F_mask = layout.F_mask(d_input, si);
    const int32_t*  par_A_min  = layout.A_min(d_input, si);
    const int32_t*  par_A_max  = layout.A_max(d_input, si);
    const int32_t*  par_F_min  = layout.F_min(d_input, si);
    const int32_t*  par_F_max  = layout.F_max(d_input, si);
    const int32_t*  par_ovf    = layout.ovf(d_input, si);

    // === STEP 2: Compute f_min, f_max, deadline check ===
    int32_t f_min = s_min_val + d_C_min[j];
    int32_t f_max = s_max_val + d_C_max[j];

    // f_min > deadline: every concrete schedule that dispatches j from this
    // reachable parent state misses → UNSCHEDULABLE witness, no successor.
    // f_max > deadline: at least one concrete schedule misses → witness, but
    // still create the successor so other branches see consistent state.
    if (f_min > d_deadline[j]) {
        if (lane == 0) atomicExch(d_unschedulable_flag, 1);
        return;
    }
    int ovf_witness = (f_max > d_deadline[j]) ? 1 : 0;
    if (ovf_witness && lane == 0) {
        atomicExch(d_unschedulable_flag, 1);
    }

    // BCRT/WCRT tracking (lane 0 only, atomic across all warps/blocks)
    if (lane == 0 && d_BCRT != nullptr) {
        atomicMin(&d_BCRT[j], f_min - d_r_min[j]);
        atomicMax(&d_WCRT[j], f_max - d_r_min[j]);
    }

    // === STEP 3: Cooperative D' = D | bit(j) into shared memory ===
    if (lane < W) {
        uint64_t dw = par_D[lane];
        if (lane == (j / 64)) dw |= (1ULL << (j % 64));
        s_D_new[lane] = dw;
    }
    __syncwarp(0xFFFFFFFF);

    // === STEP 4: Cooperative F_mask' with garbage collection ===
    // Load parent F_mask
    if (lane < W) {
        s_F_mask_new[lane] = par_F_mask[lane];
    }
    __syncwarp(0xFFFFFFFF);

    // Check if Succ[j] is subset of D' (warp-wide check)
    {
        const uint64_t* succ_j = &d_Succ[j * W];
        bool local_ok = true;
        if (lane < W) {
            local_ok = ((succ_j[lane] & ~s_D_new[lane]) == 0);
        }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, local_ok || lane >= W);
        if (ballot != 0xFFFFFFFF) {
            // Not subset: add j to F_mask
            if (lane == (j / 64)) {
                s_F_mask_new[lane] |= (1ULL << (j % 64));
            }
        }
    }
    __syncwarp(0xFFFFFFFF);

    // Garbage collection: for each F_mask bit, use warp-parallel subset check
    {
        for (int w = 0; w < W; w++) {
            uint64_t fcheck = s_F_mask_new[w];
            while (fcheck) {
                int bit = __ffsll(fcheck) - 1;
                fcheck &= fcheck - 1;
                int k = w * 64 + bit;

                // Warp-parallel subset check: each lane checks one word of d_Succ[k].
                // Note: the empty-set case (sv == 0) is subsumed by the subset test,
                // since 0 & ~D == 0 always holds, so one ballot suffices.
                const uint64_t* succ_k = &d_Succ[k * W];
                bool local_sub = true;
                if (lane < W) {
                    uint64_t sv = succ_k[lane];
                    if (sv & ~s_D_new[lane]) local_sub = false;
                }
                unsigned sub_ballot = __ballot_sync(0xFFFFFFFF, local_sub || lane >= W);
                bool is_sub = (sub_ballot == 0xFFFFFFFF);

                if (is_sub) {
                    if (lane == 0) {
                        s_F_mask_new[k / 64] &= ~(1ULL << (k % 64));
                    }
                }
                __syncwarp(0xFFFFFFFF);
            }
        }
    }

    // === STEP 5: Cooperative X' computation ===
    // X' = {j} union {k in X_parent : k not in Pred(j) AND par_F_min[k] > s_max}
    if (lane < W) {
        s_X_new[lane] = (lane == (j / 64)) ? (1ULL << (j % 64)) : 0ULL;
    }
    __syncwarp(0xFFFFFFFF);

    {
        const uint64_t* preds_j = &d_Pred[j * W];
        // X' bits only come from parent X, which is typically sparse (10-50 bits).
        // Distribute across lanes with stride-32.
        int bit_idx = 0;
        for (int w = 0; w < W; w++) {
            uint64_t xbits = par_X[w];
            while (xbits) {
                int bit = __ffsll(xbits) - 1;
                xbits &= xbits - 1;
                int k = w * 64 + bit;

                if (bit_idx % WARP_SIZE == lane) {
                    if (!bit_test_s(preds_j, k) && par_F_min[k] > s_max_val) {
                        atomicOr(
                            reinterpret_cast<unsigned long long*>(&s_X_new[k / 64]),
                            1ULL << (k % 64));
                    }
                }
                bit_idx++;
            }
        }
    }
    __syncwarp(0xFFFFFFFF);

    // === STEP 6: Compute p = popcount(par_X & Pred[j]) ===
    int p;
    {
        const uint64_t* preds_j = &d_Pred[j * W];
        int local_p = 0;
        if (lane < W) {
            local_p = __popcll(par_X[lane] & preds_j[lane]);
        }
        // Warp-wide sum via shuffle
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_p += __shfl_xor_sync(0xFFFFFFFF, local_p, offset);
        }
        p = local_p;
    }

    // === STEP 7: Core availability update (nptest insertion algorithm) ===
    // Load parent A_min/A_max into shared memory
    for (int i = lane; i < m; i += WARP_SIZE) {
        s_A_min_new[i] = par_A_min[i];
        s_A_max_new[i] = par_A_max[i];
    }
    __syncwarp(0xFFFFFFFF);

    // Lane 0 performs the insertion-based update (sequential dependency)
    // For single-core jobs: ncores_used=1, ncores_freed=0
    // Inserts eft/lft into sorted sequence, skipping core 0 (used by job)
    if (lane == 0) {
        // Read parent values into local arrays before overwriting
        // Use s_A_max_new upper half as scratch (m <= MAX_CORES/2 typical)
        // Actually, just use the shared mem directly with careful indexing
        int32_t pa_idx = 0, ca_idx = 0;
        bool eft_added = false, lft_added = false;
        int32_t est = s_min_val;
        int32_t eft_val = f_min;
        int32_t lft_val = f_max;

        // We need parent values but s_A_min_new/s_A_max_new will be overwritten.
        // Read parent into registers first (max m=64, fits in registers for lane 0)
        int32_t par_amin_local[MAX_CORES];
        int32_t par_amax_local[MAX_CORES];
        for (int i = 0; i < m; i++) {
            par_amin_local[i] = s_A_min_new[i];
            par_amax_local[i] = s_A_max_new[i];
        }

        // Build new A_min (pa) by insertion
        for (int i = 1; i < m; i++) {
            if (!eft_added && eft_val < par_amin_local[i]) {
                s_A_min_new[pa_idx++] = eft_val;
                eft_added = true;
            }
            s_A_min_new[pa_idx++] = max(est, par_amin_local[i]);
        }
        if (!eft_added) s_A_min_new[pa_idx++] = eft_val;

        // Build new A_max (ca) by insertion
        for (int i = 1; i < m; i++) {
            if (!lft_added && lft_val < par_amax_local[i]) {
                s_A_max_new[ca_idx++] = lft_val;
                lft_added = true;
            }
            s_A_max_new[ca_idx++] = max(est, par_amax_local[i]);
        }
        if (!lft_added) s_A_max_new[ca_idx++] = lft_val;
    }
    __syncwarp(0xFFFFFFFF);

    // === STEP 9: Reserve output slot and cooperative write ===
    int out_idx;
    if (lane == 0) {
        out_idx = atomicAdd(d_output_count, 1);
    }
    out_idx = __shfl_sync(0xFFFFFFFF, out_idx, 0);

    if (out_idx >= max_output_states) {
        if (lane == 0) atomicExch(d_trunc_flag, 1);
        return;
    }

    uint64_t* out_D      = layout.D(d_output, out_idx);
    uint64_t* out_X      = layout.X(d_output, out_idx);
    uint64_t* out_F_mask = layout.F_mask(d_output, out_idx);
    int32_t*  out_A_min  = layout.A_min(d_output, out_idx);
    int32_t*  out_A_max  = layout.A_max(d_output, out_idx);
    int32_t*  out_F_min  = layout.F_min(d_output, out_idx);
    int32_t*  out_F_max  = layout.F_max(d_output, out_idx);
    int32_t*  out_ovf    = layout.ovf(d_output, out_idx);

    // Write bitmask fields from shared memory
    if (lane < W) {
        out_D[lane]      = s_D_new[lane];
        out_X[lane]      = s_X_new[lane];
        out_F_mask[lane] = s_F_mask_new[lane];
    }

    // Write A_min/A_max from shared memory (32 lanes, 2 elements each for m=64)
    for (int i = lane; i < m; i += WARP_SIZE) {
        out_A_min[i] = s_A_min_new[i];
        out_A_max[i] = s_A_max_new[i];
    }

    // Cooperative F_min/F_max copy with int2 vectorization
    if ((n & 1) == 0) {
        const int2* src_fmin = reinterpret_cast<const int2*>(par_F_min);
        int2* dst_fmin       = reinterpret_cast<int2*>(out_F_min);
        int n2 = n / 2;
        for (int i = lane; i < n2; i += WARP_SIZE) {
            dst_fmin[i] = src_fmin[i];
        }

        const int2* src_fmax = reinterpret_cast<const int2*>(par_F_max);
        int2* dst_fmax       = reinterpret_cast<int2*>(out_F_max);
        for (int i = lane; i < n2; i += WARP_SIZE) {
            dst_fmax[i] = src_fmax[i];
        }
    } else {
        for (int k = lane; k < n; k += WARP_SIZE) {
            out_F_min[k] = par_F_min[k];
            out_F_max[k] = par_F_max[k];
        }
    }

    // Set job j's finish times and copy ovf
    if (lane == 0) {
        out_F_min[j] = f_min;
        out_F_max[j] = f_max;
        *out_ovf = *par_ovf | ovf_witness;
    }
}
