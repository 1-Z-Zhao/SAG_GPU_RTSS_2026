// expand_test_main.cu -- Test harness for GPU SAG expansion
//
// Supports three modes:
//   1. No arguments:    hardcoded 5-job test scenario (original behavior)
//   2. Two arguments:   load from CSV files (jobs.csv prec.csv), 1 initial state
//   3. Three arguments: load from CSV + binary states (jobs.csv prec.csv states.bin)
//
// Multi-word bitmask version: uses SAGStateLayout for dynamically-sized states.

#include "sag_config.h"
#include "sag_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <dirent.h>
#include <sys/stat.h>
#include <omp.h>
#include <cub/cub.cuh>
#include <numeric>
#include <parallel/algorithm>  // __gnu_parallel::sort

// ---------------------------------------------------------------------------
// Batch-mode helpers (Step 3): one-taskset result container
// ---------------------------------------------------------------------------
struct SolveResult {
    bool        ok = false;
    bool        schedulable = false;
    double      wall_time_s = 0.0;
    double      gpu_time_ms = 0.0;
    long long   total_nodes = 0;
    long long   total_expanded = 0;
    long long   total_merged = 0;
    int         jobs_tracked = 0;
    int         N = 0;
    int         W = 0;
    int         M = 0;
    std::string error;
};

using namespace sag;
using namespace sag::config;

// Thread-local quiet flag: when set, run_one_taskset suppresses its verbose
// per-layer / per-wave / setup prints so that parallel batch mode doesn't
// drown stdout with interleaved output (and pays the printf cost).
static thread_local bool g_quiet = false;
#define QPRINTF(...) do { if (!g_quiet) { printf(__VA_ARGS__); } } while(0)

// OPT: parallel-batch memory partitioning.
// When run_batch runs multiple tasksets concurrently via OpenMP, each worker
// would otherwise try to claim ~85% of free GPU memory for its own wave buffers
// and layer persistence buffers, leading to OOM. Set this to the parallel
// width so each worker caps its per-taskset GPU footprint.
static int g_batch_par_n = 1;

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s  %s:%d\n",
                cudaGetErrorString(code), file, line);
        exit(1);
    }
}

// Route every cudaMalloc/cudaFree through the stream-ordered allocator so
// that freed memory stays cached in the default memory pool and the next
// run_one_taskset reuses it. Paired with cudaMemPoolAttrReleaseThreshold =
// UINT64_MAX set once in main(), this turns per-taskset cudaMalloc churn
// from ~170 ms into a near-zero pool handoff.
#define cudaMalloc(ptr, size) cudaMallocAsync((ptr), (size), (cudaStream_t)0)
#define cudaFree(ptr)         cudaFreeAsync((ptr), (cudaStream_t)0)

// ---------------------------------------------------------------------------
// External kernel declarations (multi-word bitmask versions)
// ---------------------------------------------------------------------------
extern __global__ void FusedEligibilityKernel(
    const char*     d_input,
    SAGStateLayout  layout,
    const int*      d_candidates,
    int num_states,
    int num_candidates,
    const uint64_t* d_TC,
    const uint64_t* d_PO,
    const uint64_t* d_Pred,
    const int32_t*  d_r_min,
    const int32_t*  d_r_max,
    const int32_t*  d_priority,
    const int32_t*  d_sus_min,
    const int32_t*  d_sus_max,
    int n, int W,
    ValidPair*      d_valid_pairs,
    int*            d_valid_count,
    int*            d_unschedulable_flag,
    int max_valid_pairs,
    int*            d_trunc_flag);

extern __global__ void CreateSuccessorsKernel(
    const char*      d_input,
    SAGStateLayout   layout,
    const ValidPair* d_valid_pairs,
    const int*       d_num_valid_ptr,  // Phase 4: device pointer, no CPU readback needed
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
    int32_t*         d_BCRT,
    int32_t*         d_WCRT,
    const int32_t*   d_r_min,
    int*             d_trunc_flag);


// ---------------------------------------------------------------------------
// External merge kernel declarations
// ---------------------------------------------------------------------------
extern __global__ void ExtractDKeysKernel(
    const char* __restrict__ d_output,
    SAGStateLayout layout,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys);

extern __global__ void IotaKernel(int* d_indices, int count);

// Phase 4: Fused kernel
extern __global__ void ExtractDKeysAndIotaKernel(
    const char* __restrict__ d_output,
    SAGStateLayout layout,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys,
    int* __restrict__ d_indices);

// Phase 1b: Device-pointer version
extern __global__ void ComputeGroupSizesKernelDev(
    const int* __restrict__ d_group_starts,
    const int* __restrict__ d_num_groups_ptr,
    int num_states,
    int* __restrict__ d_group_sizes);

// Phase 2: Warp-cooperative merge
extern __global__ void FusedRangeMergeKernelWarp(
    const char* __restrict__ d_expanded,
    const int*  __restrict__ d_sorted_idx,
    const int*  __restrict__ d_group_starts,
    const int*  __restrict__ d_num_groups_ptr,
    int         num_states,
    SAGStateLayout layout,
    int n, int m, int W,
    char*       d_merge_buf,
    int*        d_merge_count);

extern __global__ void GatherDKeysKernel(
    const uint64_t* __restrict__ d_D_keys_unsorted,
    const int*      __restrict__ d_sorted_indices,
    int num_states, int W,
    uint64_t* __restrict__ d_D_keys_sorted);

extern __global__ void DetectBoundariesKernel(
    const uint64_t* __restrict__ d_D_keys_sorted,
    int num_states, int W,
    int* __restrict__ d_is_start);

extern __global__ void CompactGroupStartsKernel(
    const int* __restrict__ d_is_start,
    const int* __restrict__ d_group_id,
    int num_states,
    int* __restrict__ d_group_starts,
    int* __restrict__ d_num_groups);

extern __global__ void ComputeGroupSizesKernel(
    const int* __restrict__ d_group_starts,
    int num_groups, int num_states,
    int* __restrict__ d_group_sizes);

extern __global__ void FusedRangeMergeKernel(
    const char* __restrict__ d_expanded,
    const int*  __restrict__ d_sorted_idx,
    const int*  __restrict__ d_group_starts,
    const int*  __restrict__ d_group_sizes,
    int         num_groups,
    SAGStateLayout layout,
    int n, int m, int W,
    char*       d_merge_buf,
    int*        d_merge_count);

extern __global__ void UpdateBCRTWCRTFromMergedKernel(
    const char* __restrict__ d_merged_buf,
    SAGStateLayout layout,
    int num_merged,
    int n, int W,
    const int32_t* __restrict__ d_r_min,
    int32_t* d_BCRT,
    int32_t* d_WCRT);

extern __global__ void MaxPopcountKernel(
    const char* __restrict__ d_states,
    SAGStateLayout layout,
    int num_states, int W,
    int* __restrict__ d_max_popcount);

// ---------------------------------------------------------------------------
// Host-side merge helpers (CPU path)
// ---------------------------------------------------------------------------
static bool host_check_range_compatible(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayout layout, int n, int m, int W,
    bool useJobFinishTimes = false)
{
    const int32_t* a_Amin = layout.A_min(buf_a, idx_a);
    const int32_t* a_Amax = layout.A_max(buf_a, idx_a);
    const int32_t* b_Amin = layout.A_min(buf_b, idx_b);
    const int32_t* b_Amax = layout.A_max(buf_b, idx_b);

    for (int i = 0; i < m; i++) {
        if (a_Amin[i] > b_Amax[i] + 1) return false;
        if (b_Amin[i] > a_Amax[i] + 1) return false;
    }

    if (useJobFinishTimes) {
        const uint64_t* fm = layout.F_mask(buf_b, idx_b);
        const int32_t* a_Fmin = layout.F_min(buf_a, idx_a);
        const int32_t* a_Fmax = layout.F_max(buf_a, idx_a);
        const int32_t* b_Fmin = layout.F_min(buf_b, idx_b);
        const int32_t* b_Fmax = layout.F_max(buf_b, idx_b);

        for (int w = 0; w < W; w++) {
            uint64_t bits = fm[w];
            while (bits) {
                int bit = __builtin_ctzll(bits);
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

static void host_merge_into_slot(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayout layout, int n, int m, int W)
{
    const uint64_t* sX = layout.X(src_buf, src_idx);
    uint64_t*       dX = layout.X(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dX[w] &= sX[w];

    const int32_t* sAmin = layout.A_min(src_buf, src_idx);
    int32_t*       dAmin = layout.A_min(dst_buf, dst_idx);
    const int32_t* sAmax = layout.A_max(src_buf, src_idx);
    int32_t*       dAmax = layout.A_max(dst_buf, dst_idx);
    for (int i = 0; i < m; i++) {
        if (sAmin[i] < dAmin[i]) dAmin[i] = sAmin[i];
        if (sAmax[i] > dAmax[i]) dAmax[i] = sAmax[i];
    }

    // Iterate over (src.F_mask UNION dst.F_mask). Bits only in src must be
    // copied (their dst slot is stale); bits in both get min/max merged.
    const uint64_t* sFmask = layout.F_mask(src_buf, src_idx);
    const uint64_t* dFmask = layout.F_mask(dst_buf, dst_idx);
    const int32_t* sFmin = layout.F_min(src_buf, src_idx);
    int32_t*       dFmin = layout.F_min(dst_buf, dst_idx);
    const int32_t* sFmax = layout.F_max(src_buf, src_idx);
    int32_t*       dFmax = layout.F_max(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) {
        uint64_t bits = sFmask[w] | dFmask[w];
        while (bits) {
            int bit = __builtin_ctzll(bits);
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

    uint64_t* dF = layout.F_mask(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dF[w] |= sFmask[w];
    *layout.ovf(dst_buf, dst_idx) |= *layout.ovf(src_buf, src_idx);
}

static int cpu_merge_groups(
    const char* h_expanded,
    const int*  h_sorted_indices,
    const std::vector<int>& group_starts,
    const std::vector<int>& group_sizes,
    const std::vector<int>& cpu_group_ids,
    SAGStateLayout layout,
    int n, int m, int W, int bps,
    char* h_merge_out)
{
    int out_count = 0;

    for (int g = 0; g < (int)cpu_group_ids.size(); g++) {
        int gid    = cpu_group_ids[g];
        int gstart = group_starts[gid];
        int gsize  = group_sizes[gid];

        if (gsize == 1) {
            int orig_idx = h_sorted_indices[gstart];
            memcpy(h_merge_out + (long long)out_count * bps,
                   h_expanded  + (long long)orig_idx  * bps,
                   bps);
            out_count++;
            continue;
        }

        int local_slots[MAX_SLOTS_PER_GROUP];
        int num_slots = 0;

        for (int s = 0; s < gsize; s++) {
            int orig_idx = h_sorted_indices[gstart + s];
            bool merged = false;

            for (int sl = 0; sl < num_slots; sl++) {
                if (host_check_range_compatible(
                        h_expanded,  orig_idx,
                        h_merge_out, local_slots[sl],
                        layout, n, m, W)) {
                    host_merge_into_slot(
                        h_expanded,  orig_idx,
                        h_merge_out, local_slots[sl],
                        layout, n, m, W);
                    merged = true;
                    break;
                }
            }

            if (!merged) {
                int slot_idx = out_count++;
                memcpy(h_merge_out + (long long)slot_idx * bps,
                       h_expanded  + (long long)orig_idx * bps,
                       bps);
                if (num_slots < MAX_SLOTS_PER_GROUP)
                    local_slots[num_slots++] = slot_idx;
            }
        }
    }
    return out_count;
}

// Forward declarations for CPU fast-path
static void detect_groups_and_classify(
    const uint64_t* h_D_keys_sorted, int num_states, int W, int cpu_threshold,
    std::vector<int>& group_starts, std::vector<int>& group_sizes,
    std::vector<int>& cpu_group_ids, std::vector<int>& gpu_group_ids);
static int cpu_merge_groups(
    const char* h_expanded, const int* h_sorted_indices,
    const std::vector<int>& group_starts, const std::vector<int>& group_sizes,
    const std::vector<int>& cpu_group_ids,
    SAGStateLayout layout, int n, int m, int W, int bps, char* h_merge_out);

// ===========================================================================
// CPU fast-path for small layers (avoids GPU kernel launch overhead)
// ===========================================================================
static const int CPU_LAYER_THRESHOLD = 256;  // layers with fewer states use CPU

static int cpu_expand_and_merge_layer(
    const char* layer_in, int num_states,
    char* layer_out, int max_out_states,
    SAGStateLayout layout,
    const std::vector<int>& candidates, int num_candidates,
    const std::vector<uint64_t>& v_TC,
    const std::vector<uint64_t>& v_PO,
    const std::vector<uint64_t>& v_Pred,
    const std::vector<uint64_t>& v_Succ,
    const std::vector<int32_t>& v_r_min,
    const std::vector<int32_t>& v_r_max,
    const std::vector<int32_t>& v_C_min,
    const std::vector<int32_t>& v_C_max,
    const std::vector<int32_t>& v_deadline,
    const std::vector<int32_t>& v_priority,
    const std::vector<int32_t>& v_sus_min,
    const std::vector<int32_t>& v_sus_max,
    const std::vector<int>& topo_depth,
    int N, int M, int W, int bps,
    std::vector<int32_t>& h_BCRT,
    std::vector<int32_t>& h_WCRT,
    long long& out_expanded, long long& out_valid,
    int& out_unschedulable,
    int& out_truncated)
{
    out_unschedulable = 0;
    out_truncated = 0;
    // Phase 7: OpenMP-parallelized successor generation + sequential merge.
    //
    // Phase A (parallel): Each thread processes a subset of input states,
    //   generating successors into thread-local storage. No shared writes.
    // Phase B (sequential): Collect all successors and apply incremental
    //   dkey_index merge (same logic as before).
    //
    // Only use multiple threads when num_states is large enough to amortize
    // the OpenMP fork-join overhead (~10us per region).

    const int32_t INF_TIME_LOCAL = INT32_MAX / 2;
    // Phase 6f: lowered from 99999 → 64. Above ~64 states the two-phase
    // OMP fork-join cost is amortized; below 64 the serial path is faster.
    static const int OMP_MIN_STATES = 64;

    // --- Temporary successor record for thread-local storage ---
    struct TempSuccessor {
        std::vector<char> data;   // full bps-sized state
        std::string dkey;         // D-key bytes for merge lookup
        int job_j;
        int32_t f_min, f_max;
    };

    int nthreads = (num_states >= OMP_MIN_STATES) ? omp_get_max_threads() : 1;

    // Per-thread accumulators
    std::vector<std::vector<TempSuccessor>> per_thread_succs(nthreads);
    std::vector<int32_t> per_thread_bcrt(nthreads * N, INT32_MAX);
    std::vector<int32_t> per_thread_wcrt(nthreads * N, 0);
    std::vector<long long> per_thread_expanded(nthreads, 0);
    std::vector<long long> per_thread_valid(nthreads, 0);
    std::vector<int> per_thread_unsched(nthreads, 0);

    // =====================================================================
    // PHASE A: Parallel successor generation (OpenMP parallel for)
    // =====================================================================
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int32_t* my_bcrt = per_thread_bcrt.data() + tid * N;
        int32_t* my_wcrt = per_thread_wcrt.data() + tid * N;
        long long my_expanded = 0;
        long long my_valid = 0;

        // Thread-local reusable buffers
        std::vector<char> temp_state(bps, 0);
        std::vector<bool> in_ready(N, false);
        std::vector<int32_t> rho_max_arr(N, INF_TIME_LOCAL);

        #pragma omp for schedule(dynamic, 1)
        for (int si = 0; si < num_states; si++) {
            const uint64_t* D = layout.D(layer_in, si);
            const uint64_t* X = layout.X(layer_in, si);
            const int32_t* st_A_min = layout.A_min(layer_in, si);
            const int32_t* st_A_max = layout.A_max(layer_in, si);
            const int32_t* st_F_min = layout.F_min(layer_in, si);
            const int32_t* st_F_max = layout.F_max(layer_in, si);
            const uint64_t* st_F_mask = layout.F_mask(layer_in, si);

            int d_popcount = 0;
            for (int w = 0; w < W; w++) d_popcount += __builtin_popcountll(D[w]);

            // Reset thread-local ready set
            std::fill(in_ready.begin(), in_ready.end(), false);
            std::fill(rho_max_arr.begin(), rho_max_arr.end(), INF_TIME_LOCAL);
            int32_t rho_any = INF_TIME_LOCAL;

            // Sub-Phase 1: Build ready set R and rho_max for each member
            for (int ci = 0; ci < num_candidates; ci++) {
                int c = candidates[ci];
                if (!topo_depth.empty() && topo_depth[c] > d_popcount + 1) break;
                if (D[c / 64] & (1ULL << (c % 64))) continue;

                bool pred_ok = true;
                for (int w = 0; w < W; w++) {
                    if ((v_Pred[c * W + w] & D[w]) != v_Pred[c * W + w]) {
                        pred_ok = false; break;
                    }
                }
                if (!pred_ok) continue;

                int32_t rho_max_c = v_r_max[c];
                for (int w = 0; w < W; w++) {
                    uint64_t pbits = v_Pred[c * W + w];
                    while (pbits) {
                        int bit = __builtin_ctzll(pbits);
                        int i = w * 64 + bit;
                        if (i < N && (st_F_mask[i / 64] & (1ULL << (i % 64)))) {
                            int32_t val = st_F_max[i] + v_sus_max[c * N + i];
                            if (val > rho_max_c) rho_max_c = val;
                        }
                        pbits &= pbits - 1;
                    }
                }
                rho_max_arr[c] = rho_max_c;
                in_ready[c] = true;
                if (rho_max_c < rho_any) rho_any = rho_max_c;
            }

            // Sub-Phase 2: Evaluate eligibility for each candidate
            for (int ci = 0; ci < num_candidates; ci++) {
                int j = candidates[ci];
                if (!topo_depth.empty() && topo_depth[j] > d_popcount + 1) break;
                if (D[j / 64] & (1ULL << (j % 64))) continue;

                bool tc_ok = true;
                for (int w = 0; w < W; w++) {
                    if ((v_TC[j * W + w] & D[w]) != v_TC[j * W + w]) {
                        tc_ok = false; break;
                    }
                }
                if (!tc_ok) continue;

                bool po_ok = true;
                for (int w = 0; w < W; w++) {
                    if ((v_PO[j * W + w] & D[w]) != v_PO[j * W + w]) {
                        po_ok = false; break;
                    }
                }
                if (!po_ok) continue;

                if (!in_ready[j]) continue;

                int32_t rho_min_j = v_r_min[j];
                for (int w = 0; w < W; w++) {
                    uint64_t pbits = v_Pred[j * W + w];
                    while (pbits) {
                        int bit = __builtin_ctzll(pbits);
                        int i = w * 64 + bit;
                        if (i < N && (st_F_mask[i / 64] & (1ULL << (i % 64)))) {
                            int32_t val = st_F_min[i] + v_sus_min[j * N + i];
                            if (val > rho_min_j) rho_min_j = val;
                        }
                        pbits &= pbits - 1;
                    }
                }

                int32_t prio_j = v_priority[j];
                int32_t rho_hp_j = INF_TIME_LOCAL;
                for (int h = 0; h < N; h++) {
                    if (!in_ready[h] || h == j) continue;
                    if (v_priority[h] < prio_j) {
                        if (rho_max_arr[h] < rho_hp_j) rho_hp_j = rho_max_arr[h];
                    }
                }

                int32_t s_min = std::max(rho_min_j, st_A_min[0]);
                int32_t s_max = std::min(rho_hp_j - 1, std::max(st_A_max[0], rho_any));


                if (s_min > s_max) continue;

                my_valid++;

                int32_t f_min = s_min + v_C_min[j];
                int32_t f_max = s_max + v_C_max[j];

                if (f_min > v_deadline[j]) {
                    per_thread_unsched[tid] = 1;
                    continue;
                }

                // Copy parent state to temp_state
                memcpy(temp_state.data(), layer_in + (long long)si * bps, bps);

                uint64_t* out_D      = layout.D(temp_state.data(), 0);
                uint64_t* out_X      = layout.X(temp_state.data(), 0);
                uint64_t* out_F_mask = layout.F_mask(temp_state.data(), 0);
                int32_t*  out_F_min  = layout.F_min(temp_state.data(), 0);
                int32_t*  out_F_max  = layout.F_max(temp_state.data(), 0);
                int32_t*  out_A_min  = layout.A_min(temp_state.data(), 0);
                int32_t*  out_A_max  = layout.A_max(temp_state.data(), 0);

                // STEP 3: D' = D | {j}
                out_D[j / 64] |= (1ULL << (j % 64));

                // STEP 4: F_mask GC + add j
                bool succ_j_subset = true;
                for (int w = 0; w < W; w++) {
                    if (v_Succ[j * W + w] & ~out_D[w]) {
                        succ_j_subset = false; break;
                    }
                }
                if (!succ_j_subset) {
                    out_F_mask[j / 64] |= (1ULL << (j % 64));
                }

                for (int w = 0; w < W; w++) {
                    uint64_t bits = out_F_mask[w];
                    while (bits) {
                        int bit = __builtin_ctzll(bits);
                        int k = w * 64 + bit;
                        if (k < N) {
                            bool all_zero = true;
                            bool is_sub = true;
                            for (int ww = 0; ww < W; ww++) {
                                uint64_t sv = v_Succ[k * W + ww];
                                if (sv != 0) all_zero = false;
                                if (sv & ~out_D[ww]) is_sub = false;
                            }
                            if (all_zero || is_sub) {
                                out_F_mask[w] &= ~(1ULL << bit);
                            }
                        }
                        bits &= bits - 1;
                    }
                }

                out_F_min[j] = f_min;
                out_F_max[j] = f_max;

                // STEP 5: X'
                for (int w = 0; w < W; w++) out_X[w] = 0;
                out_X[j / 64] |= (1ULL << (j % 64));
                for (int w = 0; w < W; w++) {
                    uint64_t xbits = X[w];
                    while (xbits) {
                        int bit = __builtin_ctzll(xbits);
                        int k = w * 64 + bit;
                        if (k < N) {
                            bool in_pred_j = (v_Pred[j * W + k/64] >> (k%64)) & 1;
                            if (!in_pred_j && st_F_min[k] > s_max) {
                                out_X[k / 64] |= (1ULL << (k % 64));
                            }
                        }
                        xbits &= xbits - 1;
                    }
                }

                // STEP 6+7: Core availability update
                {
                    int32_t pa[MAX_CORES], ca[MAX_CORES];
                    int pa_idx = 0, ca_idx = 0;
                    bool eft_added = false, lft_added = false;
                    int32_t est = s_min;
                    int32_t eft = f_min;
                    int32_t lft = f_max;

                    for (int i = 1; i < M; i++) {
                        if (!eft_added && eft < out_A_min[i]) {
                            pa[pa_idx++] = eft;
                            eft_added = true;
                        }
                        pa[pa_idx++] = std::max(est, out_A_min[i]);

                        if (!lft_added && lft < out_A_max[i]) {
                            ca[ca_idx++] = lft;
                            lft_added = true;
                        }
                        ca[ca_idx++] = std::max(est, out_A_max[i]);
                    }
                    if (!eft_added) pa[pa_idx++] = eft;
                    if (!lft_added) ca[ca_idx++] = lft;

                    for (int i = 0; i < M; i++) {
                        out_A_min[i] = pa[i];
                        out_A_max[i] = ca[i];
                    }
                }

                if (f_max > v_deadline[j]) {
                    // Witness: a concrete schedule in this reachable state misses j's deadline.
                    *layout.ovf(temp_state.data(), 0) = 1;
                    per_thread_unsched[tid] = 1;
                }

                // Thread-local BCRT/WCRT update
                int32_t bcrt = f_min - v_r_min[j];
                int32_t wcrt = f_max - v_r_min[j];
                if (bcrt < my_bcrt[j]) my_bcrt[j] = bcrt;
                if (wcrt > my_wcrt[j]) my_wcrt[j] = wcrt;

                my_expanded++;

                // Store successor for Phase B merge
                TempSuccessor ts;
                ts.data.assign(temp_state.data(), temp_state.data() + bps);
                ts.dkey.assign((const char*)out_D, W * sizeof(uint64_t));
                ts.job_j = j;
                ts.f_min = f_min;
                ts.f_max = f_max;
                per_thread_succs[tid].push_back(std::move(ts));
            }
        }

        per_thread_expanded[tid] = my_expanded;
        per_thread_valid[tid] = my_valid;
    } // end omp parallel

    // =====================================================================
    // Reduce per-thread BCRT/WCRT into h_BCRT/h_WCRT
    // =====================================================================
    long long expand_count = 0;
    long long valid_pairs = 0;
    for (int t = 0; t < nthreads; t++) {
        expand_count += per_thread_expanded[t];
        valid_pairs += per_thread_valid[t];
        if (per_thread_unsched[t]) out_unschedulable = 1;
        for (int j = 0; j < N; j++) {
            int32_t bc = per_thread_bcrt[t * N + j];
            int32_t wc = per_thread_wcrt[t * N + j];
            if (bc < h_BCRT[j]) h_BCRT[j] = bc;
            if (wc > h_WCRT[j]) h_WCRT[j] = wc;
        }
    }

    // =====================================================================
    // PHASE B: Sequential merge of all successors
    // =====================================================================
    struct DKeyHash {
        size_t operator()(const std::string& k) const {
            size_t h = 14695981039346656037ULL;
            for (size_t i = 0; i < k.size(); i++) {
                h ^= (unsigned char)k[i];
                h *= 1099511628211ULL;
            }
            return h;
        }
    };
    std::unordered_map<std::string, std::vector<int>, DKeyHash> dkey_index;

    int out_count = 0;
    for (int t = 0; t < nthreads; t++) {
        for (auto& succ : per_thread_succs[t]) {
            bool merged = false;
            auto it = dkey_index.find(succ.dkey);
            if (it != dkey_index.end()) {
                for (int slot_idx : it->second) {
                    if (host_check_range_compatible(
                            succ.data.data(), 0,
                            layer_out, slot_idx,
                            layout, N, M, W)) {
                        host_merge_into_slot(
                            succ.data.data(), 0,
                            layer_out, slot_idx,
                            layout, N, M, W);
                        merged = true;
                        break;
                    }
                }
            }

            if (!merged) {
                if (out_count >= max_out_states) {
                    out_truncated = 1;  // L1: silent drop converted to truncation signal
                    continue;
                }
                int slot_idx = out_count++;
                memcpy(layer_out + (long long)slot_idx * bps,
                       succ.data.data(), bps);
                dkey_index[succ.dkey].push_back(slot_idx);
            }
        }
    }

    out_expanded = expand_count;
    out_valid = valid_pairs;

    // Update BCRT/WCRT from merged states (post-merge F_min/F_max may be wider)
    for (int s = 0; s < out_count; s++) {
        const uint64_t* fm = layout.F_mask(layer_out, s);
        const int32_t* fmin = layout.F_min(layer_out, s);
        const int32_t* fmax = layout.F_max(layer_out, s);
        for (int w = 0; w < W; w++) {
            uint64_t bits = fm[w];
            while (bits) {
                int bit = __builtin_ctzll(bits);
                int j = w * 64 + bit;
                if (j >= N) break;
                int32_t bcrt_val = fmin[j] - v_r_min[j];
                int32_t wcrt_val = fmax[j] - v_r_min[j];
                if (bcrt_val < h_BCRT[j]) h_BCRT[j] = bcrt_val;
                if (wcrt_val > h_WCRT[j]) h_WCRT[j] = wcrt_val;
                bits &= bits - 1;
            }
        }
    }

    return out_count;
}

static void detect_groups_and_classify(
    const uint64_t* h_D_keys_sorted,
    int num_states, int W,
    int cpu_threshold,
    std::vector<int>& group_starts,
    std::vector<int>& group_sizes,
    std::vector<int>& cpu_group_ids,
    std::vector<int>& gpu_group_ids)
{
    group_starts.clear();
    group_sizes.clear();
    cpu_group_ids.clear();
    gpu_group_ids.clear();

    if (num_states == 0) return;

    group_starts.push_back(0);
    for (int i = 1; i < num_states; i++) {
        bool same = true;
        for (int w = 0; w < W; w++) {
            if (h_D_keys_sorted[i * W + w] != h_D_keys_sorted[(i - 1) * W + w]) {
                same = false;
                break;
            }
        }
        if (!same) {
            group_starts.push_back(i);
        }
    }

    int num_groups = (int)group_starts.size();
    group_sizes.resize(num_groups);
    for (int g = 0; g < num_groups - 1; g++)
        group_sizes[g] = group_starts[g + 1] - group_starts[g];
    group_sizes[num_groups - 1] = num_states - group_starts[num_groups - 1];

    for (int g = 0; g < num_groups; g++) {
        if (group_sizes[g] <= cpu_threshold)
            cpu_group_ids.push_back(g);
        else
            gpu_group_ids.push_back(g);
    }
}

// ===========================================================================
// main
// ===========================================================================
// ---------------------------------------------------------------------------
// run_one_taskset: solve a single (jobs.csv, prec.csv [, states.bin]) problem.
//
// This is the original main() body, wrapped as a function so that batch mode
// can call it multiple times in one process (reusing the CUDA context).
//
// jobs_csv_path / prec_csv_path: CSV file paths. If both are nullptr, fall back
//   to the hardcoded 5-job test scenario (preserved for regression).
// states_bin_path: optional path to states.bin (may be nullptr).
// M_in: cores (already validated by caller).
// result_out: filled with per-run stats on success or error message on failure.
//
// Returns 0 on successful solve (even if UNSCHEDULABLE), 1 on hard error.
// ---------------------------------------------------------------------------
static int run_one_taskset(const char* jobs_csv_path,
                           const char* prec_csv_path,
                           const char* states_bin_path,
                           int M_in,
                           SolveResult* result_out)
{
    // In parallel batch mode g_quiet is set; shadow printf so the tons of
    // per-layer / per-wave logs inside this function become no-ops without
    // having to touch every individual call site.
    #define printf(...) QPRINTF(__VA_ARGS__)
    int N;
    int M = M_in;      // cores (validated by caller)
    int W;
    int NUM_STATES;

    // Vectors for job attributes (used by both modes)
    std::vector<int32_t> v_r_min, v_r_max, v_C_min, v_C_max, v_deadline, v_priority;
    std::vector<int32_t> v_task_id;
    std::vector<uint64_t> v_Pred, v_Succ, v_TC, v_PO;
    std::vector<int32_t> v_sus_min, v_sus_max;
    std::vector<int> v_candidates;
    std::vector<int> topo_depth;  // Phase 1: topological depth per job
    std::vector<int32_t> v_min_path_delay;  // Phase 5: minimum path delay per job

    // Synthesize pos_args[] from function arguments so the original
    // body (which uses pos_args / pos_argc / load_states_file) works
    // unmodified.
    std::vector<char*> pos_args;
    static char prog_name[] = "expand_test";
    pos_args.push_back(prog_name);
    if (jobs_csv_path) pos_args.push_back(const_cast<char*>(jobs_csv_path));
    if (prec_csv_path) pos_args.push_back(const_cast<char*>(prec_csv_path));
    if (states_bin_path) pos_args.push_back(const_cast<char*>(states_bin_path));
    int pos_argc = (int)pos_args.size();

    // Flag for states.bin mode
    bool load_states_file = (pos_argc == 4);

    // Helper macro to bail out early with a clear error message in result_out.
    #define RUN_FAIL(msg) do { \
        if (result_out) { result_out->ok = false; result_out->error = (msg); } \
        return 1; \
    } while(0)

    if (pos_argc == 3 || pos_argc == 4) {
        // =================================================================
        // CSV mode: parse files (+ optional states.bin)
        // =================================================================
        if (load_states_file)
            printf("Loading from CSV + states file:\n  Jobs: %s\n  Prec: %s\n  States: %s\n\n",
                   pos_args[1], pos_args[2], pos_args[3]);
        else
            printf("Loading from CSV files:\n  Jobs: %s\n  Prec: %s\n\n", pos_args[1], pos_args[2]);

        // --- Parse jobs CSV ---
        // Format: job_id, instance_id, r_min, r_max, C_min, C_max, deadline, priority
        FILE* fj = fopen(pos_args[1], "r");
        if (!fj) {
            fprintf(stderr, "Error: cannot open jobs file '%s'\n", pos_args[1]);
            RUN_FAIL(std::string("cannot open jobs file '") + pos_args[1] + "'");
        }

        struct JobRecord {
            int job_id, instance_id;
            int r_min, r_max, C_min, C_max, deadline, priority;
        };
        std::vector<JobRecord> job_records;
        int max_instance_id = -1;

        {
            char line[512];
            while (fgets(line, sizeof(line), fj)) {
                // Skip empty lines
                if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

                JobRecord rec;
                int parsed = sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d",
                    &rec.job_id, &rec.instance_id,
                    &rec.r_min, &rec.r_max,
                    &rec.C_min, &rec.C_max,
                    &rec.deadline, &rec.priority);
                if (parsed != 8) continue;  // skip malformed lines

                job_records.push_back(rec);
                if (rec.instance_id > max_instance_id)
                    max_instance_id = rec.instance_id;
            }
        }
        fclose(fj);

        N = max_instance_id + 1;
        W = bitset_words(N);
        printf("Parsed %zu job records, N=%d (max instance_id=%d), W=%d\n",
               job_records.size(), N, max_instance_id, W);

        // Allocate and zero-fill job attribute arrays
        v_r_min.assign(N, 0);
        v_r_max.assign(N, 0);
        v_C_min.assign(N, 0);
        v_C_max.assign(N, 0);
        v_deadline.assign(N, 0);
        v_priority.assign(N, 0);
        v_task_id.assign(N, 0);

        for (const auto& rec : job_records) {
            int id = rec.instance_id;
            v_r_min[id]    = rec.r_min;
            v_r_max[id]    = rec.r_max;
            v_C_min[id]    = rec.C_min;
            v_C_max[id]    = rec.C_max;
            v_deadline[id] = rec.deadline;
            v_priority[id] = rec.priority;
            v_task_id[id]  = rec.job_id;  // job_id in CSV is actually task_id
        }

        // --- Parse precedence CSV ---
        // Format: pred_job_id, pred_instance_id, succ_job_id, succ_instance_id, sus_min, sus_max, type
        FILE* fp = fopen(pos_args[2], "r");
        if (!fp) {
            fprintf(stderr, "Error: cannot open precedence file '%s'\n", pos_args[2]);
            RUN_FAIL(std::string("cannot open precedence file '") + pos_args[2] + "'");
        }

        // Allocate topology matrices
        v_Pred.assign(N * W, 0ULL);
        v_Succ.assign(N * W, 0ULL);
        v_sus_min.assign(N * N, 0);
        v_sus_max.assign(N * N, 0);

        int edge_count = 0;
        {
            char line[512];
            while (fgets(line, sizeof(line), fp)) {
                if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

                int pred_jid, pred_iid, succ_jid, succ_iid, smin, smax;
                char type_ch;
                int parsed = sscanf(line, "%d,%d,%d,%d,%d,%d,%c",
                    &pred_jid, &pred_iid, &succ_jid, &succ_iid,
                    &smin, &smax, &type_ch);
                if (parsed < 6) continue;  // skip malformed lines

                // Set bit pred_iid in Pred[succ_iid]
                v_Pred[succ_iid * W + (pred_iid / 64)] |= (1ULL << (pred_iid % 64));
                // Set bit succ_iid in Succ[pred_iid]
                v_Succ[pred_iid * W + (succ_iid / 64)] |= (1ULL << (succ_iid % 64));
                // Suspension delays
                v_sus_min[succ_iid * N + pred_iid] = smin;
                v_sus_max[succ_iid * N + pred_iid] = smax;

                edge_count++;
            }
        }
        fclose(fp);
        printf("Parsed %d precedence edges\n", edge_count);

        // --- Build transitive closure TC from Pred ---
        // TC[j] = all ancestors of j (transitive closure of predecessors)
        // Initialize TC = copy of Pred
        v_TC.assign(N * W, 0ULL);
        for (int j = 0; j < N; j++) {
            for (int w = 0; w < W; w++) {
                v_TC[j * W + w] = v_Pred[j * W + w];
            }
        }

        // Iterative fixpoint: repeat until no changes
        for (int iter = 0; iter < N; iter++) {
            bool changed = false;
            for (int j = 0; j < N; j++) {
                for (int pw = 0; pw < W; pw++) {
                    uint64_t pbits = v_Pred[j * W + pw];
                    while (pbits) {
                        int bit = __builtin_ctzll(pbits);
                        pbits &= pbits - 1;
                        int p = pw * 64 + bit;
                        // j inherits all ancestors of p
                        for (int w = 0; w < W; w++) {
                            uint64_t old = v_TC[j * W + w];
                            v_TC[j * W + w] |= v_TC[p * W + w];
                            if (v_TC[j * W + w] != old) changed = true;
                        }
                    }
                }
            }
            if (!changed) break;
        }

        printf("Transitive closure computed\n");

        // --- Phase 1: Compute topological depth (longest path from any root) ---
        topo_depth.assign(N, 0);
        {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int j = 0; j < N; j++) {
                    for (int w = 0; w < W; w++) {
                        uint64_t pbits = v_Pred[j * W + w];
                        while (pbits) {
                            int bit = __builtin_ctzll(pbits);
                            pbits &= pbits - 1;
                            int pred_id = w * 64 + bit;
                            if (topo_depth[pred_id] + 1 > topo_depth[j]) {
                                topo_depth[j] = topo_depth[pred_id] + 1;
                                changed = true;
                            }
                        }
                    }
                }
            }
            int max_depth = *std::max_element(topo_depth.begin(), topo_depth.end());
            printf("Topological depth computed (max depth: %d)\n", max_depth);
        }

        // --- Phase 2: Populate PO guard (priority-based dominance relation) ---
        // Job i dominates j if: priority[i] < priority[j] AND r_max[i] <= r_min[j]
        // AND Pred[i] ⊆ Pred[j]. Then j cannot be dispatched before i.
        v_PO.assign(N * W, 0ULL);
        {
            int po_edges = 0;
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    if (i == j) continue;
                    if (v_priority[i] >= v_priority[j]) continue;
                    if (v_r_max[i] > v_r_min[j]) continue;
                    // Check Pred[i] ⊆ Pred[j]
                    bool pred_subset = true;
                    for (int w = 0; w < W; w++) {
                        if (v_Pred[i * W + w] & ~v_Pred[j * W + w]) {
                            pred_subset = false; break;
                        }
                    }
                    if (!pred_subset) continue;
                    v_PO[j * W + (i / 64)] |= (1ULL << (i % 64));
                    po_edges++;
                }
            }
            // Transitive closure of PO
            for (int iter = 0; iter < N; iter++) {
                bool changed = false;
                for (int j = 0; j < N; j++) {
                    for (int pw = 0; pw < W; pw++) {
                        uint64_t pobits = v_PO[j * W + pw];
                        while (pobits) {
                            int bit = __builtin_ctzll(pobits);
                            pobits &= pobits - 1;
                            int dom = pw * 64 + bit;
                            for (int w = 0; w < W; w++) {
                                uint64_t old = v_PO[j * W + w];
                                v_PO[j * W + w] |= v_PO[dom * W + w];
                                if (v_PO[j * W + w] != old) changed = true;
                            }
                        }
                    }
                }
                if (!changed) break;
            }
            // Count total PO bits after transitive closure
            int po_total = 0;
            for (int j = 0; j < N; j++) {
                for (int w = 0; w < W; w++) {
                    po_total += __builtin_popcountll(v_PO[j * W + w]);
                }
            }
            printf("PO guard: %d direct dominance edges, %d total after transitive closure\n",
                   po_edges, po_total);
        }

        // --- Phase 1 (cont): Candidates sorted by (topo_depth, priority) ---
        v_candidates.resize(N);
        for (int i = 0; i < N; i++) v_candidates[i] = i;
        std::sort(v_candidates.begin(), v_candidates.end(),
            [&](int a, int b) {
                if (topo_depth[a] != topo_depth[b]) return topo_depth[a] < topo_depth[b];
                return v_priority[a] < v_priority[b];
            });
        printf("Candidates sorted by (depth, priority)\n");

        // --- Phase 5: Compute minimum path delay per job ---
        // min_path_delay[j] = min time from t=0 to earliest start of j (BCET + min suspension)
        v_min_path_delay.assign(N, 0);
        {
            // Process in topological order (low depth first = sorted candidates)
            for (int j = 0; j < N; j++) v_min_path_delay[j] = v_r_min[j];
            bool changed = true;
            while (changed) {
                changed = false;
                for (int j = 0; j < N; j++) {
                    for (int w = 0; w < W; w++) {
                        uint64_t pbits = v_Pred[j * W + w];
                        while (pbits) {
                            int bit = __builtin_ctzll(pbits);
                            pbits &= pbits - 1;
                            int pred_id = w * 64 + bit;
                            int32_t via_pred = v_min_path_delay[pred_id] + v_C_min[pred_id]
                                             + v_sus_min[j * N + pred_id];
                            if (via_pred > v_min_path_delay[j]) {
                                v_min_path_delay[j] = via_pred;
                                changed = true;
                            }
                        }
                    }
                }
            }
            int32_t max_delay = *std::max_element(v_min_path_delay.begin(), v_min_path_delay.end());
            printf("Min path delay computed (max: %d)\n", max_delay);
        }

        // Phase 5 (cont): Remove statically infeasible jobs from candidate list
        // A job is infeasible if its earliest possible start exceeds the latest start
        // that would meet its deadline: min_path_delay[j] > deadline[j] - C_min[j]
        {
            int before = (int)v_candidates.size();
            v_candidates.erase(
                std::remove_if(v_candidates.begin(), v_candidates.end(),
                    [&](int j) {
                        return v_min_path_delay[j] > v_deadline[j] - v_C_min[j];
                    }),
                v_candidates.end());
            int pruned = before - (int)v_candidates.size();
            if (pruned > 0)
                printf("Phase 5: pruned %d statically infeasible candidates (%d -> %d)\n",
                       pruned, before, (int)v_candidates.size());
        }

        if (load_states_file) {
            // --- Load states from binary file ---
            // Defer actual loading until after layout is created (below).
            // For now, just set NUM_STATES = 0 as placeholder.
            NUM_STATES = 0;  // will be set after reading header
        } else {
            // --- Single initial state (empty: nothing dispatched) ---
            NUM_STATES = 1;
        }

        printf("Setup complete: N=%d, M=%d, W=%d\n\n", N, M, W);

    } else if (pos_argc == 1) {
        // =================================================================
        // Hardcoded 5-job test (original behavior)
        // =================================================================
        printf("Setting up 5-job SAG expansion test...\n\n");

        N = 5;
        W = bitset_words(N);  // ceil(5/64) = 1
        printf("Job count N=%d, bitset words W=%d\n", N, W);

        // Job attributes
        v_r_min    = {0, 0, 0, 0, 0};
        v_r_max    = {0, 0, 0, 0, 0};
        v_C_min    = {2, 1, 3, 2, 1};
        v_C_max    = {4, 3, 5, 2, 3};
        v_deadline = {20, 20, 25, 30, 30};
        v_priority = {1, 2, 3, 4, 5};
        v_task_id  = {0, 0, 0, 0, 0};

        // Topology: Pred, Succ, TC, PO
        v_Pred.assign(N * W, 0ULL);
        v_Pred[0 * W] = 0x00;
        v_Pred[1 * W] = 0x00;
        v_Pred[2 * W] = 0x01;
        v_Pred[3 * W] = 0x02;
        v_Pred[4 * W] = 0x04;

        v_Succ.assign(N * W, 0ULL);
        v_Succ[0 * W] = 0x04;
        v_Succ[1 * W] = 0x08;
        v_Succ[2 * W] = 0x10;
        v_Succ[3 * W] = 0x00;
        v_Succ[4 * W] = 0x00;

        v_TC.assign(N * W, 0ULL);
        v_TC[0 * W] = 0x00;
        v_TC[1 * W] = 0x00;
        v_TC[2 * W] = 0x01;
        v_TC[3 * W] = 0x02;
        v_TC[4 * W] = 0x05;

        // Phase 1: Topological depth for hardcoded test
        topo_depth.assign(N, 0);
        {
            bool changed = true;
            while (changed) {
                changed = false;
                for (int j = 0; j < N; j++) {
                    for (int w = 0; w < W; w++) {
                        uint64_t pbits = v_Pred[j * W + w];
                        while (pbits) {
                            int bit = __builtin_ctzll(pbits);
                            pbits &= pbits - 1;
                            int pred_id = w * 64 + bit;
                            if (topo_depth[pred_id] + 1 > topo_depth[j]) {
                                topo_depth[j] = topo_depth[pred_id] + 1;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: PO guard for hardcoded test
        v_PO.assign(N * W, 0ULL);
        {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    if (i == j) continue;
                    if (v_priority[i] >= v_priority[j]) continue;
                    if (v_r_max[i] > v_r_min[j]) continue;
                    bool pred_subset = true;
                    for (int w = 0; w < W; w++) {
                        if (v_Pred[i * W + w] & ~v_Pred[j * W + w]) {
                            pred_subset = false; break;
                        }
                    }
                    if (!pred_subset) continue;
                    v_PO[j * W + (i / 64)] |= (1ULL << (i % 64));
                }
            }
            // Transitive closure of PO
            for (int iter = 0; iter < N; iter++) {
                bool changed = false;
                for (int j = 0; j < N; j++) {
                    for (int pw = 0; pw < W; pw++) {
                        uint64_t pobits = v_PO[j * W + pw];
                        while (pobits) {
                            int bit = __builtin_ctzll(pobits);
                            pobits &= pobits - 1;
                            int dom = pw * 64 + bit;
                            for (int w = 0; w < W; w++) {
                                uint64_t old = v_PO[j * W + w];
                                v_PO[j * W + w] |= v_PO[dom * W + w];
                                if (v_PO[j * W + w] != old) changed = true;
                            }
                        }
                    }
                }
                if (!changed) break;
            }
        }

        // Suspension delays
        v_sus_min.assign(N * N, 0);
        v_sus_max.assign(N * N, 0);
        v_sus_min[2 * N + 0] = 1;   v_sus_max[2 * N + 0] = 2;   // 0 -> 2
        v_sus_min[3 * N + 1] = 0;   v_sus_max[3 * N + 1] = 1;   // 1 -> 3
        v_sus_min[4 * N + 2] = 0;   v_sus_max[4 * N + 2] = 0;   // 2 -> 4

        // Phase 1 (cont): Candidates sorted by (depth, priority)
        v_candidates = {0, 1, 2, 3, 4};
        std::sort(v_candidates.begin(), v_candidates.end(),
            [&](int a, int b) {
                if (topo_depth[a] != topo_depth[b]) return topo_depth[a] < topo_depth[b];
                return v_priority[a] < v_priority[b];
            });

        // Phase 5: min path delay for hardcoded test
        v_min_path_delay.assign(N, 0);
        {
            for (int j = 0; j < N; j++) v_min_path_delay[j] = v_r_min[j];
            bool changed = true;
            while (changed) {
                changed = false;
                for (int j = 0; j < N; j++) {
                    for (int w = 0; w < W; w++) {
                        uint64_t pbits = v_Pred[j * W + w];
                        while (pbits) {
                            int bit = __builtin_ctzll(pbits);
                            pbits &= pbits - 1;
                            int pred_id = w * 64 + bit;
                            int32_t via_pred = v_min_path_delay[pred_id] + v_C_min[pred_id]
                                             + v_sus_min[j * N + pred_id];
                            if (via_pred > v_min_path_delay[j]) {
                                v_min_path_delay[j] = via_pred;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Two parent states for hardcoded test
        NUM_STATES = 2;

    } else {
        fprintf(stderr, "Usage: %s [-m cores] [jobs.csv prec.csv [states.bin]]\n", pos_args[0]);
        RUN_FAIL("invalid argument count to run_one_taskset");
    }

    // =====================================================================
    // Common code: setup layout, allocate GPU, launch kernels, print results
    // =====================================================================

    // Create layout descriptor
    SAGStateLayout layout;
    layout.W = W;
    layout.n = N;
    layout.m = M;
    int bps = layout.bytes_per_state();
    printf("Bytes per state: %d\n\n", bps);

    // -----------------------------------------------------------------------
    // Build parent states in flat byte buffer (pinned for async transfers)
    // -----------------------------------------------------------------------
    char* h_state_buf = nullptr;  // pinned host memory (cudaMallocHost)

    if (load_states_file) {
        // --- Load states from binary file ---
        FILE* fs = fopen(pos_args[3], "rb");
        if (!fs) {
            fprintf(stderr, "Error: cannot open states file '%s'\n", pos_args[3]);
            RUN_FAIL(std::string("cannot open states file '") + pos_args[3] + "'");
        }

        uint32_t hdr[4];
        if (fread(hdr, sizeof(uint32_t), 4, fs) != 4) {
            fprintf(stderr, "Error: failed to read states.bin header\n");
            fclose(fs);
            RUN_FAIL("failed to read states.bin header");
        }

        uint32_t file_num_states = hdr[0];
        uint32_t file_n          = hdr[1];
        uint32_t file_W          = hdr[2];
        uint32_t file_m          = hdr[3];

        // Verify consistency with CSV data
        if ((int)file_n != N || (int)file_W != W || (int)file_m != M) {
            fprintf(stderr, "Error: states.bin header mismatch! "
                    "file(n=%u,W=%u,m=%u) vs CSV(n=%d,W=%d,m=%d)\n",
                    file_n, file_W, file_m, N, W, M);
            fclose(fs);
            RUN_FAIL("states.bin header mismatch");
        }

        NUM_STATES = (int)file_num_states;

        printf("Loading %d states from %s (%.1f MB)...\n",
               NUM_STATES, pos_args[3],
               (double)NUM_STATES * bps / (1024.0 * 1024.0));

        cudaCheckError(cudaMallocHost(&h_state_buf, (long long)NUM_STATES * bps));
        size_t bytes_read = fread(h_state_buf, 1,
                                  (long long)NUM_STATES * bps, fs);
        fclose(fs);

        if ((long long)bytes_read != (long long)NUM_STATES * bps) {
            fprintf(stderr, "Error: expected %lld bytes, read %zu\n",
                    (long long)NUM_STATES * bps, bytes_read);
            RUN_FAIL("short read on states.bin");
        }
        printf("States loaded. NUM_STATES=%d\n\n", NUM_STATES);

    } else if (pos_argc == 3) {
        // CSV mode: single initial state (empty -- nothing dispatched)
        cudaCheckError(cudaMallocHost(&h_state_buf, NUM_STATES * bps));
        memset(h_state_buf, 0, NUM_STATES * bps);
        char* h_base = h_state_buf;
        {
            uint64_t* D      = layout.D(h_base, 0);
            uint64_t* X      = layout.X(h_base, 0);
            uint64_t* F_mask = layout.F_mask(h_base, 0);
            int32_t*  A_min  = layout.A_min(h_base, 0);
            int32_t*  A_max  = layout.A_max(h_base, 0);
            int32_t*  F_min  = layout.F_min(h_base, 0);
            int32_t*  F_max  = layout.F_max(h_base, 0);
            int32_t*  ovf    = layout.ovf(h_base, 0);

            for (int w = 0; w < W; w++) { D[w] = 0; X[w] = 0; F_mask[w] = 0; }
            for (int i = 0; i < M; i++) { A_min[i] = 0; A_max[i] = 0; }
            for (int k = 0; k < N; k++) { F_min[k] = 0; F_max[k] = 0; }
            *ovf = 0;
        }
    } else {
        // Hardcoded mode: State 0 (initial) and State 1 (jobs 0,1 dispatched)
        cudaCheckError(cudaMallocHost(&h_state_buf, NUM_STATES * bps));
        memset(h_state_buf, 0, NUM_STATES * bps);
        char* h_base = h_state_buf;

        // State 0: initial (nothing dispatched)
        {
            uint64_t* D      = layout.D(h_base, 0);
            uint64_t* X      = layout.X(h_base, 0);
            uint64_t* F_mask = layout.F_mask(h_base, 0);
            int32_t*  A_min  = layout.A_min(h_base, 0);
            int32_t*  A_max  = layout.A_max(h_base, 0);
            int32_t*  F_min  = layout.F_min(h_base, 0);
            int32_t*  F_max  = layout.F_max(h_base, 0);
            int32_t*  ovf    = layout.ovf(h_base, 0);

            for (int w = 0; w < W; w++) { D[w] = 0; X[w] = 0; F_mask[w] = 0; }
            for (int i = 0; i < M; i++) { A_min[i] = 0; A_max[i] = 0; }
            for (int k = 0; k < N; k++) { F_min[k] = 0; F_max[k] = 0; }
            *ovf = 0;
        }

        // State 1: jobs 0 and 1 dispatched
        {
            uint64_t* D      = layout.D(h_base, 1);
            uint64_t* X      = layout.X(h_base, 1);
            uint64_t* F_mask = layout.F_mask(h_base, 1);
            int32_t*  A_min  = layout.A_min(h_base, 1);
            int32_t*  A_max  = layout.A_max(h_base, 1);
            int32_t*  F_min  = layout.F_min(h_base, 1);
            int32_t*  F_max  = layout.F_max(h_base, 1);
            int32_t*  ovf    = layout.ovf(h_base, 1);

            for (int w = 0; w < W; w++) { D[w] = 0; X[w] = 0; F_mask[w] = 0; }
            D[0]      = 0x03;   // jobs 0 and 1 dispatched
            X[0]      = 0x03;
            F_mask[0] = 0x03;

            A_min[0] = 0;  A_min[1] = 0;  A_min[2] = 1;  A_min[3] = 2;
            A_max[0] = 0;  A_max[1] = 0;  A_max[2] = 3;  A_max[3] = 4;

            for (int k = 0; k < N; k++) { F_min[k] = 0; F_max[k] = 0; }
            F_min[0] = 2;  F_max[0] = 4;
            F_min[1] = 1;  F_max[1] = 3;

            *ovf = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Query GPU memory and compute wave sizing
    // -----------------------------------------------------------------------
    int num_candidates = (int)v_candidates.size();

    size_t gpu_free = 0, gpu_total = 0;
    cudaCheckError(cudaMemGetInfo(&gpu_free, &gpu_total));
    printf("GPU memory: total=%.1f MB, free=%.1f MB\n",
           gpu_total / (1024.0 * 1024.0), gpu_free / (1024.0 * 1024.0));

    // Fixed allocations (topology + job attributes + counters + candidates)
    long long fixed_bytes = 0;
    fixed_bytes += 4LL * N * W * sizeof(uint64_t);        // d_TC, d_PO, d_Pred, d_Succ
    fixed_bytes += 6LL * N * sizeof(int32_t);              // r_min, r_max, C_min, C_max, deadline, priority
    fixed_bytes += 2LL * N * N * sizeof(int32_t);          // sus_min, sus_max
    fixed_bytes += (long long)num_candidates * sizeof(int); // d_candidates
    fixed_bytes += 6LL * sizeof(int);                      // d_valid_count, d_output_count, d_unschedulable_flag (×2 for double buffering)

    // Remaining GPU memory for wave-variable buffers (input + output + pairs).
    // OPT: in parallel-batch mode, divide by parallel width so N concurrent
    // workers don't each claim the entire free pool. Cap the divisor so
    // single-taskset mode and the par=1 batch path still see the full pool
    // (they rely on one very large wave).
    int par_divisor = g_batch_par_n > 0 ? g_batch_par_n : 1;
    if (par_divisor > 4) par_divisor = 4;
    long long usable_gpu = (long long)((gpu_free * 0.85) / par_divisor) - fixed_bytes;
    if (usable_gpu <= 0) {
        fprintf(stderr, "Error: not enough GPU memory. Free=%.1f MB, fixed=%.1f MB\n",
                gpu_free / (1024.0 * 1024.0), fixed_bytes / (1024.0 * 1024.0));
        RUN_FAIL("not enough GPU memory");
    }

    // Cost per input state in a wave:
    //   input: bps bytes
    //   pairs: up to N valid pairs, each sizeof(ValidPair) = 16 bytes
    //   output: up to N successor states, each bps bytes
    // Double buffering: need 2 sets of wave buffers, so divide usable GPU by 2
    // Cost includes: input + output + merge_buf + pairs + D-keys + sort indices + group arrays
    long long cost_per_state = (long long)bps * (1 + N + N)    // input + output + merge_buf
        + (long long)N * sizeof(ValidPair)                     // valid pairs
        + (long long)N * W * sizeof(uint64_t) * 2              // D-keys + D-keys sorted
        + (long long)N * sizeof(int) * 4;                      // sort indices×2 + group starts + group sizes
    int wave_states = (int)(usable_gpu / (2 * cost_per_state));
    if (wave_states <= 0) {
        fprintf(stderr, "Error: cost per state (%lld bytes) exceeds available GPU memory (%lld bytes)\n",
                cost_per_state, usable_gpu);
        RUN_FAIL("cost per state exceeds available GPU memory");
    }
    // Cap wave_states to a reasonable max (not NUM_STATES, since layers grow during iteration)
    // For iteration mode starting from 1 state, layers can grow to thousands+ states
    if (wave_states > 1000000)
        wave_states = 1000000;  // 1M states/wave is plenty

    // Step-3 batch-mode fix: cap total wave allocation to avoid claiming
    // tens of GB per run on the A100. wave_max_output = wave_states * N
    // scales with N^2 for the output buffer, which was producing 70+ GB
    // allocations. Bound the output buffer to ~4 GB per side; this leaves
    // plenty for the n=75 tasksets (observed max layer width ~1M states)
    // while drastically cutting cudaMalloc/Free churn between batch calls.
    int wave_max_pairs  = wave_states * N;
    int wave_max_output = wave_max_pairs;
    {
        // OPT: raised from 2 GiB → 8 GiB. Single-taskset tests show returns
        // flatten above 8 GiB because kernel compute (not launch count) becomes
        // the bottleneck. In parallel-batch mode this cap is further divided
        // by par_divisor.
        long long OUTPUT_CAP_BYTES = 8LL * 1024 * 1024 * 1024;  // 8 GiB
        OUTPUT_CAP_BYTES /= par_divisor;
        long long per_state_bytes = (long long)N * bps;
        if (per_state_bytes > 0) {
            long long cap_states = OUTPUT_CAP_BYTES / per_state_bytes;
            if (cap_states > 0 && cap_states < wave_states) {
                wave_states = (int)cap_states;
                wave_max_pairs = wave_states * N;
                wave_max_output = wave_max_pairs;
            }
        }
    }
    int num_waves = (NUM_STATES + wave_states - 1) / wave_states;  // initial; recomputed per layer

    long long wave_input_bytes  = (long long)wave_states * bps;
    long long wave_output_bytes = (long long)wave_max_output * bps;
    long long wave_pairs_bytes  = (long long)wave_max_pairs * sizeof(ValidPair);

    printf("Fixed GPU alloc:   %.1f MB\n", fixed_bytes / (1024.0 * 1024.0));
    printf("Usable for waves:  %.1f MB\n", usable_gpu / (1024.0 * 1024.0));
    printf("Wave sizing:       %d states/wave, %d max_pairs, %d max_output\n",
           wave_states, wave_max_pairs, wave_max_output);
    printf("Wave buffers:      input=%.1f MB, output=%.1f MB, pairs=%.1f MB\n",
           wave_input_bytes / (1024.0 * 1024.0),
           wave_output_bytes / (1024.0 * 1024.0),
           wave_pairs_bytes / (1024.0 * 1024.0));
    printf("Streaming:         %d wave(s) for %d total states\n\n",
           num_waves, NUM_STATES);

    // -----------------------------------------------------------------------
    // Allocate GPU memory
    // -----------------------------------------------------------------------

    // Device pointers -- fixed (persist across waves)
    int*       d_candidates;
    uint64_t*  d_TC;
    uint64_t*  d_PO;
    uint64_t*  d_Pred;
    uint64_t*  d_Succ;
    int32_t*   d_r_min;
    int32_t*   d_r_max;
    int32_t*   d_C_min;
    int32_t*   d_C_max;
    int32_t*   d_deadline;
    int32_t*   d_priority;
    int32_t*   d_sus_min;
    int32_t*   d_sus_max;
    // Device pointers -- wave-variable, DOUBLE-BUFFERED [2] for ping-pong streaming
    char*      d_input[2];
    ValidPair* d_valid_pairs[2];
    char*      d_output[2];
    int*       d_valid_count[2];
    int*       d_output_count[2];
    int*       d_unschedulable_flag[2];
    int*       d_trunc_flag[2];  // L1 truncation safety: set by K1/K2 on overflow

    // Pinned host counters for async readback (2 per counter for double buffering)
    int* h_valid_count  = nullptr;
    int* h_output_count = nullptr;
    int* h_unsched_flag = nullptr;
    int* h_trunc_flag   = nullptr;

    // Allocate fixed buffers
    cudaCheckError(cudaMalloc(&d_candidates,  num_candidates * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_TC,          N * W * sizeof(uint64_t)));
    cudaCheckError(cudaMalloc(&d_PO,          N * W * sizeof(uint64_t)));
    cudaCheckError(cudaMalloc(&d_Pred,        N * W * sizeof(uint64_t)));
    cudaCheckError(cudaMalloc(&d_Succ,        N * W * sizeof(uint64_t)));
    cudaCheckError(cudaMalloc(&d_r_min,       N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_r_max,       N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_C_min,       N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_C_max,       N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_deadline,    N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_priority,    N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_sus_min,     N * N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_sus_max,     N * N * sizeof(int32_t)));
    // Allocate double-buffered wave-variable buffers (2 sets for ping-pong)
    for (int b = 0; b < 2; b++) {
        cudaCheckError(cudaMalloc(&d_input[b],              wave_input_bytes));
        cudaCheckError(cudaMalloc(&d_valid_pairs[b],        wave_pairs_bytes));
        cudaCheckError(cudaMalloc(&d_output[b],             wave_output_bytes));
        cudaCheckError(cudaMalloc(&d_valid_count[b],        sizeof(int)));
        cudaCheckError(cudaMalloc(&d_output_count[b],       sizeof(int)));
        cudaCheckError(cudaMalloc(&d_unschedulable_flag[b], sizeof(int)));
        cudaCheckError(cudaMalloc(&d_trunc_flag[b], sizeof(int)));
    }

    // Phase B: Sorted ValidPairs buffers + CUB temp storage (double-buffered)
    ValidPair* d_sorted_pairs[2];
    void*      d_sort_temp[2] = {nullptr, nullptr};
    size_t     sort_temp_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        nullptr, sort_temp_bytes,
        (ValidPair*)nullptr, wave_max_pairs,
        [] __device__ (const ValidPair& a, const ValidPair& b) {
            return a.state_idx < b.state_idx;
        });
    for (int b = 0; b < 2; b++) {
        cudaCheckError(cudaMalloc(&d_sorted_pairs[b], wave_pairs_bytes));
        cudaCheckError(cudaMalloc(&d_sort_temp[b],    sort_temp_bytes));
    }

    // Merge pipeline GPU buffers (double-buffered)
    uint64_t* d_D_keys[2];
    uint64_t* d_D_keys_sorted[2];   // sorted output (for RadixSort) or gather output
    int*      d_D_sort_indices[2];   // iota input for sort
    int*      d_merge_sorted_indices[2];  // sort output: sorted index mapping
    int*      d_gpu_group_starts[2];
    int*      d_gpu_group_sizes[2];
    char*     d_merge_buf[2];
    int*      d_merge_count[2];
    void*     d_sort_temp_merge[2] = {nullptr, nullptr};
    size_t    sort_temp_merge_bytes = 0;

    // Query CUB sort temp storage size for merge D-key sort
    if (W == 1) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_merge_bytes,
            (uint64_t*)nullptr, (uint64_t*)nullptr,
            (int*)nullptr, (int*)nullptr,
            wave_max_output, 0, 64);
    } else {
        // For W>1, sort indices by multi-word D-key using MergeSort
        cub::DeviceMergeSort::SortKeys(
            nullptr, sort_temp_merge_bytes,
            (int*)nullptr, wave_max_output,
            [] __device__ (const int& a, const int& b) { return a < b; });
    }

    // GPU-only group detection buffers (double-buffered)
    int*      d_is_start[2];
    int*      d_group_id[2];   // exclusive prefix sum of d_is_start
    int*      d_num_groups_gpu[2];
    void*     d_prefix_sum_temp[2] = {nullptr, nullptr};
    size_t    prefix_sum_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, prefix_sum_temp_bytes,
        (int*)nullptr, (int*)nullptr, wave_max_output);

    for (int b = 0; b < 2; b++) {
        cudaCheckError(cudaMalloc(&d_D_keys[b],                (long long)wave_max_output * W * sizeof(uint64_t)));
        cudaCheckError(cudaMalloc(&d_D_keys_sorted[b],         (long long)wave_max_output * W * sizeof(uint64_t)));
        cudaCheckError(cudaMalloc(&d_D_sort_indices[b],        (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_merge_sorted_indices[b],  (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_gpu_group_starts[b],      (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_gpu_group_sizes[b],       (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_merge_buf[b],             (long long)wave_max_output * bps));
        cudaCheckError(cudaMalloc(&d_merge_count[b],           sizeof(int)));
        cudaCheckError(cudaMalloc(&d_sort_temp_merge[b],       sort_temp_merge_bytes));
        cudaCheckError(cudaMalloc(&d_is_start[b],              (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_group_id[b],              (long long)wave_max_output * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_num_groups_gpu[b],        sizeof(int)));
        cudaCheckError(cudaMalloc(&d_prefix_sum_temp[b],       prefix_sum_temp_bytes));
    }

    // Pinned host counter for synchronous merge-count readback
    int*      h_merge_count    = nullptr;
    cudaCheckError(cudaMallocHost(&h_merge_count,           2 * sizeof(int)));

    // Allocate pinned host counters for async readback
    cudaCheckError(cudaMallocHost(&h_valid_count,  2 * sizeof(int)));
    cudaCheckError(cudaMallocHost(&h_output_count, 2 * sizeof(int)));
    cudaCheckError(cudaMallocHost(&h_unsched_flag, 2 * sizeof(int)));
    cudaCheckError(cudaMallocHost(&h_trunc_flag,   2 * sizeof(int)));

    // Upload fixed data (topology + job attributes) -- done once
    cudaCheckError(cudaMemcpy(d_candidates, v_candidates.data(),    num_candidates * sizeof(int),  cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_TC,         v_TC.data(),            N * W * sizeof(uint64_t),      cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_PO,         v_PO.data(),            N * W * sizeof(uint64_t),      cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_Pred,       v_Pred.data(),          N * W * sizeof(uint64_t),      cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_Succ,       v_Succ.data(),          N * W * sizeof(uint64_t),      cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_r_min,      v_r_min.data(),         N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_r_max,      v_r_max.data(),         N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C_min,      v_C_min.data(),         N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C_max,      v_C_max.data(),         N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_deadline,   v_deadline.data(),      N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_priority,   v_priority.data(),      N * sizeof(int32_t),           cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_sus_min,    v_sus_min.data(),       N * N * sizeof(int32_t),       cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_sus_max,    v_sus_max.data(),       N * N * sizeof(int32_t),       cudaMemcpyHostToDevice));
    // Dynamic shared memory for eligibility kernel
    size_t shared_bytes = STATES_PER_BLOCK * N * (sizeof(int32_t) + sizeof(bool));

    // -----------------------------------------------------------------------
    // BCRT/WCRT device arrays (persistent across layers)
    // -----------------------------------------------------------------------
    int32_t* d_BCRT = nullptr;
    int32_t* d_WCRT = nullptr;
    cudaCheckError(cudaMalloc(&d_BCRT, N * sizeof(int32_t)));
    cudaCheckError(cudaMalloc(&d_WCRT, N * sizeof(int32_t)));
    // Initialize BCRT to INT_MAX (large), WCRT to 0
    // Host-side copies for CPU fast-path (synced to GPU at end)
    std::vector<int32_t> host_BCRT(N, INT32_MAX);
    std::vector<int32_t> host_WCRT(N, 0);
    {
        cudaCheckError(cudaMemcpy(d_BCRT, host_BCRT.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_WCRT, host_WCRT.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // Phase 5: GPU-persistent layer buffers (ping-pong on GPU)
    // -----------------------------------------------------------------------
    // Keep layer data on GPU to avoid H2D/D2H per layer.
    // Two GPU buffers for ping-pong. CPU fast-path uses D2H/H2D only for small layers.
    long long max_layer_states = (long long)wave_max_output * 4;  // generous bound
    long long max_layer_bytes = max_layer_states * bps;
    if (max_layer_bytes < (long long)NUM_STATES * bps)
        max_layer_bytes = (long long)NUM_STATES * bps;
    // Cap GPU layer buffers: use at most 35% of free GPU per buffer (2 buffers).
    // Scale down by parallel batch width.
    {
        size_t gfree2 = 0, gtot2 = 0;
        cudaCheckError(cudaMemGetInfo(&gfree2, &gtot2));
        long long max_per_buf = (long long)((gfree2 * 0.35) / par_divisor);
        if (max_layer_bytes > max_per_buf) {
            max_layer_bytes = max_per_buf;
            max_layer_states = max_layer_bytes / bps;
        }
    }
    // Step-3 batch-mode fix: hard-cap each layer buffer to 4 GiB. The
    // unbounded 4 * wave_max_output heuristic was requesting 16 GiB per
    // buffer for n=75, even when actual layer widths are ~1M states
    // (~1 GiB). Two 4-GiB buffers give 4.5M-state headroom, still
    // plenty. Big wins: cudaMalloc/Free between batch tasksets is now
    // a few ms instead of seconds.
    {
        // OPT: 8 GiB layer buffer cap to match wave_max_output cap.
        long long LAYER_BUF_CAP = 8LL * 1024 * 1024 * 1024;  // 8 GiB
        // L1 test hook: allow forcing a small layer buffer via env var
        if (const char* env_cap = getenv("SAG_LAYER_BUF_CAP_MB")) {
            long long env_mb = atoll(env_cap);
            if (env_mb > 0) LAYER_BUF_CAP = env_mb * 1024 * 1024;
        }
        LAYER_BUF_CAP /= par_divisor;
        if (max_layer_bytes > LAYER_BUF_CAP) {
            max_layer_bytes = LAYER_BUF_CAP;
            max_layer_states = max_layer_bytes / bps;
        }
    }
    // Ensure at least enough for initial states
    if (max_layer_bytes < (long long)NUM_STATES * bps) {
        max_layer_bytes = (long long)NUM_STATES * bps;
        max_layer_states = NUM_STATES;
    }

    char* d_layer_buf[2] = {nullptr, nullptr};
    cudaCheckError(cudaMalloc(&d_layer_buf[0], max_layer_bytes));
    cudaCheckError(cudaMalloc(&d_layer_buf[1], max_layer_bytes));
    printf("Phase 5: GPU layer buffers: 2 x %.1f MB = %.1f MB (max %lld states)\n",
           max_layer_bytes / (1024.0 * 1024.0),
           2.0 * max_layer_bytes / (1024.0 * 1024.0),
           max_layer_states);

    // Upload initial states to d_layer_buf[0]
    cudaCheckError(cudaMemcpy(d_layer_buf[0], h_state_buf,
                              (long long)NUM_STATES * bps, cudaMemcpyHostToDevice));
    int d_layer_cur = 0;  // index into d_layer_buf[] for current layer input

    // GPU-side max popcount scratch — no longer needed (A1 optimization:
    // in BFS-style layer processing, popcount(D) = layer for all states,
    // so wave_num_cands is computed directly from the layer index).
    // int* d_max_popcount = nullptr;
    // cudaCheckError(cudaMalloc(&d_max_popcount, sizeof(int)));

    // Phase 5 stats
    long long phase5_d2h_avoided_bytes = 0;
    long long phase5_d2d_bytes = 0;
    long long phase5_total_gpu_alloc = 2 * max_layer_bytes;

    // CPU fast-path only fires for layers with < CPU_LAYER_THRESHOLD states.
    // Output can grow by at most N successors per input state (pre-merge).
    long long host_layer_bytes = (long long)CPU_LAYER_THRESHOLD * (long long)(N + 1) * bps;
    if (host_layer_bytes < 4LL  * 1024 * 1024) host_layer_bytes = 4LL  * 1024 * 1024;
    if (host_layer_bytes > 256LL * 1024 * 1024) host_layer_bytes = 256LL * 1024 * 1024;
    long long host_max_layer_states = host_layer_bytes / bps;

    char* h_layer_buf_A = nullptr;
    char* h_layer_buf_B = nullptr;
    cudaCheckError(cudaMallocHost(&h_layer_buf_A, host_layer_bytes));
    cudaCheckError(cudaMallocHost(&h_layer_buf_B, host_layer_bytes));

    // -----------------------------------------------------------------------
    // L2 DRAM Spill buffers for layer overflow
    // -----------------------------------------------------------------------
    // Two pinned host buffers ping-pong: current layer's overflow output goes
    // to h_spill_out (grows as needed), and the next layer reads from
    // h_spill_in. They swap at layer boundaries. Size in states, not bytes.
    char* h_spill_out = nullptr;            // current layer spill output
    long long h_spill_out_cap = 0;          // capacity (states)
    long long h_spill_out_count = 0;        // states spilled this layer
    char* h_spill_in = nullptr;             // previous layer spill, consumed by current
    long long h_spill_in_cap = 0;
    long long h_spill_in_count = 0;
    long long grand_total_spilled = 0;      // diagnostic: total spilled states

    auto ensure_spill_out_cap = [&](long long need_states) {
        if (need_states <= h_spill_out_cap) return;
        long long new_cap = h_spill_out_cap > 0 ? h_spill_out_cap : 1024;
        while (new_cap < need_states) new_cap *= 2;
        char* new_buf = nullptr;
        cudaCheckError(cudaMallocHost(&new_buf, new_cap * bps));
        if (h_spill_out && h_spill_out_count > 0) {
            memcpy(new_buf, h_spill_out, h_spill_out_count * bps);
        }
        if (h_spill_out) cudaCheckError(cudaFreeHost(h_spill_out));
        h_spill_out = new_buf;
        h_spill_out_cap = new_cap;
    };

    // Sort-then-stream global-merge staging buffers, all in pinned DRAM.
    //
    // Replaces the earlier chunked-accumulator + foldin cascade that could
    // truncate once both tiers filled. The new design:
    //   h_gm_stage_in  — all states of the layer gathered from GPU + DRAM spill
    //   h_gm_stage_out — merged output (concatenation of merged D-groups)
    //   h_gm_batch_pack — CPU scatter/pack scratch for one GPU-sized batch
    //
    // Sort by D puts same-D states contiguous, so we stream batches that end
    // at D-group boundaries. No group is ever split across batches, so no
    // cross-batch accumulation is needed and the merge kernel's in-GPU output
    // is final for those groups. Capacity scales with DRAM, not VRAM.
    char*     h_gm_stage_in = nullptr;    long long h_gm_stage_in_cap = 0;
    char*     h_gm_stage_out = nullptr;   long long h_gm_stage_out_cap = 0;
    char*     h_gm_batch_pack = nullptr;  long long h_gm_batch_pack_cap = 0;
    auto ensure_pinned_cap = [&](char*& buf, long long& cap, long long need_states) {
        if (need_states <= cap) return;
        long long new_cap = cap > 0 ? cap : 1024;
        while (new_cap < need_states) new_cap *= 2;
        char* new_buf = nullptr;
        cudaCheckError(cudaMallocHost(&new_buf, new_cap * bps));
        if (buf) cudaCheckError(cudaFreeHost(buf));
        buf = new_buf;
        cap = new_cap;
    };

    // -----------------------------------------------------------------------
    // Create CUDA streams and events for timing
    // -----------------------------------------------------------------------
    cudaStream_t streams[2];
    cudaCheckError(cudaStreamCreate(&streams[0]));
    cudaCheckError(cudaStreamCreate(&streams[1]));

    cudaEvent_t ev_k1_start, ev_k1_stop, ev_k2_start, ev_k2_stop;
    cudaEvent_t ev_merge_start, ev_merge_stop;
    cudaCheckError(cudaEventCreate(&ev_k1_start));
    cudaCheckError(cudaEventCreate(&ev_k1_stop));
    cudaCheckError(cudaEventCreate(&ev_k2_start));
    cudaCheckError(cudaEventCreate(&ev_k2_stop));
    cudaCheckError(cudaEventCreate(&ev_merge_start));
    cudaCheckError(cudaEventCreate(&ev_merge_stop));

    // -----------------------------------------------------------------------
    // Phase 3: Precompute depth-to-candidate-count mapping
    // -----------------------------------------------------------------------
    int max_topo_depth = topo_depth.empty() ? 0 : *std::max_element(topo_depth.begin(), topo_depth.end());
    std::vector<int> depth_candidate_count(max_topo_depth + 2, 0);
    for (int i = 0; i < num_candidates; i++) {
        int d = topo_depth[v_candidates[i]];
        if (d <= max_topo_depth)
            depth_candidate_count[d]++;
    }
    for (int d = 1; d <= max_topo_depth + 1; d++) {
        depth_candidate_count[d] += depth_candidate_count[d - 1];
    }

    // =======================================================================
    // OUTER ITERATION LOOP (Algorithm 1 from RTSS 2024)
    // =======================================================================
    // Phase 5: Layer data stays on GPU in d_layer_buf[d_layer_cur].
    // Each iteration: expand all states -> merge -> d_layer_buf[1-d_layer_cur].
    // CPU fast-path: D2H/H2D for layers < CPU_LAYER_THRESHOLD.

    auto wall_start = std::chrono::high_resolution_clock::now();

    int layer = 0;
    int layer_state_count = NUM_STATES;
    bool schedulable = true;
    bool analysis_failed = false;  // set on buffer overflow — distinct from UNSCHED verdict

    // Global accumulators
    long long grand_total_expanded = 0;
    long long grand_total_merged   = 0;
    long long grand_total_valid    = 0;
    int       max_layer_width      = 0;
    long long grand_total_nodes    = 0;  // total SAG nodes = sum of layer_state_count
    float     grand_total_time_k1  = 0.0f;
    float     grand_total_time_k2  = 0.0f;
    float     grand_total_time_merge = 0.0f;
    int       cpu_layers_count     = 0;
    int       gpu_layers_count     = 0;

    while (layer_state_count > 0) {
        printf("\n=== LAYER %d: %d input states ===\n", layer, layer_state_count);
        grand_total_nodes += layer_state_count;
        if (layer_state_count > max_layer_width)
            max_layer_width = layer_state_count;

        // Recompute wave sizing for this layer
        int cur_num_waves = (layer_state_count + wave_states - 1) / wave_states;

        // Per-layer accumulators
        long long layer_expanded = 0;
        long long layer_valid    = 0;
        int       layer_unsched  = 0;
        int       truncation_flag = 0;  // L1: any buffer overflow this layer
        float     layer_time_k1  = 0.0f;
        float     layer_time_k2  = 0.0f;
        float     layer_time_merge = 0.0f;

        // Phase 5: output goes to d_layer_buf[1 - d_layer_cur] on GPU
        int d_layer_out = 1 - d_layer_cur;
        int layer_output_offset = 0;  // offset in states into output layer buffer
        int wave_num_cands = num_candidates;  // A1: declared here (before goto) to avoid jump-past-init
        long long gpu_count_for_waves = 0;    // L2: declared here for same reason

        // --- CPU FAST-PATH for small layers ---
        if (layer_state_count < CPU_LAYER_THRESHOLD) {
            // D2H: download current layer from GPU for CPU processing
            long long cpu_layer_bytes = (long long)layer_state_count * bps;
            cudaCheckError(cudaMemcpy(h_layer_buf_A, d_layer_buf[d_layer_cur],
                                      cpu_layer_bytes, cudaMemcpyDeviceToHost));

            long long cpu_expanded = 0, cpu_valid = 0;
            int cpu_unsched = 0, cpu_truncated = 0;
            int merged = cpu_expand_and_merge_layer(
                h_layer_buf_A, layer_state_count,
                h_layer_buf_B, (int)host_max_layer_states,
                layout,
                v_candidates, num_candidates,
                v_TC, v_PO, v_Pred, v_Succ,
                v_r_min, v_r_max, v_C_min, v_C_max,
                v_deadline, v_priority, v_sus_min, v_sus_max,
                topo_depth,
                N, M, W, bps,
                host_BCRT, host_WCRT,
                cpu_expanded, cpu_valid, cpu_unsched, cpu_truncated);
            layer_unsched |= cpu_unsched;
            truncation_flag |= cpu_truncated;  // L1: CPU fast-path overflow

            layer_output_offset = merged;
            layer_expanded = cpu_expanded;
            layer_valid = cpu_valid;

            // H2D: upload merged output back to GPU layer buffer
            if (merged > 0) {
                cudaCheckError(cudaMemcpy(d_layer_buf[d_layer_out], h_layer_buf_B,
                                          (long long)merged * bps, cudaMemcpyHostToDevice));
            }

            printf("  CPU fast-path: %d states -> %lld pairs -> %lld expanded -> %d merged\n",
                   layer_state_count, cpu_valid, cpu_expanded, merged);

            cpu_layers_count++;
            goto layer_done;
        }

        // A1 optimization: compute wave_num_cands once per layer.
        // In BFS-style processing, all states in layer L have popcount(D) = L,
        // so max_popcount = layer and depth_limit = layer + 1.
        // This eliminates the MaxPopcountKernel launch + D2H sync per wave.
        wave_num_cands = num_candidates;
        if (!topo_depth.empty()) {
            int depth_limit = layer + 1;
            if (depth_limit <= max_topo_depth) {
                wave_num_cands = depth_candidate_count[depth_limit];
            }
            if (wave_num_cands < 1) wave_num_cands = 1;
        }

        // Phase 5 + L2: Pre-copy wave 0 to d_input[0], handling spill_in
        // L2: gpu_count = GPU-resident states; remainder is in h_spill_in
        gpu_count_for_waves = layer_state_count - h_spill_in_count;
        if (gpu_count_for_waves < 0) gpu_count_for_waves = 0;
        {
            int wave0_count = (wave_states < layer_state_count) ? wave_states : layer_state_count;
            cudaCheckError(cudaMemsetAsync(d_valid_count[0], 0, sizeof(int), streams[0]));
            cudaCheckError(cudaMemsetAsync(d_output_count[0], 0, sizeof(int), streams[0]));
            cudaCheckError(cudaMemsetAsync(d_unschedulable_flag[0], 0, sizeof(int), streams[0]));
            cudaCheckError(cudaMemsetAsync(d_trunc_flag[0], 0, sizeof(int), streams[0]));
            // L2: split copy between GPU and host spill
            long long end = (long long)wave0_count;
            long long gpu_end = (end <= gpu_count_for_waves) ? end : gpu_count_for_waves;
            long long gpu_bytes = gpu_end * bps;
            long long host_bytes = (end - gpu_end) * bps;
            if (gpu_bytes > 0) {
                cudaCheckError(cudaMemcpyAsync(d_input[0], d_layer_buf[d_layer_cur],
                                               gpu_bytes, cudaMemcpyDeviceToDevice, streams[0]));
                phase5_d2d_bytes += gpu_bytes;
            }
            if (host_bytes > 0) {
                cudaCheckError(cudaMemcpyAsync(d_input[0] + gpu_bytes, h_spill_in,
                                               host_bytes, cudaMemcpyHostToDevice, streams[0]));
            }
            phase5_d2h_avoided_bytes += (long long)wave0_count * bps;
        }

        for (int wave = 0; wave < cur_num_waves; wave++) {
            int buf = wave % 2;
            int wave_start = wave * wave_states;
            int wave_count = wave_states;
            if (wave_count > layer_state_count - wave_start)
                wave_count = layer_state_count - wave_start;

            // Wait for THIS wave's D2D copy to complete
            cudaCheckError(cudaStreamSynchronize(streams[buf]));

            // Pre-launch NEXT wave's D2D copy on the OTHER stream (Phase 5: D2D instead of H2D)
            if (wave + 1 < cur_num_waves) {
                int nb = 1 - buf;
                int next_start = (wave + 1) * wave_states;
                int next_count = wave_states;
                if (next_count > layer_state_count - next_start)
                    next_count = layer_state_count - next_start;
                long long next_bytes = (long long)next_count * bps;

                cudaCheckError(cudaMemsetAsync(d_valid_count[nb], 0, sizeof(int), streams[nb]));
                cudaCheckError(cudaMemsetAsync(d_output_count[nb], 0, sizeof(int), streams[nb]));
                cudaCheckError(cudaMemsetAsync(d_unschedulable_flag[nb], 0, sizeof(int), streams[nb]));
                cudaCheckError(cudaMemsetAsync(d_trunc_flag[nb], 0, sizeof(int), streams[nb]));
                // L2: split between GPU and host spill
                {
                    long long ns = next_start;
                    long long ne = ns + next_count;
                    long long gpu_e = (ne <= gpu_count_for_waves) ? ne : gpu_count_for_waves;
                    long long gpu_b = (ns <= gpu_count_for_waves) ? ns : gpu_count_for_waves;
                    long long g_bytes = (gpu_e - gpu_b) * bps;
                    long long h_begin = (ns > gpu_count_for_waves) ? (ns - gpu_count_for_waves) : 0;
                    long long h_end = (ne > gpu_count_for_waves) ? (ne - gpu_count_for_waves) : 0;
                    long long h_bytes = (h_end - h_begin) * bps;
                    if (g_bytes > 0) {
                        cudaCheckError(cudaMemcpyAsync(d_input[nb],
                            d_layer_buf[d_layer_cur] + gpu_b * bps,
                            g_bytes, cudaMemcpyDeviceToDevice, streams[nb]));
                        phase5_d2d_bytes += g_bytes;
                    }
                    if (h_bytes > 0) {
                        cudaCheckError(cudaMemcpyAsync(d_input[nb] + g_bytes,
                            h_spill_in + h_begin * bps,
                            h_bytes, cudaMemcpyHostToDevice, streams[nb]));
                    }
                }
                phase5_d2h_avoided_bytes += next_bytes;
            }

            // A1 optimization: wave_num_cands is now computed once per layer
            // (before the wave loop) using the known layer index, eliminating
            // the MaxPopcountKernel launch + D2H sync that was here.

            // --- K1: Fused eligibility ---
            dim3 grid1((wave_count + STATES_PER_BLOCK - 1) / STATES_PER_BLOCK);
            dim3 block1(WARP_SIZE, STATES_PER_BLOCK);

            cudaCheckError(cudaEventRecord(ev_k1_start, streams[buf]));
            FusedEligibilityKernel<<<grid1, block1, shared_bytes, streams[buf]>>>(
                d_input[buf], layout, d_candidates, wave_count, wave_num_cands,
                d_TC, d_PO, d_Pred,
                d_r_min, d_r_max, d_priority,
                d_sus_min, d_sus_max, N, W,
                d_valid_pairs[buf], d_valid_count[buf], d_unschedulable_flag[buf],
                wave_max_pairs, d_trunc_flag[buf]);
            cudaCheckError(cudaEventRecord(ev_k1_stop, streams[buf]));

            // Phase 1a: NO sync after K1, NO ValidPair sort
            // K2 reads d_valid_count from device pointer (same stream guarantees ordering)

            // --- K2: Successor creation (over-allocated grid) ---
            // Phase 1a: Over-allocate grid — use wave_count * candidates as tighter bound.
            // Max gridDim.x on CC 8.0 is 2^31-1 so no clamping is needed.
            int k2_upper = wave_count * wave_num_cands;  // tighter than wave_max_pairs
            if (k2_upper > wave_max_pairs) k2_upper = wave_max_pairs;
            long long k2_blocks_needed = (long long)(k2_upper + K2_WARPS_PER_BLOCK - 1)
                                         / K2_WARPS_PER_BLOCK;
            if (k2_blocks_needed < 1) k2_blocks_needed = 1;
            int grid2_x = (int)k2_blocks_needed;
            dim3 grid2(grid2_x);
            dim3 block2(K2_BLOCK_SIZE);
            size_t k2_smem_per_warp = 3 * W * sizeof(uint64_t) + 2 * M * sizeof(int32_t);
            size_t k2_shared_bytes = K2_WARPS_PER_BLOCK * k2_smem_per_warp;

            cudaCheckError(cudaEventRecord(ev_k2_start, streams[buf]));
            CreateSuccessorsKernel<<<grid2, block2, k2_shared_bytes, streams[buf]>>>(
                d_input[buf], layout, d_valid_pairs[buf],  // Phase 1a: direct, no sort
                d_valid_count[buf],
                d_Pred, d_Succ,
                d_C_min, d_C_max, d_deadline,
                N, M, W,
                d_output[buf], d_output_count[buf], d_unschedulable_flag[buf],
                wave_max_output,
                d_BCRT, d_WCRT, d_r_min, d_trunc_flag[buf]);
            cudaCheckError(cudaEventRecord(ev_k2_stop, streams[buf]));

            // Phase 1a: Combined readback after K2 (valid_count + output_count + unsched)
            cudaCheckError(cudaMemcpyAsync(&h_valid_count[buf], d_valid_count[buf],
                sizeof(int), cudaMemcpyDeviceToHost, streams[buf]));
            cudaCheckError(cudaMemcpyAsync(&h_output_count[buf], d_output_count[buf],
                sizeof(int), cudaMemcpyDeviceToHost, streams[buf]));
            cudaCheckError(cudaMemcpyAsync(&h_unsched_flag[buf], d_unschedulable_flag[buf],
                sizeof(int), cudaMemcpyDeviceToHost, streams[buf]));
            cudaCheckError(cudaMemcpyAsync(&h_trunc_flag[buf], d_trunc_flag[buf],
                sizeof(int), cudaMemcpyDeviceToHost, streams[buf]));
            cudaCheckError(cudaStreamSynchronize(streams[buf]));

            float ms_k1;
            cudaCheckError(cudaEventElapsedTime(&ms_k1, ev_k1_start, ev_k1_stop));
            float ms_k2;
            cudaCheckError(cudaEventElapsedTime(&ms_k2, ev_k2_start, ev_k2_stop));

            int actual_valid = h_valid_count[buf];
            if (actual_valid > wave_max_pairs) actual_valid = wave_max_pairs;

            int clamped_output = h_output_count[buf];
            if (clamped_output > wave_max_output) clamped_output = wave_max_output;

            // L1: aggregate per-wave truncation flag into per-layer flag
            truncation_flag |= h_trunc_flag[buf];

            // --- MERGE PHASE ---
            float ms_merge = 0.0f;
            int wave_merged = clamped_output;

            if (clamped_output > 1) {
                int merge_grid = (clamped_output + MERGE_BLOCK_SIZE - 1) / MERGE_BLOCK_SIZE;

                // Phase 4: Fused ExtractDKeys + Iota
                ExtractDKeysAndIotaKernel<<<merge_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                    d_output[buf], layout, clamped_output, W, d_D_keys[buf], d_D_sort_indices[buf]);

                if (W == 1) {
                    size_t temp_bytes = sort_temp_merge_bytes;
                    cub::DeviceRadixSort::SortPairs(
                        d_sort_temp_merge[buf], temp_bytes,
                        d_D_keys[buf], d_D_keys_sorted[buf],
                        d_D_sort_indices[buf], d_merge_sorted_indices[buf],
                        clamped_output, 0, 64, streams[buf]);
                } else {
                    cudaCheckError(cudaMemcpyAsync(d_merge_sorted_indices[buf], d_D_sort_indices[buf],
                        clamped_output * sizeof(int), cudaMemcpyDeviceToDevice, streams[buf]));
                    const uint64_t* dk = d_D_keys[buf];
                    int local_W = W;
                    size_t temp_bytes = sort_temp_merge_bytes;
                    cub::DeviceMergeSort::SortKeys(
                        d_sort_temp_merge[buf], temp_bytes,
                        d_merge_sorted_indices[buf], clamped_output,
                        [dk, local_W] __device__ (const int& a, const int& b) {
                            for (int w = 0; w < local_W; w++) {
                                if (dk[a * local_W + w] < dk[b * local_W + w]) return true;
                                if (dk[a * local_W + w] > dk[b * local_W + w]) return false;
                            }
                            return false;
                        }, streams[buf]);
                }

                GatherDKeysKernel<<<merge_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                    d_D_keys[buf], d_merge_sorted_indices[buf],
                    clamped_output, W, d_D_keys_sorted[buf]);

                // GPU-only group detection: detect boundaries → prefix sum → compact starts → sizes
                cudaCheckError(cudaEventRecord(ev_merge_start, streams[buf]));

                DetectBoundariesKernel<<<merge_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                    d_D_keys_sorted[buf], clamped_output, W, d_is_start[buf]);

                {
                    size_t ps_bytes = prefix_sum_temp_bytes;
                    cub::DeviceScan::ExclusiveSum(
                        d_prefix_sum_temp[buf], ps_bytes,
                        d_is_start[buf], d_group_id[buf],
                        clamped_output, streams[buf]);
                }

                CompactGroupStartsKernel<<<merge_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                    d_is_start[buf], d_group_id[buf], clamped_output,
                    d_gpu_group_starts[buf], d_num_groups_gpu[buf]);

                // Phase 1b: NO sync for num_groups — merge kernels read from device pointer
                // Phase 2: Use warp-cooperative merge kernel
                cudaCheckError(cudaMemsetAsync(d_merge_count[buf], 0, sizeof(int), streams[buf]));

                // Phase 2: Warp-cooperative merge (reads num_groups from device pointer)
                {
                    int warp_merge_grid = (clamped_output + MERGE_WARPS_PER_BLOCK - 1) / MERGE_WARPS_PER_BLOCK;
                    int warp_merge_block = MERGE_WARPS_PER_BLOCK * WARP_SIZE;
                    size_t merge_smem = MERGE_WARPS_PER_BLOCK * MAX_SLOTS_PER_GROUP * sizeof(int);

                    FusedRangeMergeKernelWarp<<<warp_merge_grid, warp_merge_block, merge_smem, streams[buf]>>>(
                        d_output[buf], d_merge_sorted_indices[buf],
                        d_gpu_group_starts[buf], d_num_groups_gpu[buf],
                        clamped_output, layout, N, M, W,
                        d_merge_buf[buf], d_merge_count[buf]);
                }

                // D2H: merge count (4 bytes) - async + sync (replaces sync cudaMemcpy which double-syncs)
                cudaCheckError(cudaMemcpyAsync(&h_merge_count[buf], d_merge_count[buf],
                    sizeof(int), cudaMemcpyDeviceToHost, streams[buf]));
                cudaCheckError(cudaStreamSynchronize(streams[buf]));
                int gpu_merged = h_merge_count[buf];

                // --- Iterative re-merge passes ---
                // OPT: re-merge passes disabled. Profiling on the big tasksets showed
                // pass-1 almost never reduces the state count (e.g. 504->504, 1e5->1e5),
                // because the warp-cooperative merge kernel is already idempotent within
                // a D-group: each state is linearly compared against ALL existing slots,
                // so a second pass has no opportunity to find new merges. Removing them
                // saves ~40% of the merge-pipeline work (K3 + sort + D2D).
                {
                    int prev_merged = gpu_merged;
                    for (int pass = 1; pass < 1 && gpu_merged > 1; pass++) {
                        // Copy merge result to d_output[buf] for re-processing
                        cudaCheckError(cudaMemcpyAsync(d_output[buf], d_merge_buf[buf],
                            (long long)gpu_merged * bps, cudaMemcpyDeviceToDevice, streams[buf]));

                        int re_count = gpu_merged;
                        int re_grid = (re_count + MERGE_BLOCK_SIZE - 1) / MERGE_BLOCK_SIZE;

                        // Phase 4: Fused ExtractDKeys + Iota
                        ExtractDKeysAndIotaKernel<<<re_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                            d_output[buf], layout, re_count, W, d_D_keys[buf], d_D_sort_indices[buf]);

                        if (W == 1) {
                            size_t temp_bytes = sort_temp_merge_bytes;
                            cub::DeviceRadixSort::SortPairs(
                                d_sort_temp_merge[buf], temp_bytes,
                                d_D_keys[buf], d_D_keys_sorted[buf],
                                d_D_sort_indices[buf], d_merge_sorted_indices[buf],
                                re_count, 0, 64, streams[buf]);
                        } else {
                            cudaCheckError(cudaMemcpyAsync(d_merge_sorted_indices[buf], d_D_sort_indices[buf],
                                re_count * sizeof(int), cudaMemcpyDeviceToDevice, streams[buf]));
                            const uint64_t* dk = d_D_keys[buf];
                            int local_W = W;
                            size_t temp_bytes = sort_temp_merge_bytes;
                            cub::DeviceMergeSort::SortKeys(
                                d_sort_temp_merge[buf], temp_bytes,
                                d_merge_sorted_indices[buf], re_count,
                                [dk, local_W] __device__ (const int& a, const int& b) {
                                    for (int w = 0; w < local_W; w++) {
                                        if (dk[a * local_W + w] < dk[b * local_W + w]) return true;
                                        if (dk[a * local_W + w] > dk[b * local_W + w]) return false;
                                    }
                                    return false;
                                }, streams[buf]);
                        }

                        GatherDKeysKernel<<<re_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                            d_D_keys[buf], d_merge_sorted_indices[buf], re_count, W, d_D_keys_sorted[buf]);

                        DetectBoundariesKernel<<<re_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                            d_D_keys_sorted[buf], re_count, W, d_is_start[buf]);

                        {
                            size_t ps_bytes = prefix_sum_temp_bytes;
                            cub::DeviceScan::ExclusiveSum(
                                d_prefix_sum_temp[buf], ps_bytes,
                                d_is_start[buf], d_group_id[buf],
                                re_count, streams[buf]);
                        }

                        CompactGroupStartsKernel<<<re_grid, MERGE_BLOCK_SIZE, 0, streams[buf]>>>(
                            d_is_start[buf], d_group_id[buf], re_count,
                            d_gpu_group_starts[buf], d_num_groups_gpu[buf]);

                        // Phase 1b + Phase 2: Warp-cooperative re-merge, no num_groups sync
                        cudaCheckError(cudaMemsetAsync(d_merge_count[buf], 0, sizeof(int), streams[buf]));

                        {
                            int warp_merge_grid = (re_count + MERGE_WARPS_PER_BLOCK - 1) / MERGE_WARPS_PER_BLOCK;
                                    int warp_merge_block = MERGE_WARPS_PER_BLOCK * WARP_SIZE;
                            size_t merge_smem = MERGE_WARPS_PER_BLOCK * MAX_SLOTS_PER_GROUP * sizeof(int);

                            FusedRangeMergeKernelWarp<<<warp_merge_grid, warp_merge_block, merge_smem, streams[buf]>>>(
                                d_output[buf], d_merge_sorted_indices[buf],
                                d_gpu_group_starts[buf], d_num_groups_gpu[buf],
                                re_count, layout, N, M, W,
                                d_merge_buf[buf], d_merge_count[buf]);
                        }

                        cudaCheckError(cudaStreamSynchronize(streams[buf]));
                        cudaCheckError(cudaMemcpy(&h_merge_count[buf], d_merge_count[buf],
                            sizeof(int), cudaMemcpyDeviceToHost));
                        gpu_merged = h_merge_count[buf];

                        printf("    Re-merge pass %d: %d -> %d states\n", pass, prev_merged, gpu_merged);
                        if (gpu_merged >= prev_merged) break;  // no improvement
                        prev_merged = gpu_merged;
                    }
                }

                cudaCheckError(cudaEventRecord(ev_merge_stop, streams[buf]));
                cudaCheckError(cudaEventSynchronize(ev_merge_stop));
                cudaCheckError(cudaEventElapsedTime(&ms_merge, ev_merge_start, ev_merge_stop));

                wave_merged = gpu_merged;

                // OPT: UpdateBCRTWCRTFromMergedKernel removed.
                // BCRT/WCRT are already maintained inside CreateSuccessorsKernel (K2) via
                // atomicMin/atomicMax on the freshly dispatched job's (f_min, f_max) at
                // successor creation time. Merging only combines states with identical D
                // via min(F_min)/max(F_max), so any merged-state bounds are already
                // captured by the per-pair K2 updates. The post-merge kernel was redundant
                // and consumed ~28% of GPU kernel time on large tasksets.

                // L2: split copy between GPU layer buffer and DRAM spill
                // Per-wave merge result copies run on streams[buf] (same as merge kernel)
                // so they're sequential with the merge. Using async avoids CPU block.
                if (gpu_merged > 0) {
                    long long fit_gpu = (long long)max_layer_states - layer_output_offset;
                    if (fit_gpu < 0) fit_gpu = 0;
                    int to_gpu = (gpu_merged <= fit_gpu) ? gpu_merged : (int)fit_gpu;
                    int to_host = gpu_merged - to_gpu;
                    if (to_gpu > 0) {
                        cudaCheckError(cudaMemcpyAsync(
                            d_layer_buf[d_layer_out] + (long long)layer_output_offset * bps,
                            d_merge_buf[buf],
                            (long long)to_gpu * bps,
                            cudaMemcpyDeviceToDevice, streams[buf]));
                        phase5_d2d_bytes += (long long)to_gpu * bps;
                        phase5_d2h_avoided_bytes += (long long)to_gpu * bps;
                        layer_output_offset += to_gpu;
                    }
                    if (to_host > 0) {
                        // L2 DRAM spill: states that don't fit in VRAM go here
                        ensure_spill_out_cap(h_spill_out_count + to_host);
                        cudaCheckError(cudaMemcpyAsync(
                            h_spill_out + h_spill_out_count * bps,
                            d_merge_buf[buf] + (long long)to_gpu * bps,
                            (long long)to_host * bps,
                            cudaMemcpyDeviceToHost, streams[buf]));
                        h_spill_out_count += to_host;
                        grand_total_spilled += to_host;
                    }
                }
            } else if (clamped_output == 1) {
                if (layer_output_offset + 1 <= max_layer_states) {
                    cudaCheckError(cudaMemcpyAsync(
                        d_layer_buf[d_layer_out] + (long long)layer_output_offset * bps,
                        d_output[buf], bps, cudaMemcpyDeviceToDevice, streams[buf]));
                    phase5_d2d_bytes += bps;
                    phase5_d2h_avoided_bytes += bps;
                    layer_output_offset += 1;
                } else {
                    ensure_spill_out_cap(h_spill_out_count + 1);
                    cudaCheckError(cudaMemcpyAsync(
                        h_spill_out + h_spill_out_count * bps,
                        d_output[buf], bps, cudaMemcpyDeviceToHost, streams[buf]));
                    h_spill_out_count += 1;
                    grand_total_spilled += 1;
                }
            }

            layer_expanded += clamped_output;
            layer_valid    += actual_valid;
            layer_unsched  |= h_unsched_flag[buf];
            layer_time_k1  += ms_k1;
            layer_time_k2  += ms_k2;
            layer_time_merge += ms_merge;

            if (cur_num_waves <= 10 || wave % (cur_num_waves / 10 + 1) == 0 || wave == cur_num_waves - 1) {
                printf("  Wave %d/%d: %d states -> %d pairs -> %d expanded -> %d merged "
                       "(K1: %.2f, K2: %.2f, M: %.2f ms)\n",
                       wave + 1, cur_num_waves, wave_count, actual_valid, clamped_output, wave_merged,
                       ms_k1, ms_k2, ms_merge);
            }
        } // end wave loop

        // Phase 5 + L2: Global merge across waves, including spilled states
        //   GPU portion:  d_layer_buf[d_layer_out][0 .. layer_output_offset)
        //   DRAM portion: h_spill_out[0 .. h_spill_out_count)
        //   Single-pass merge if (layer_output_offset + h_spill_out_count) <= wave_max_output
        //   Otherwise: chunked merge against an accumulator (TODO L2 chunked merge)
        if ((cur_num_waves > 1 || h_spill_out_count > 0) && (layer_output_offset + h_spill_out_count) > 0) {
            cudaCheckError(cudaStreamSynchronize(streams[0]));
            cudaCheckError(cudaStreamSynchronize(streams[1]));
            long long total_pre_merge = (long long)layer_output_offset + h_spill_out_count;
            printf("  Global merge across %d waves: %lld states (gpu=%d, spill=%lld)",
                   cur_num_waves, total_pre_merge, layer_output_offset, h_spill_out_count);

            // L2: Merge pipeline wrapped as a lambda. Input: d_output[0][0..input_count),
            // Output: d_merge_buf[0][0..return_value). Runs full merge pipeline.
            auto run_merge_pipeline = [&](int input_count) -> int {
                int mg = (input_count + MERGE_BLOCK_SIZE - 1) / MERGE_BLOCK_SIZE;
                ExtractDKeysAndIotaKernel<<<mg, MERGE_BLOCK_SIZE>>>(
                    d_output[0], layout, input_count, W, d_D_keys[0], d_D_sort_indices[0]);
                if (W == 1) {
                    size_t tb = sort_temp_merge_bytes;
                    cub::DeviceRadixSort::SortPairs(
                        d_sort_temp_merge[0], tb,
                        d_D_keys[0], d_D_keys_sorted[0],
                        d_D_sort_indices[0], d_merge_sorted_indices[0],
                        input_count, 0, 64);
                } else {
                    cudaCheckError(cudaMemcpy(d_merge_sorted_indices[0], d_D_sort_indices[0],
                        input_count * sizeof(int), cudaMemcpyDeviceToDevice));
                    const uint64_t* dk = d_D_keys[0];
                    int lw = W;
                    size_t tb = sort_temp_merge_bytes;
                    cub::DeviceMergeSort::SortKeys(
                        d_sort_temp_merge[0], tb,
                        d_merge_sorted_indices[0], input_count,
                        [dk, lw] __device__ (const int& a, const int& b) {
                            for (int w = 0; w < lw; w++) {
                                if (dk[a * lw + w] < dk[b * lw + w]) return true;
                                if (dk[a * lw + w] > dk[b * lw + w]) return false;
                            }
                            return false;
                        });
                }
                GatherDKeysKernel<<<mg, MERGE_BLOCK_SIZE>>>(
                    d_D_keys[0], d_merge_sorted_indices[0], input_count, W, d_D_keys_sorted[0]);
                DetectBoundariesKernel<<<mg, MERGE_BLOCK_SIZE>>>(
                    d_D_keys_sorted[0], input_count, W, d_is_start[0]);
                {
                    size_t ps = prefix_sum_temp_bytes;
                    cub::DeviceScan::ExclusiveSum(
                        d_prefix_sum_temp[0], ps,
                        d_is_start[0], d_group_id[0], input_count);
                }
                CompactGroupStartsKernel<<<mg, MERGE_BLOCK_SIZE>>>(
                    d_is_start[0], d_group_id[0], input_count,
                    d_gpu_group_starts[0], d_num_groups_gpu[0]);
                cudaCheckError(cudaMemsetAsync(d_merge_count[0], 0, sizeof(int), 0));
                int wmg = (input_count + MERGE_WARPS_PER_BLOCK - 1) / MERGE_WARPS_PER_BLOCK;
                int wmb = MERGE_WARPS_PER_BLOCK * WARP_SIZE;
                size_t msm = MERGE_WARPS_PER_BLOCK * MAX_SLOTS_PER_GROUP * sizeof(int);
                FusedRangeMergeKernelWarp<<<wmg, wmb, msm>>>(
                    d_output[0], d_merge_sorted_indices[0],
                    d_gpu_group_starts[0], d_num_groups_gpu[0],
                    input_count, layout, N, M, W,
                    d_merge_buf[0], d_merge_count[0]);
                cudaCheckError(cudaDeviceSynchronize());
                cudaCheckError(cudaMemcpy(&h_merge_count[0], d_merge_count[0],
                    sizeof(int), cudaMemcpyDeviceToHost));
                return h_merge_count[0];
            };

            int gm_merged = 0;
            if (total_pre_merge <= wave_max_output) {
                // Single-pass merge (async on default stream - sequential with merge pipeline)
                int gm_count = (int)total_pre_merge;
                cudaCheckError(cudaMemcpyAsync(d_output[0], d_layer_buf[d_layer_out],
                    (long long)layer_output_offset * bps, cudaMemcpyDeviceToDevice));
                if (h_spill_out_count > 0) {
                    cudaCheckError(cudaMemcpyAsync(
                        d_output[0] + (long long)layer_output_offset * bps,
                        h_spill_out, h_spill_out_count * bps, cudaMemcpyHostToDevice));
                    printf(" [uploaded %lld spilled]", h_spill_out_count);
                }
                phase5_d2d_bytes += (long long)gm_count * bps;
                gm_merged = run_merge_pipeline(gm_count);
            } else {
                // Sort-then-stream global merge. Unbounded in DRAM.
                //
                // Invariants this relies on:
                //   * Two states can only merge if their D bitmasks are identical.
                //   * The merge operations (X := AND, A/F := min/max, F_mask := OR)
                //     are monotone in interval width, so first-fit within a group
                //     is already sound — it just occasionally leaves a pair
                //     unmerged, which is why we only need per-group completeness.
                //
                // Therefore: sort all states by D, batch consecutive complete
                // D-groups up to GPU capacity, merge each batch independently,
                // concatenate results. Never accumulates across batches.
                printf(" [sort-stream:");
                auto t_gm_stage0 = std::chrono::high_resolution_clock::now();

                // 1. Gather GPU portion + DRAM spill into one pinned buffer.
                ensure_pinned_cap(h_gm_stage_in,  h_gm_stage_in_cap,  total_pre_merge);
                ensure_pinned_cap(h_gm_stage_out, h_gm_stage_out_cap, total_pre_merge);
                ensure_pinned_cap(h_gm_batch_pack, h_gm_batch_pack_cap, wave_max_output);
                if (layer_output_offset > 0) {
                    cudaCheckError(cudaMemcpy(h_gm_stage_in,
                        d_layer_buf[d_layer_out],
                        (long long)layer_output_offset * bps,
                        cudaMemcpyDeviceToHost));
                }
                if (h_spill_out_count > 0) {
                    memcpy(h_gm_stage_in + (long long)layer_output_offset * bps,
                           h_spill_out,
                           (long long)h_spill_out_count * bps);
                }

                // 2. Indirect sort of state indices by full D lexicographically.
                //    Parallelized via OpenMP (__gnu_parallel::sort).
                std::vector<int> sorted_idx((size_t)total_pre_merge);
                std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
                const int lW = W;
                char* const stage = h_gm_stage_in;
                SAGStateLayout lay = layout;
                auto t_sort0 = std::chrono::high_resolution_clock::now();
                __gnu_parallel::sort(sorted_idx.begin(), sorted_idx.end(),
                    [stage, lay, lW](int a, int b) {
                        const uint64_t* Da = lay.D(stage, a);
                        const uint64_t* Db = lay.D(stage, b);
                        for (int w = 0; w < lW; w++) {
                            if (Da[w] != Db[w]) return Da[w] < Db[w];
                        }
                        return false;
                    });
                auto t_sort1 = std::chrono::high_resolution_clock::now();

                // 3. Locate D-group boundaries in sorted order.
                std::vector<long long> group_starts;
                group_starts.reserve(1024);
                group_starts.push_back(0);
                for (long long i = 1; i < total_pre_merge; i++) {
                    const uint64_t* Dp = layout.D(h_gm_stage_in, sorted_idx[(size_t)(i - 1)]);
                    const uint64_t* Dc = layout.D(h_gm_stage_in, sorted_idx[(size_t)i]);
                    bool differ = false;
                    for (int w = 0; w < W; w++) {
                        if (Dp[w] != Dc[w]) { differ = true; break; }
                    }
                    if (differ) group_starts.push_back(i);
                }
                group_starts.push_back(total_pre_merge);
                long long ngroups = (long long)group_starts.size() - 1;
                printf(" groups=%lld sort=%.2fms",
                       ngroups,
                       std::chrono::duration<double, std::milli>(t_sort1 - t_sort0).count());

                // 4. Stream batches of complete D-groups through the GPU merge
                //    pipeline. Append merged output to h_gm_stage_out.
                long long out_count = 0;
                long long g = 0;
                int batches_run = 0;
                while (g < ngroups) {
                    long long g_lo = g;
                    long long batch_start_pos = group_starts[g_lo];
                    long long batch_states = 0;
                    while (g < ngroups &&
                           (group_starts[g + 1] - batch_start_pos) <= wave_max_output) {
                        batch_states = group_starts[g + 1] - batch_start_pos;
                        g++;
                    }

                    if (batch_states > 0) {
                        // Pack sorted state bytes into pinned scratch, then one H2D.
                        #pragma omp parallel for schedule(static)
                        for (long long i = 0; i < batch_states; i++) {
                            int src_i = sorted_idx[(size_t)(batch_start_pos + i)];
                            memcpy(h_gm_batch_pack + i * bps,
                                   h_gm_stage_in + (long long)src_i * bps,
                                   bps);
                        }
                        cudaCheckError(cudaMemcpy(d_output[0], h_gm_batch_pack,
                            batch_states * bps, cudaMemcpyHostToDevice));
                        int merged = run_merge_pipeline((int)batch_states);
                        cudaCheckError(cudaMemcpy(h_gm_stage_out + out_count * bps,
                            d_merge_buf[0],
                            (long long)merged * bps,
                            cudaMemcpyDeviceToHost));
                        out_count += merged;
                        batches_run++;
                    } else {
                        // One D-group alone exceeds wave_max_output. Stream it in
                        // sub-batches through the same merge kernel: merging within
                        // each sub-batch is first-fit-safe (same soundness
                        // guarantee as the in-kernel greedy merge within a group),
                        // and unmerged pairs across sub-batches are just left as
                        // separate states — exactly what the kernel would do with
                        // a group too large for its local_slots cache anyway.
                        long long pos = group_starts[g_lo];
                        long long end = group_starts[g_lo + 1];
                        while (pos < end) {
                            long long sub = (end - pos < (long long)wave_max_output)
                                            ? (end - pos) : (long long)wave_max_output;
                            #pragma omp parallel for schedule(static)
                            for (long long i = 0; i < sub; i++) {
                                int src_i = sorted_idx[(size_t)(pos + i)];
                                memcpy(h_gm_batch_pack + i * bps,
                                       h_gm_stage_in + (long long)src_i * bps,
                                       bps);
                            }
                            cudaCheckError(cudaMemcpy(d_output[0], h_gm_batch_pack,
                                sub * bps, cudaMemcpyHostToDevice));
                            int merged = run_merge_pipeline((int)sub);
                            cudaCheckError(cudaMemcpy(h_gm_stage_out + out_count * bps,
                                d_merge_buf[0],
                                (long long)merged * bps,
                                cudaMemcpyDeviceToHost));
                            out_count += merged;
                            batches_run++;
                            pos += sub;
                        }
                        g = g_lo + 1;
                    }
                }

                auto t_gm_stage1 = std::chrono::high_resolution_clock::now();
                printf(" batches=%d merged=%lld total=%.1fms]",
                       batches_run, out_count,
                       std::chrono::duration<double, std::milli>(t_gm_stage1 - t_gm_stage0).count());

                // 5. Dispatch h_gm_stage_out directly to d_layer_buf + h_spill_out.
                //    (The single-pass branch dispatches from d_merge_buf[0]; sort-
                //    stream's output can exceed wave_max_output so we route from
                //    pinned DRAM here and signal the downstream block to no-op.)
                long long to_gpu  = (out_count <= max_layer_states)
                                    ? out_count : max_layer_states;
                long long to_host = out_count - to_gpu;
                if (to_gpu > 0) {
                    cudaCheckError(cudaMemcpy(d_layer_buf[d_layer_out],
                        h_gm_stage_out, to_gpu * bps, cudaMemcpyHostToDevice));
                    phase5_d2d_bytes += to_gpu * bps;
                }
                if (to_host > 0) {
                    ensure_spill_out_cap(to_host);
                    memcpy(h_spill_out,
                           h_gm_stage_out + to_gpu * bps,
                           to_host * bps);
                    grand_total_spilled += to_host;
                }
                layer_output_offset = (int)to_gpu;
                h_spill_out_count   = to_host;
                gm_merged           = -1;  // signal: already dispatched, skip downstream
            }

            // OPT: UpdateBCRTWCRT after global merge removed (redundant — see comment above).

            // Phase 5 + L2: Copy global merge result. GPU portion fits in d_layer_buf,
            // overflow goes to h_spill_out for next layer to consume via H2D waves.
            // Sort-stream path self-dispatches and sets gm_merged = -1; skip here.
            if (gm_merged == -1) {
                // already dispatched by sort-stream branch
            } else if (gm_merged > 0) {
                int gm_to_gpu = (gm_merged <= max_layer_states) ? gm_merged : (int)max_layer_states;
                int gm_to_host = gm_merged - gm_to_gpu;
                // Async on default stream - run_merge_pipeline ends on default stream,
                // so this copy is ordered after merge without explicit sync.
                cudaCheckError(cudaMemcpyAsync(d_layer_buf[d_layer_out], d_merge_buf[0],
                    (long long)gm_to_gpu * bps, cudaMemcpyDeviceToDevice));
                phase5_d2d_bytes += (long long)gm_to_gpu * bps;
                if (gm_to_host > 0) {
                    // L2: re-spill excess after merge (will become h_spill_in for next layer)
                    ensure_spill_out_cap(gm_to_host);
                    cudaCheckError(cudaMemcpyAsync(
                        h_spill_out, d_merge_buf[0] + (long long)gm_to_gpu * bps,
                        (long long)gm_to_host * bps, cudaMemcpyDeviceToHost));
                    h_spill_out_count = gm_to_host;
                    grand_total_spilled += gm_to_host;
                } else {
                    h_spill_out_count = 0;  // merged result fit in GPU
                }
                layer_output_offset = gm_to_gpu;  // GPU portion count
            } else {
                layer_output_offset = 0;
                h_spill_out_count = 0;
            }
            printf(" -> %d gpu + %lld spill after global merge\n", layer_output_offset, h_spill_out_count);

            // --- Iterative re-merge on global merged result (all on GPU) ---
            // OPT: disabled — see per-wave rationale above.
            {
                int gm_prev = layer_output_offset;
                for (int pass = 1; pass < 1 && layer_output_offset > 1; pass++) {
                    int re_count = layer_output_offset;
                    // d_layer_buf[d_layer_out] has the current result; copy to d_output[0]
                    cudaCheckError(cudaMemcpy(d_output[0], d_layer_buf[d_layer_out],
                        (long long)re_count * bps, cudaMemcpyDeviceToDevice));
                    phase5_d2d_bytes += (long long)re_count * bps;

                    int re_grid = (re_count + MERGE_BLOCK_SIZE - 1) / MERGE_BLOCK_SIZE;

                    ExtractDKeysAndIotaKernel<<<re_grid, MERGE_BLOCK_SIZE>>>(
                        d_output[0], layout, re_count, W, d_D_keys[0], d_D_sort_indices[0]);

                    if (W == 1) {
                        size_t temp_bytes = sort_temp_merge_bytes;
                        cub::DeviceRadixSort::SortPairs(
                            d_sort_temp_merge[0], temp_bytes,
                            d_D_keys[0], d_D_keys_sorted[0],
                            d_D_sort_indices[0], d_merge_sorted_indices[0],
                            re_count, 0, 64);
                    } else {
                        cudaCheckError(cudaMemcpy(d_merge_sorted_indices[0], d_D_sort_indices[0],
                            re_count * sizeof(int), cudaMemcpyDeviceToDevice));
                        const uint64_t* dk = d_D_keys[0];
                        int local_W = W;
                        size_t temp_bytes = sort_temp_merge_bytes;
                        cub::DeviceMergeSort::SortKeys(
                            d_sort_temp_merge[0], temp_bytes,
                            d_merge_sorted_indices[0], re_count,
                            [dk, local_W] __device__ (const int& a, const int& b) {
                                for (int w = 0; w < local_W; w++) {
                                    if (dk[a * local_W + w] < dk[b * local_W + w]) return true;
                                    if (dk[a * local_W + w] > dk[b * local_W + w]) return false;
                                }
                                return false;
                            });
                    }

                    GatherDKeysKernel<<<re_grid, MERGE_BLOCK_SIZE>>>(
                        d_D_keys[0], d_merge_sorted_indices[0], re_count, W, d_D_keys_sorted[0]);

                    DetectBoundariesKernel<<<re_grid, MERGE_BLOCK_SIZE>>>(
                        d_D_keys_sorted[0], re_count, W, d_is_start[0]);

                    {
                        size_t ps_bytes = prefix_sum_temp_bytes;
                        cub::DeviceScan::ExclusiveSum(
                            d_prefix_sum_temp[0], ps_bytes,
                            d_is_start[0], d_group_id[0], re_count);
                    }

                    CompactGroupStartsKernel<<<re_grid, MERGE_BLOCK_SIZE>>>(
                        d_is_start[0], d_group_id[0], re_count,
                        d_gpu_group_starts[0], d_num_groups_gpu[0]);

                    cudaCheckError(cudaMemsetAsync(d_merge_count[0], 0, sizeof(int), 0));

                    {
                        int warp_merge_grid = (re_count + MERGE_WARPS_PER_BLOCK - 1) / MERGE_WARPS_PER_BLOCK;
                            int warp_merge_block = MERGE_WARPS_PER_BLOCK * WARP_SIZE;
                        size_t merge_smem = MERGE_WARPS_PER_BLOCK * MAX_SLOTS_PER_GROUP * sizeof(int);

                        FusedRangeMergeKernelWarp<<<warp_merge_grid, warp_merge_block, merge_smem>>>(
                            d_output[0], d_merge_sorted_indices[0],
                            d_gpu_group_starts[0], d_num_groups_gpu[0],
                            re_count, layout, N, M, W,
                            d_merge_buf[0], d_merge_count[0]);
                    }

                    cudaCheckError(cudaStreamSynchronize(streams[0]));
                    cudaCheckError(cudaMemcpy(&h_merge_count[0], d_merge_count[0],
                        sizeof(int), cudaMemcpyDeviceToHost));
                    int re_merged = h_merge_count[0];

                    // OPT: UpdateBCRTWCRT after re-merge pass removed (redundant).

                    // Phase 5: copy re-merged to d_layer_buf[d_layer_out] (D2D)
                    if (re_merged > 0) {
                        cudaCheckError(cudaMemcpy(d_layer_buf[d_layer_out], d_merge_buf[0],
                            (long long)re_merged * bps, cudaMemcpyDeviceToDevice));
                        phase5_d2d_bytes += (long long)re_merged * bps;
                    }
                    layer_output_offset = re_merged;

                    printf("    Global re-merge pass %d: %d -> %d states\n", pass, gm_prev, layer_output_offset);
                    if (layer_output_offset >= gm_prev) break;  // no improvement
                    gm_prev = layer_output_offset;
                }
            }
        }

        gpu_layers_count++;

        layer_truncated:  // L1: jump here when global merge would truncate
        layer_done:  // CPU fast-path jumps here after completing expand+merge

        grand_total_expanded += layer_expanded;
        grand_total_merged   += layer_output_offset;
        grand_total_valid    += layer_valid;
        grand_total_time_k1  += layer_time_k1;
        grand_total_time_k2  += layer_time_k2;
        grand_total_time_merge += layer_time_merge;

        printf("  Layer %d summary: %lld expanded -> %d merged (K1: %.2f, K2: %.2f, M: %.2f ms)\n",
               layer, layer_expanded, layer_output_offset,
               layer_time_k1, layer_time_k2, layer_time_merge);

        // L1: Check truncation BEFORE other termination checks.
        // Truncation = "analysis failed to fit state space". DO NOT conflate with
        // UNSCHED: the correct user-visible outcome is ok=false, error="TRUNCATED".
        // Dropping nodes would violate R4 (every expanded node must be explicitly
        // re-expanded in the next layer). Report failure, let caller decide.
        if (truncation_flag) {
            fprintf(stderr, "\n*** ANALYSIS FAILED: buffer overflow (truncation) at layer %d ***\n", layer);
            fprintf(stderr, "    Possible causes: layer width exceeds max_layer_states=%lld"
                            " or wave_max_output=%d\n",
                    max_layer_states, wave_max_output);
            fprintf(stderr, "    Fix: increase buffer caps, enable DRAM spill (Level 2), or add cascade-merge\n");
            analysis_failed = true;
            break;
        }

        // --- Termination checks ---

        if (layer_unsched) {
            printf("\n*** UNSCHEDULABLE: deadline-miss witness in layer %d ***\n", layer);
            schedulable = false;
            break;
        }

        if (layer_output_offset == 0) {
            printf("\n*** UNSCHEDULABLE: no valid successors at layer %d ***\n", layer);
            schedulable = false;
            break;
        }

        // Phase 5: Check completeness on GPU — download first state's D, check popcount
        // (All states in a BFS layer have the same popcount(D) = layer+1 if starting from empty)
        {
            bool all_complete = false;
            uint64_t h_D_check[MAX_BITSET_WORDS];
            cudaCheckError(cudaMemcpy(h_D_check, d_layer_buf[d_layer_out],
                W * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            int pc = 0;
            for (int w = 0; w < W; w++) pc += __builtin_popcountll(h_D_check[w]);
            if (pc >= N) all_complete = true;
            if (all_complete) {
                printf("\n*** All states fully scheduled at layer %d ***\n", layer);
                break;
            }
        }

        // --- Prepare next layer ---
        // Total input for next layer = GPU residents + DRAM residents
        layer_state_count = layer_output_offset + (int)h_spill_out_count;

        // Phase 5: Swap GPU layer buffer index (no host-side swap needed)
        d_layer_cur = d_layer_out;

        // L2: spill_out (which holds current-layer overflow) becomes next-layer's spill_in.
        // Swap them, and reset out for the new layer.
        {
            char* tmp_buf = h_spill_in;
            long long tmp_cap = h_spill_in_cap;
            h_spill_in = h_spill_out;
            h_spill_in_cap = h_spill_out_cap;
            h_spill_in_count = h_spill_out_count;
            h_spill_out = tmp_buf;
            h_spill_out_cap = tmp_cap;
            h_spill_out_count = 0;
        }

        layer++;
    } // end outer iteration loop

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time_s = std::chrono::duration<double>(wall_end - wall_start).count();

    // -----------------------------------------------------------------------
    // Read back BCRT/WCRT and merge GPU + CPU fast-path results
    // -----------------------------------------------------------------------
    // With --default-stream=per-thread, the synchronous cudaMemcpy below only
    // drains the per-thread default stream, NOT streams[0/1]. Some
    // UpdateBCRTWCRTFromMergedKernel launches inside the wave loop run on
    // streams[buf] and may still be pending. Drain them explicitly.
    cudaCheckError(cudaStreamSynchronize(streams[0]));
    cudaCheckError(cudaStreamSynchronize(streams[1]));
    std::vector<int32_t> h_BCRT(N), h_WCRT(N);
    cudaCheckError(cudaMemcpy(h_BCRT.data(), d_BCRT, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(h_WCRT.data(), d_WCRT, N * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // Merge: CPU fast-path may have updated host_BCRT/host_WCRT
    for (int j = 0; j < N; j++) {
        if (host_BCRT[j] < h_BCRT[j]) h_BCRT[j] = host_BCRT[j];
        if (host_WCRT[j] > h_WCRT[j]) h_WCRT[j] = host_WCRT[j];
    }

    // -----------------------------------------------------------------------
    // Print final results (nptest-compatible format)
    // -----------------------------------------------------------------------
    printf("\n========================================\n");
    printf("   GPU SAG Full Iteration Results\n");
    printf("========================================\n");
    printf("Schedulable:            %s\n",
           analysis_failed ? "TRUNCATED" : (schedulable ? "YES" : "NO"));
    printf("Total layers:           %d\n", layer + 1);
    printf("Total SAG nodes:        %lld\n", grand_total_nodes);
    printf("Total expanded:         %lld\n", grand_total_expanded);
    printf("Total merged:           %lld\n", grand_total_merged);
    printf("Max layer width:        %d\n", max_layer_width);
    printf("Wall time:              %.6f s\n", wall_time_s);
    printf("GPU time (K1+K2+M):     %.4f ms\n",
           grand_total_time_k1 + grand_total_time_k2 + grand_total_time_merge);
    printf("  K1: %.4f ms, K2: %.4f ms, Merge: %.4f ms\n",
           grand_total_time_k1, grand_total_time_k2, grand_total_time_merge);
    printf("Jobs (n): %d, Cores (m): %d, W: %d, bps: %d\n", N, M, W, bps);
    printf("========================================\n");

    // Optimization summary
    float total_gpu_ms = grand_total_time_k1 + grand_total_time_k2 + grand_total_time_merge;
    printf("\n--- Optimization Summary ---\n");
    printf("Phase 1a (K1-K2 sync elim):  ACTIVE\n");
    printf("Phase 1b (merge sync elim):  ACTIVE\n");
    printf("Phase 2  (warp-coop merge):  ACTIVE\n");
    printf("Phase 3  (K2 occupancy=%d):  ACTIVE\n", K2_WARPS_PER_BLOCK);
    printf("Phase 4  (kernel fusion):    ACTIVE\n");
    printf("Phase 5  (GPU persistence):  ACTIVE\n");
    printf("  GPU persistence: %.2f MB D2H avoided, %.2f MB D2D transferred\n",
           phase5_d2h_avoided_bytes / (1024.0 * 1024.0),
           phase5_d2d_bytes / (1024.0 * 1024.0));
    printf("  GPU layer buffers: %.1f MB allocated\n",
           phase5_total_gpu_alloc / (1024.0 * 1024.0));
    printf("Phase 6  (K1 rho_hp opt):    PENDING\n");
    if (total_gpu_ms > 0.0f) {
        printf("Timing breakdown: K1=%.1f%%, K2=%.1f%%, Merge=%.1f%%\n",
               100.0f * grand_total_time_k1 / total_gpu_ms,
               100.0f * grand_total_time_k2 / total_gpu_ms,
               100.0f * grand_total_time_merge / total_gpu_ms);
    }
    if (grand_total_expanded > 0) {
        printf("Overall compression: %lld expanded -> %lld merged (%.2fx)\n",
               grand_total_expanded, grand_total_merged,
               (double)grand_total_expanded / grand_total_merged);
    }
    printf("----------------------------\n");

    // --- Detailed Performance Metrics ---
    printf("\n--- Detailed Performance Metrics ---\n");
    printf("Layers processed:       %d\n", layer + 1);
    printf("  CPU fast-path layers: %d (< %d states)\n", cpu_layers_count, CPU_LAYER_THRESHOLD);
    printf("  GPU-accelerated layers: %d\n", gpu_layers_count);
    printf("States explored:\n");
    printf("  Total input nodes:    %lld\n", grand_total_nodes);
    printf("  Total expanded:       %lld (avg %.1f successors/input)\n",
           grand_total_expanded,
           grand_total_nodes > 0 ? (double)grand_total_expanded / grand_total_nodes : 0.0);
    printf("  Total after merge:    %lld (compression: %.2fx)\n",
           grand_total_merged,
           grand_total_merged > 0 ? (double)grand_total_expanded / grand_total_merged : 0.0);
    printf("  Max layer width:      %d states\n", max_layer_width);
    printf("Kernel timing:\n");
    {
        float detail_total_gpu_ms = grand_total_time_k1 + grand_total_time_k2 + grand_total_time_merge;
        if (detail_total_gpu_ms > 0) {
            printf("  K1 (eligibility):     %.4f ms (%.1f%%)\n", grand_total_time_k1, 100*grand_total_time_k1/detail_total_gpu_ms);
            printf("  K2 (successor):       %.4f ms (%.1f%%)\n", grand_total_time_k2, 100*grand_total_time_k2/detail_total_gpu_ms);
            printf("  Merge:                %.4f ms (%.1f%%)\n", grand_total_time_merge, 100*grand_total_time_merge/detail_total_gpu_ms);
            printf("  Total GPU kernel:     %.4f ms\n", detail_total_gpu_ms);
        }
    }
    printf("Throughput:\n");
    if (wall_time_s > 0) {
        printf("  Input states/sec:     %.2f M/s\n", grand_total_nodes / (wall_time_s * 1e6));
        printf("  Expanded states/sec:  %.2f M/s\n", grand_total_expanded / (wall_time_s * 1e6));
    }
    printf("Hardware:\n");
    {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("  GPU memory used:      %.0f MB / %.0f MB (%.0f%%)\n",
               (total_mem - free_mem) / (1024.0*1024.0), total_mem / (1024.0*1024.0),
               100.0 * (total_mem - free_mem) / total_mem);
    }

    // nptest-compatible one-line output.
    // When analysis truncated the state space we cannot claim a verdict, so
    // we emit -1 in the schedulable column to force the Python harness's
    // parser to fall through to UNKNOWN rather than matching SCHED/UNSCHED.
    const char* input_file = (pos_argc >= 2) ? pos_args[1] : "hardcoded";
    int verdict_code = analysis_failed ? -1 : (schedulable ? 1 : 0);
    printf("\n%s, %d, %d, %lld, %lld, %lld, %d, %.6f, %.1f, 0, 0, %d\n",
           input_file,
           verdict_code,
           N,
           grand_total_nodes,
           grand_total_expanded,
           grand_total_merged,
           max_layer_width,
           wall_time_s,
           (fixed_bytes + 2 * (wave_input_bytes + wave_output_bytes + wave_pairs_bytes)) / (1024.0 * 1024.0),
           M);

    // Print BCRT/WCRT per job
    if (schedulable && !analysis_failed) {
        printf("\n--- BCRT/WCRT per job ---\n");
        for (int j = 0; j < N; j++) {
            if (h_BCRT[j] < INT32_MAX) {
                printf("  Job %d: BCRT=%d, WCRT=%d\n", j, h_BCRT[j], h_WCRT[j]);
            }
        }
    }

    // BCRT/WCRT summary
    {
        int bcrt_count = 0, wcrt_count = 0;
        for (int j = 0; j < N; j++) {
            if (h_BCRT[j] < INT32_MAX) bcrt_count++;
            if (h_WCRT[j] > 0) wcrt_count++;
        }
        printf("\n--- BCRT/WCRT Summary ---\n");
        printf("  Jobs with BCRT data:  %d\n", bcrt_count);
        printf("  Jobs with WCRT data:  %d\n", wcrt_count);
        printf("  Jobs tracked:         %d / %d\n", bcrt_count, N);
    }

    // Write .rta.csv if in CSV mode.
    // Skip writing on truncation: partial per-job BCRT/WCRT would report 0
    // for jobs that were never dispatched, which a downstream harness could
    // misread as "this job has a trivial bound".
    if (pos_argc >= 3 && schedulable && !analysis_failed) {
        // Derive output filename from input
        std::string rta_file = std::string(pos_args[1]);
        auto dot = rta_file.rfind('.');
        if (dot != std::string::npos) rta_file = rta_file.substr(0, dot);
        rta_file += ".rta.csv";

        FILE* frta = fopen(rta_file.c_str(), "w");
        if (frta) {
            fprintf(frta, "Task ID, Job ID, BCCT, WCCT, BCRT, WCRT\n");
            for (int j = 0; j < N; j++) {
                // BCRT = f_min - r_min, WCRT = f_max - r_min
                // BCCT = f_min (completion time), WCCT = f_max
                int bcrt = (h_BCRT[j] < INT32_MAX) ? h_BCRT[j] : 0;
                int wcrt = h_WCRT[j];
                int bcct = bcrt + v_r_min[j];
                int wcct = wcrt + v_r_min[j];
                int task_id = (j < (int)v_task_id.size()) ? v_task_id[j] : 0;
                fprintf(frta, "%d, %d, %d, %d, %d, %d\n",
                        task_id, j, bcct, wcct, bcrt, wcrt);
            }
            fclose(frta);
            printf("Wrote %s\n", rta_file.c_str());
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaCheckError(cudaStreamDestroy(streams[0]));
    cudaCheckError(cudaStreamDestroy(streams[1]));
    cudaCheckError(cudaEventDestroy(ev_k1_start));
    cudaCheckError(cudaEventDestroy(ev_k1_stop));
    cudaCheckError(cudaEventDestroy(ev_k2_start));
    cudaCheckError(cudaEventDestroy(ev_k2_stop));
    cudaCheckError(cudaEventDestroy(ev_merge_start));
    cudaCheckError(cudaEventDestroy(ev_merge_stop));

    for (int b = 0; b < 2; b++) {
        cudaCheckError(cudaFree(d_input[b]));
        cudaCheckError(cudaFree(d_valid_pairs[b]));
        cudaCheckError(cudaFree(d_output[b]));
        cudaCheckError(cudaFree(d_valid_count[b]));
        cudaCheckError(cudaFree(d_output_count[b]));
        cudaCheckError(cudaFree(d_unschedulable_flag[b]));
        cudaCheckError(cudaFree(d_trunc_flag[b]));
        cudaCheckError(cudaFree(d_D_keys[b]));
        cudaCheckError(cudaFree(d_D_keys_sorted[b]));
        cudaCheckError(cudaFree(d_D_sort_indices[b]));
        cudaCheckError(cudaFree(d_merge_sorted_indices[b]));
        cudaCheckError(cudaFree(d_gpu_group_starts[b]));
        cudaCheckError(cudaFree(d_gpu_group_sizes[b]));
        cudaCheckError(cudaFree(d_merge_buf[b]));
        cudaCheckError(cudaFree(d_merge_count[b]));
        cudaCheckError(cudaFree(d_sort_temp_merge[b]));
        // --- Fix: 12 previously-leaked buffers (Step 4 leak patch) ---
        cudaCheckError(cudaFree(d_sorted_pairs[b]));
        cudaCheckError(cudaFree(d_sort_temp[b]));
        cudaCheckError(cudaFree(d_is_start[b]));
        cudaCheckError(cudaFree(d_group_id[b]));
        cudaCheckError(cudaFree(d_num_groups_gpu[b]));
        cudaCheckError(cudaFree(d_prefix_sum_temp[b]));
    }

    cudaCheckError(cudaFree(d_candidates));
    cudaCheckError(cudaFree(d_TC));
    cudaCheckError(cudaFree(d_PO));
    cudaCheckError(cudaFree(d_Pred));
    cudaCheckError(cudaFree(d_Succ));
    cudaCheckError(cudaFree(d_r_min));
    cudaCheckError(cudaFree(d_r_max));
    cudaCheckError(cudaFree(d_C_min));
    cudaCheckError(cudaFree(d_C_max));
    cudaCheckError(cudaFree(d_deadline));
    cudaCheckError(cudaFree(d_priority));
    cudaCheckError(cudaFree(d_sus_min));
    cudaCheckError(cudaFree(d_sus_max));
    cudaCheckError(cudaFree(d_BCRT));
    cudaCheckError(cudaFree(d_WCRT));
    cudaCheckError(cudaFree(d_layer_buf[0]));
    cudaCheckError(cudaFree(d_layer_buf[1]));
    // cudaCheckError(cudaFree(d_max_popcount));  // A1: allocation removed

    cudaCheckError(cudaFreeHost(h_state_buf));
    cudaCheckError(cudaFreeHost(h_layer_buf_A));
    cudaCheckError(cudaFreeHost(h_layer_buf_B));
    if (h_spill_out) cudaCheckError(cudaFreeHost(h_spill_out));
    if (h_spill_in)  cudaCheckError(cudaFreeHost(h_spill_in));
    if (h_gm_stage_in)   cudaCheckError(cudaFreeHost(h_gm_stage_in));
    if (h_gm_stage_out)  cudaCheckError(cudaFreeHost(h_gm_stage_out));
    if (h_gm_batch_pack) cudaCheckError(cudaFreeHost(h_gm_batch_pack));
    cudaCheckError(cudaFreeHost(h_valid_count));
    cudaCheckError(cudaFreeHost(h_output_count));
    cudaCheckError(cudaFreeHost(h_unsched_flag));
    cudaCheckError(cudaFreeHost(h_trunc_flag));
    cudaCheckError(cudaFreeHost(h_merge_count));

    // Populate out-parameter for batch-mode aggregation.
    // ok=false + error="TRUNCATED" is returned when analysis couldn't fit the state
    // space; this is distinct from schedulable=false (which is a real UNSCHED verdict).
    if (result_out) {
        int bcrt_count_final = 0;
        for (int j = 0; j < N; j++) if (h_BCRT[j] < INT32_MAX) bcrt_count_final++;
        result_out->ok            = !analysis_failed;
        result_out->schedulable   = analysis_failed ? false : schedulable;
        result_out->wall_time_s   = wall_time_s;
        result_out->gpu_time_ms   = (double)(grand_total_time_k1 + grand_total_time_k2 + grand_total_time_merge);
        result_out->total_nodes   = grand_total_nodes;
        result_out->total_expanded= grand_total_expanded;
        result_out->total_merged  = grand_total_merged;
        result_out->jobs_tracked  = bcrt_count_final;
        result_out->N             = N;
        result_out->W             = W;
        result_out->M             = M;
        result_out->error         = analysis_failed ? std::string("TRUNCATED") : std::string();
    }

    printf("\nDone. All GPU memory freed.\n");
    #undef printf
    return 0;
    #undef RUN_FAIL
}

// ---------------------------------------------------------------------------
// Batch mode helpers
// ---------------------------------------------------------------------------
static bool path_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static bool is_directory(const std::string& p) {
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

// Discover taskset_* subdirectories (alphabetically sorted).
static std::vector<std::string> discover_tasksets(const std::string& base_dir)
{
    std::vector<std::string> out;
    DIR* d = opendir(base_dir.c_str());
    if (!d) {
        fprintf(stderr, "Error: cannot open directory '%s'\n", base_dir.c_str());
        return out;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        const char* name = ent->d_name;
        if (strncmp(name, "taskset_", 8) != 0) continue;
        std::string sub = base_dir + "/" + name;
        if (!is_directory(sub)) continue;
        // Must contain jobs.csv and jobsprec.csv
        if (!path_exists(sub + "/jobs.csv")) continue;
        if (!path_exists(sub + "/jobsprec.csv")) continue;
        out.push_back(name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// Run every taskset in `base_dir` in a single process, reusing the CUDA
// context (but reallocating per-run GPU buffers inside run_one_taskset).
static int run_batch(const std::string& base_dir, int M)
{
    auto batch_start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> tasksets = discover_tasksets(base_dir);
    if (tasksets.empty()) {
        fprintf(stderr, "Error: no taskset_* subdirectories with jobs.csv/jobsprec.csv in '%s'\n",
                base_dir.c_str());
        return 1;
    }

    fprintf(stdout, "\n=============================================================\n");
    fprintf(stdout, " BATCH MODE: %zu tasksets in '%s' (M=%d)\n",
            tasksets.size(), base_dir.c_str(), M);
    fprintf(stdout, "=============================================================\n");

    struct PerTs {
        std::string name;
        SolveResult r;
    };
    std::vector<PerTs> results;
    results.reserve(tasksets.size());

    // Warm up CUDA context once so the first taskset doesn't pay init.
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Use the raw CUDA runtime functions here (bypassing the cudaMalloc
        // -> cudaMallocAsync macro) so we actually trigger context init and
        // avoid passing nullptr to cudaFreeAsync.
        #undef cudaFree
        #undef cudaMalloc
        cudaFree(0);  // triggers context init
        cudaDeviceSynchronize();
        #define cudaMalloc(ptr, size) cudaMallocAsync((ptr), (size), (cudaStream_t)0)
        #define cudaFree(ptr)         cudaFreeAsync((ptr), (cudaStream_t)0)
        auto t1 = std::chrono::high_resolution_clock::now();
        double ctx_s = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stdout, "\n[batch] CUDA context warmup: %.3f s\n", ctx_s);
    }

    double total_gpu_ms = 0.0;
    double total_run_s = 0.0;
    int num_fail = 0;
    int num_unsched = 0;

    // Decide parallel width: environment override > default.
    // Default: 1 (serial). Multi-stream concurrency on these workloads
    // doesn't speed up wall clock because the K1/K2 kernels already
    // saturate the A100 SMs at single-taskset level — running multiple
    // streams concurrently just time-slices them. Override via
    // EXPAND_TEST_PARALLEL=N for experimentation.
    int par_n = 1;
    if (const char* env = getenv("EXPAND_TEST_PARALLEL")) {
        int v = atoi(env);
        if (v >= 1 && v <= 32) par_n = v;
    }
    if (par_n > (int)tasksets.size()) par_n = (int)tasksets.size();
    g_batch_par_n = par_n;  // propagate to run_one_taskset so wave/layer allocs scale down
    fprintf(stdout, "\n[batch] parallel width: %d threads (GPU memory partitioned %d ways)\n",
            par_n, par_n);
    fflush(stdout);

    // Pre-size results so parallel workers can write by index without locking.
    results.assign(tasksets.size(), PerTs{});
    for (size_t i = 0; i < tasksets.size(); i++) results[i].name = tasksets[i];

    // Mutex for the tiny amount of per-taskset logging we still emit.
    omp_lock_t log_lock;
    omp_init_lock(&log_lock);

    #pragma omp parallel for num_threads(par_n) schedule(dynamic, 1)
    for (int i = 0; i < (int)tasksets.size(); i++) {
        // Silence the chatty per-layer / per-wave output from run_one_taskset.
        g_quiet = true;

        std::string ts_dir   = base_dir + "/" + tasksets[i];
        std::string jobs_csv = ts_dir + "/jobs.csv";
        std::string prec_csv = ts_dir + "/jobsprec.csv";

        int tid = omp_get_thread_num();
        auto t_start = std::chrono::high_resolution_clock::now();

        omp_set_lock(&log_lock);
        fprintf(stdout, "[T%d] %s START\n", tid, tasksets[i].c_str());
        fflush(stdout);
        omp_unset_lock(&log_lock);

        int rc = run_one_taskset(jobs_csv.c_str(), prec_csv.c_str(),
                                 nullptr, M, &results[i].r);
        auto t_end = std::chrono::high_resolution_clock::now();
        double wall_this = std::chrono::duration<double>(t_end - t_start).count();

        omp_set_lock(&log_lock);
        if (rc != 0 || !results[i].r.ok) {
            fprintf(stdout, "[T%d] %s FAILED (%s)\n", tid, tasksets[i].c_str(),
                    results[i].r.error.empty() ? "unknown" : results[i].r.error.c_str());
        } else {
            fprintf(stdout,
                "[T%d] %s DONE wall=%.3fs run=%.3fs gpu=%.1fms %d/%d %s\n",
                tid, tasksets[i].c_str(),
                wall_this, results[i].r.wall_time_s, results[i].r.gpu_time_ms,
                results[i].r.jobs_tracked, results[i].r.N,
                results[i].r.schedulable ? "SCHED" : "UNSCHED");
        }
        fflush(stdout);
        omp_unset_lock(&log_lock);
    }

    omp_destroy_lock(&log_lock);

    // Aggregate.
    for (auto& e : results) {
        if (!e.r.ok) {
            num_fail++;
            continue;
        }
        if (!e.r.schedulable) num_unsched++;
        total_gpu_ms += e.r.gpu_time_ms;
        total_run_s  += e.r.wall_time_s;
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_wall = std::chrono::duration<double>(batch_end - batch_start).count();

    fprintf(stdout, "\n=============================================================\n");
    fprintf(stdout, " BATCH SUMMARY (%zu tasksets)\n", tasksets.size());
    fprintf(stdout, "=============================================================\n");
    fprintf(stdout, " Total wall clock:         %.3f s\n", batch_wall);
    fprintf(stdout, " Sum of per-run wall:      %.3f s\n", total_run_s);
    fprintf(stdout, " Sum of per-run GPU time:  %.3f ms\n", total_gpu_ms);
    fprintf(stdout, " Failed tasksets:          %d\n", num_fail);
    fprintf(stdout, " Unschedulable tasksets:   %d\n", num_unsched);
    fprintf(stdout, "\n per-taskset breakdown:\n");
    fprintf(stdout, "   %-14s %8s %10s %9s %6s %s\n",
            "taskset", "nodes", "merged", "wall(s)", "jobs", "status");
    for (auto& e : results) {
        if (!e.r.ok) {
            fprintf(stdout, "   %-14s %8s %10s %9s %6s FAIL: %s\n",
                    e.name.c_str(), "-", "-", "-", "-", e.r.error.c_str());
        } else {
            fprintf(stdout, "   %-14s %8lld %10lld %9.3f %5d/%-2d %s\n",
                    e.name.c_str(),
                    e.r.total_nodes, e.r.total_merged, e.r.wall_time_s,
                    e.r.jobs_tracked, e.r.N,
                    e.r.schedulable ? "SCHED" : "UNSCHED");
        }
    }
    fprintf(stdout, "=============================================================\n");

    return num_fail > 0 ? 2 : 0;
}

// ---------------------------------------------------------------------------
// main -- arg parsing and dispatch between single-taskset and batch mode
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Cache freed memory forever in the default pool so batch mode reuses
    // buffers from the first taskset across every subsequent taskset.
    {
        cudaMemPool_t mempool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&mempool, 0) == cudaSuccess && mempool) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    }

    int M = MAX_CORES;
    bool batch_mode = false;
    std::string batch_dir;

    // Parse flags: -m <cores>, --batch <dir>
    std::vector<char*> pos_args;
    pos_args.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
            if (M < 1 || M > MAX_CORES) {
                fprintf(stderr, "Error: -m must be between 1 and %d\n", MAX_CORES);
                return 1;
            }
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_mode = true;
            batch_dir = argv[++i];
        } else {
            pos_args.push_back(argv[i]);
        }
    }

    if (batch_mode) {
        if (!is_directory(batch_dir)) {
            fprintf(stderr, "Error: --batch argument '%s' is not a directory\n", batch_dir.c_str());
            return 1;
        }
        if (pos_args.size() > 1) {
            fprintf(stderr, "Warning: positional args ignored in --batch mode\n");
        }
        return run_batch(batch_dir, M);
    }

    // Single-taskset mode (legacy CLI).
    int pa = (int)pos_args.size();
    const char* jobs  = (pa >= 2) ? pos_args[1] : nullptr;
    const char* prec  = (pa >= 3) ? pos_args[2] : nullptr;
    const char* states= (pa >= 4) ? pos_args[3] : nullptr;

    if (pa != 1 && pa != 3 && pa != 4) {
        fprintf(stderr, "Usage:\n"
                        "  %s [-m cores]                              (hardcoded 5-job test)\n"
                        "  %s [-m cores] jobs.csv prec.csv [states.bin]\n"
                        "  %s --batch <dir> [-m cores]\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }

    SolveResult r;
    int rc = run_one_taskset(jobs, prec, states, M, &r);
    if (rc != 0) {
        fprintf(stderr, "run_one_taskset failed: %s\n", r.error.c_str());
        return rc;
    }
    return 0;
}
