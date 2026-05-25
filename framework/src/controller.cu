// controller.cu -- framework GPU-resident layer-loop controller (iter310,
// updated for sparse F-pair encoding).
//
// Sparse encoding (SAGStateLayoutV5, sag_types_v5.h): replaces dense
// F_pair[n] (n × 8 bytes) with sparse F_entries indexed by F_avail =
// F_mask | X. For n=140 workloads with typical popcount(F_mask|X) << n,
// this cuts per-state bytes by ~25-40%.
//
// Generic across A/H/B-series NV GPUs:
//   - cooperative_groups::this_grid().sync() in V5K1K2Kernel is SM_70+
//   - V5MergeKernel uses only warp-level shuffle / shared-mem (SM_70+)
//   - No Hopper-only features (TMA, wgmma, DSM/cluster-shared-mem)

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "sag_config.h"
#include "sag_types.h"            // ValidPair, IJPInfo (shared)
#include "sag_types_v5.h"         // SAGStateLayoutV5, FEntryV5
#include "const_job_arrays_v5.cuh" // c_C_min/c_C_max/c_deadline/c_r_min/c_priority decls
#include "ijp_detect_v5.cuh"      // IJPInfoV5, IJPDetectBodyV5
#include "k1_body_v5.cuh"
#include "k2_body_v5.cuh"
#include "merge_body_v5.cuh"

// Per-variant body headers. The K1/K2 forwarder bodies in
// k1_body_rtss17.cuh / k2_body_rtss17.cuh / k1_body_ecrts19.cuh /
// k2_body_ecrts19.cuh / k2_body_ecrts22.cuh are dead -- the cooperative
// kernels (V5K1K2KernelRTSS17/ECRTS19/ECRTS22 below) call K1BodyV5 /
// K2BodyV5 / K1BodyECRTS22 directly without going through them. Only
// the merge bodies + K1BodyECRTS22 are actually used.
#include "merge_body_rtss17.cuh"
#include "merge_body_ecrts19.cuh"
#include "k1_body_ecrts22.cuh"
#include "merge_body_ecrts22.cuh"

namespace cg = cooperative_groups;
using namespace sag;
using namespace sag::config;
using sag::v5::SAGStateLayoutV5;
using sag::v5::FEntryV5;
using sag::v5::ijp::IJPInfoV5;

// ---------------------------------------------------------------------------
// Read-only per-job arrays moved from global to constant memory.
// Forward decls live in include/const_job_arrays_v5.cuh; definitions here.
//
// K2 reads c_C_min[j], c_C_max[j], c_deadline[j], c_r_min[j] per warp where
// all 32 lanes broadcast j (k2_body_v5.cuh line ~265-268). Constant memory's
// broadcast cache beats L1 for this access pattern. K1 already shmem-caches
// priority/prio_order so we do not change K1; this targets K2 + IJP detect.
//
// 256 entries * 4 bytes = 1024 bytes per array; 5 arrays = 5 KB / 64 KB cmem.
// Generic across A/H/B-series NV GPUs (constant memory exists on all SM_70+).
// ---------------------------------------------------------------------------
__constant__ int32_t c_C_min   [V5_MAX_CONST_N];
__constant__ int32_t c_C_max   [V5_MAX_CONST_N];
__constant__ int32_t c_deadline[V5_MAX_CONST_N];
__constant__ int32_t c_r_min   [V5_MAX_CONST_N];
__constant__ int32_t c_priority[V5_MAX_CONST_N];

extern "C" bool V5_upload_const_job_arrays(
    const int32_t* h_C_min, const int32_t* h_C_max,
    const int32_t* h_deadline, const int32_t* h_r_min,
    const int32_t* h_priority, int N)
{
    if (N > V5_MAX_CONST_N) return false;
    cudaError_t e1 = cudaMemcpyToSymbol(c_C_min,    h_C_min,    (size_t)N * sizeof(int32_t));
    cudaError_t e2 = cudaMemcpyToSymbol(c_C_max,    h_C_max,    (size_t)N * sizeof(int32_t));
    cudaError_t e3 = cudaMemcpyToSymbol(c_deadline, h_deadline, (size_t)N * sizeof(int32_t));
    cudaError_t e4 = cudaMemcpyToSymbol(c_r_min,    h_r_min,    (size_t)N * sizeof(int32_t));
    cudaError_t e5 = cudaMemcpyToSymbol(c_priority, h_priority, (size_t)N * sizeof(int32_t));
    return (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess
            && e4 == cudaSuccess && e5 == cudaSuccess);
}

enum class V5Status : int32_t {
    UNKNOWN  = 0,
    SCHED    = 1,
    UNSCHED  = 2,
    TRUNC    = 3,
    EMPTY    = 4,
    MAX_LAYER_EXCEEDED = 5
};

// POR (Partial-Order Reduction) Phase B observability scaffold. When
// `g_por_observe = 1` (set via `cudaMemcpyToSymbol` in main when
// `SAG_V5_POR=1`), K1BodyV5's lane-0 priority-scan block also counts the
// number of states with multiple in-ready candidates (P1: |ES| >= 2).
// `g_por_dbg[]` indices: 0=states with |in_ready|>=2 (P1 candidates),
// 1=total K1 sub-phase-1 calls.
// Default off: K1 reads g_por_observe once per state (lane 0); compiler
// may keep it in a register. Branch is constant-after-init so the
// observation block is essentially free when off.
__device__ int g_por_observe = 0;
__device__ int g_por_dbg[8]  = {0};

extern "C" bool V5_set_por_observe(int enabled) {
    int v = enabled ? 1 : 0;
    cudaError_t e = cudaMemcpyToSymbol(g_por_observe, &v, sizeof(int));
    return e == cudaSuccess;
}

extern "C" bool V5_get_por_dbg(int* out, int n_ints) {
    if (!out || n_ints < 1) return false;
    int max_n = (n_ints < 8) ? n_ints : 8;
    cudaError_t e = cudaMemcpyFromSymbol(out, g_por_dbg, max_n * sizeof(int));
    return e == cudaSuccess;
}

extern "C" bool V5_clear_por_dbg() {
    int zeros[8] = {0};
    cudaError_t e = cudaMemcpyToSymbol(g_por_dbg, zeros, 8 * sizeof(int));
    return e == cudaSuccess;
}

constexpr int V5_BLOCK_WARPS = K2_WARPS_PER_BLOCK;     // 12
constexpr int V5_BLOCK_SIZE  = V5_BLOCK_WARPS * WARP_SIZE;  // 384

constexpr int V5_MERGE_WARPS_PER_BLOCK = MERGE_WARPS_PER_BLOCK;
constexpr int V5_MERGE_BLOCK_SIZE      = V5_MERGE_WARPS_PER_BLOCK * WARP_SIZE;

__host__ static inline size_t v5_k1_smem_bytes(int n) {
    // iter500: +2 * V5_BLOCK_WARPS * n * int32 for s_F_min, s_F_max
    // dense per-warp F-pair cache (replaces per-bit linear scans of the
    // sparse F_entries array in K1's two sub-phases).
    // Walk-sharing: +V5_BLOCK_WARPS * n * int32 for s_rho_min cache so
    // K1 phase 2 can read rho_min computed during phase 1's walk
    // instead of redoing the same `preds & F_mask` walk.
    // ECRTS22 gang-aware rho_hp: +2 * n * int32 for block-shared
    // s_p_max + s_p_min, hoisted from d_p_max/d_p_min globals so the
    // O(n) Eq. 9 walk reads from shmem instead of repeated global hits.
    // Allocated for all variants because V5_smem_bytes returns the max
    // across K1 layouts; the extra 8 KB at n=512 is well within budget.
    return (size_t)V5_BLOCK_WARPS * n * (sizeof(int32_t) + sizeof(bool))
         + (size_t)n * sizeof(int32_t)
         + (size_t)n * sizeof(int32_t)
         + (size_t)V5_BLOCK_WARPS * n * sizeof(int32_t)
         + (size_t)V5_BLOCK_WARPS * n * sizeof(int32_t)
         + (size_t)V5_BLOCK_WARPS * n * sizeof(int32_t)
         + (size_t)V5_BLOCK_WARPS * n * sizeof(int32_t)   // s_rho_min
         + (size_t)n * sizeof(int32_t)                     // s_p_max (ECRTS22)
         + (size_t)n * sizeof(int32_t);                    // s_p_min (ECRTS22)
}

__host__ static inline size_t v5_k2_smem_bytes(int W, int m, int n) {
    // iter500: +V5_BLOCK_WARPS * n * int32 for dense parent F_min cache
    // (replaces per-bit linear scans of par_entries in K2 STEP 5).
    // Round per-warp slice up to 8 bytes -- the device-side computation
    // does the same so each warp's s_D_new starts on a uint64_t-aligned
    // address (avoids misaligned 8-byte writes for odd N).
    size_t per_warp = 3 * W * sizeof(uint64_t)
                    + 2 * m * sizeof(int32_t)
                    + n * sizeof(int32_t);
    per_warp = (per_warp + 7) & ~static_cast<size_t>(7);
    return (size_t)V5_BLOCK_WARPS * per_warp;
}

__host__ static inline size_t v5_smem_bytes(int n, int W, int m) {
    size_t a = v5_k1_smem_bytes(n);
    size_t b = v5_k2_smem_bytes(W, m, n);
    return a > b ? a : b;
}

__host__ static inline size_t v5_merge_smem_bytes() {
    return V5_MERGE_WARPS_PER_BLOCK * MAX_SLOTS_PER_GROUP * sizeof(int);
}

// iter410: templated body so default-off (ENABLE_IJP=false) keeps the
// iter310 register footprint and is byte-identical to the prior V5 path.
// Two extern-C wrappers below pin instantiations and are dispatched by
// the host based on the SAG_V5_IJP env var.
template<bool ENABLE_IJP>
__device__ __forceinline__ void V5K1K2Body(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    IJPInfoV5* __restrict__       d_ijp_info,
    int                           layer_dbg)
{
    cg::grid_group g = cg::this_grid();
    extern __shared__ char smem[];

    const int lane         = threadIdx.x % WARP_SIZE;
    const int warpInBlock  = threadIdx.x / WARP_SIZE;
    const int globalWarp   = blockIdx.x * V5_BLOCK_WARPS + warpInBlock;
    const int total_warps  = gridDim.x * V5_BLOCK_WARPS;
    const int tid_in_block = threadIdx.x;

    // ---- Phase 1: K1 (eligibility) ----
    int round_up = ((num_states_in + total_warps - 1) / total_warps) * total_warps;
    if (round_up == 0) round_up = total_warps;
    for (int s = globalWarp; s < round_up; s += total_warps) {
        int eff_state = (s < num_states_in) ? s : -1;
        sag::v5::k1::K1BodyV5<V5_BLOCK_WARPS>(
            smem,
            d_layer_curr, layout, d_candidates, num_states_in, num_candidates,
            d_TC, d_PO, d_Pred,
            d_r_min, d_r_max, d_priority, d_prio_order,
            d_sus_min, d_sus_max,
            n, W,
            d_valid_pairs, d_valid_count, d_unsched_flag,
            max_valid_pairs, d_trunc_flag,
            eff_state, lane, warpInBlock, tid_in_block,
            // K1 deadline pruning: pass C_min and deadline arrays so K1
            // can skip statically-doomed dispatches before K2 sees them.
            d_C_min, d_deadline);
    }
    g.sync();

    int num_valid = *d_valid_count;
    if (num_valid > max_valid_pairs) num_valid = max_valid_pairs;

    if (num_valid == 0) return;
    if (*d_unsched_flag != 0) return;
    if (*d_trunc_flag   != 0) return;

    // ---- Phase 2: IJP detection (one warp per state) ----
    // Only compiled in when ENABLE_IJP=true; otherwise the body is empty
    // and grid_sync is unneeded (g.sync() following K1 is sufficient).
    if constexpr (ENABLE_IJP) {
        if (d_ijp_info != nullptr) {
            for (int s = globalWarp; s < num_states_in; s += total_warps) {
                sag::v5::ijp::IJPDetectBodyV5(
                    d_layer_curr, layout, d_candidates,
                    num_states_in, num_candidates,
                    d_TC, d_PO, d_Pred,
                    d_r_min, d_r_max, d_priority, d_C_max, d_deadline,
                    d_sus_min, d_sus_max,
                    n, m, W,
                    d_ijp_info,
                    s, lane);
            }
            g.sync();
        }
    }

    // ---- Phase 3: K2 (successor creation) ----
    for (int p = globalWarp; p < num_valid; p += total_warps) {
        sag::v5::K2BodyV5<ENABLE_IJP>(
            d_layer_curr, layout, d_valid_pairs, num_valid,
            d_Pred, d_Succ, d_C_min, d_C_max, d_deadline,
            n, m, W,
            d_layer_next, d_output_count, d_unsched_flag,
            layer_capacity,
            d_BCRT, d_WCRT, d_r_min, d_trunc_flag,
            d_max_F_count_observed,
            d_ijp_info,
            p, warpInBlock, lane, smem,
            layer_dbg);
    }
    g.sync();
}

extern "C" __global__ __launch_bounds__(V5_BLOCK_SIZE, 2)
void V5K1K2Kernel(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    int                           layer_dbg)
{
    V5K1K2Body<false>(
        layout, n, m, W,
        d_TC, d_PO, d_Pred, d_Succ,
        d_r_min, d_r_max, d_C_min, d_C_max, d_deadline,
        d_priority, d_prio_order, d_sus_min, d_sus_max,
        d_candidates, num_candidates,
        d_layer_curr, num_states_in,
        d_layer_next, layer_capacity,
        d_valid_pairs, max_valid_pairs,
        d_valid_count, d_output_count,
        d_unsched_flag, d_trunc_flag,
        d_BCRT, d_WCRT, d_max_F_count_observed,
        /*d_ijp_info=*/nullptr,
        layer_dbg);
}

extern "C" __global__ __launch_bounds__(V5_BLOCK_SIZE, 2)
void V5K1K2KernelIJP(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    IJPInfoV5* __restrict__       d_ijp_info,
    int                           layer_dbg)
{
    V5K1K2Body<true>(
        layout, n, m, W,
        d_TC, d_PO, d_Pred, d_Succ,
        d_r_min, d_r_max, d_C_min, d_C_max, d_deadline,
        d_priority, d_prio_order, d_sus_min, d_sus_max,
        d_candidates, num_candidates,
        d_layer_curr, num_states_in,
        d_layer_next, layer_capacity,
        d_valid_pairs, max_valid_pairs,
        d_valid_count, d_output_count,
        d_unsched_flag, d_trunc_flag,
        d_BCRT, d_WCRT, d_max_F_count_observed,
        d_ijp_info,
        layer_dbg);
}

// ---------------------------------------------------------------------------
// Phase 1 variant stub kernels.
//
// Each has the same signature as V5K1K2Kernel (no-IJP). The body:
//   1. prints a one-shot warning naming the variant + paper reference,
//   2. sets d_unsched_flag = 1 so the host driver records UNSCHED status,
//   3. returns without touching d_output_count / d_valid_count.
//
// In Phase 1 the host short-circuits unimplemented variants BEFORE launching
// any kernel (see framework_main.cu SAG_V5_VARIANT parsing), so these
// stubs are never reached at runtime. They exist as Phase 2-4 implementation
// slots: replacing the body with a real V5K1K2Body<...>-style call lights up
// the variant.
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(V5_BLOCK_SIZE, 2)
void V5K1K2KernelRTSS17(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    int                           layer_dbg)
{
    // RTSS 2017 K1+K2 path. K1BodyRTSS17 forwards to K1BodyV5 (m=1
    // simplifications fall out of the existing code); K2BodyRTSS17 forwards
    // to K2BodyV5<false>. No IJP path.
    //
    // Paper exactness is preserved by the conservative merge kernel
    // V5MergeKernelRTSS17 (defined below) -- NOT by this kernel.
    V5K1K2Body<false>(
        layout, n, m, W,
        d_TC, d_PO, d_Pred, d_Succ,
        d_r_min, d_r_max, d_C_min, d_C_max, d_deadline,
        d_priority, d_prio_order, d_sus_min, d_sus_max,
        d_candidates, num_candidates,
        d_layer_curr, num_states_in,
        d_layer_next, layer_capacity,
        d_valid_pairs, max_valid_pairs,
        d_valid_count, d_output_count,
        d_unsched_flag, d_trunc_flag,
        d_BCRT, d_WCRT, d_max_F_count_observed,
        /*d_ijp_info=*/nullptr,
        layer_dbg);
}

// ECRTS 2019 K1+K2 wrapper. With segment-expansion (host parser turns each
// task's segments into chain-of-jobs connected by zero-delay finish-to-start
// edges), the V5 default K1+K2 analysis already produces the limited-
// preemptive analysis: a segment becomes K1-eligible only after its
// predecessor segment has finished, and any higher-priority job's segment
// can dispatch at any same-state segment boundary.
//
// Phase 4 therefore forwards V5K1K2KernelECRTS19 to V5K1K2Body<false>
// verbatim. The "ECRTS 2019 vs RTSS 2024" delta lives entirely in the host
// parser (which expands segments) plus the merge kernel (V5 default lossy
// l1 -- matches nptest's published default for the LP DAG analysis).
extern "C" __global__ __launch_bounds__(V5_BLOCK_SIZE, 2)
void V5K1K2KernelECRTS19(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    int                           layer_dbg)
{
    V5K1K2Body<false>(
        layout, n, m, W,
        d_TC, d_PO, d_Pred, d_Succ,
        d_r_min, d_r_max, d_C_min, d_C_max, d_deadline,
        d_priority, d_prio_order, d_sus_min, d_sus_max,
        d_candidates, num_candidates,
        d_layer_curr, num_states_in,
        d_layer_next, layer_capacity,
        d_valid_pairs, max_valid_pairs,
        d_valid_count, d_output_count,
        d_unsched_flag, d_trunc_flag,
        d_BCRT, d_WCRT, d_max_F_count_observed,
        /*d_ijp_info=*/nullptr,
        layer_dbg);
}

// ECRTS 2022 cooperative-launch body. Same structure as V5K1K2Body<false>
// (no IJP) but calls K1BodyECRTS22 in Phase 1 (parallelism-aware eligibility,
// optional moldable greedy-earliest p enumeration) and K2BodyV5<false> in
// Phase 3 (multi-core commit via d_p_max or per-pair d_pair_p_arg). For p==1
// rigid sequential gang the body is byte-identical to V5K1K2Body<false>.
__device__ __forceinline__ void V5K1K2BodyECRTS22(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int32_t*  __restrict__  d_p_max,
    const int32_t*  __restrict__  d_p_min,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    // Phase 3.3.b CSR cost map + per-pair side arrays. Pass nullptrs to
    // disable moldable enumeration (rigid-gang path stays byte-identical
    // to the pre-3.3.b ECRTS22 behaviour).
    const int32_t* __restrict__   d_costmap_offset,
    const int32_t* __restrict__   d_costmap_p,
    const int32_t* __restrict__   d_costmap_cmin,
    const int32_t* __restrict__   d_costmap_cmax,
    int8_t*  __restrict__         d_pair_p,
    int32_t* __restrict__         d_pair_cmin,
    int32_t* __restrict__         d_pair_cmax,
    int                           layer_dbg)
{
    cg::grid_group g = cg::this_grid();
    extern __shared__ char smem[];

    const int lane         = threadIdx.x % WARP_SIZE;
    const int warpInBlock  = threadIdx.x / WARP_SIZE;
    const int globalWarp   = blockIdx.x * V5_BLOCK_WARPS + warpInBlock;
    const int total_warps  = gridDim.x * V5_BLOCK_WARPS;
    const int tid_in_block = threadIdx.x;

    // Phase 1: K1 (parallelism-aware eligibility).
    int round_up = ((num_states_in + total_warps - 1) / total_warps) * total_warps;
    if (round_up == 0) round_up = total_warps;
    for (int s = globalWarp; s < round_up; s += total_warps) {
        int eff_state = (s < num_states_in) ? s : -1;
        sag::v5::k1_ecrts22::K1BodyECRTS22<V5_BLOCK_WARPS>(
            smem,
            d_layer_curr, layout, d_candidates, num_states_in, num_candidates,
            d_TC, d_PO, d_Pred,
            d_r_min, d_r_max, d_priority,
            d_sus_min, d_sus_max,
            d_p_max,
            d_p_min,
            n, m, W,
            d_valid_pairs, d_valid_count, d_unsched_flag,
            max_valid_pairs, d_trunc_flag,
            eff_state, lane, warpInBlock, tid_in_block,
            // K1 deadline pruning: parity with V5K1K2Body. Skips
            // statically-doomed dispatches before K2 sees them.
            d_C_min, d_deadline, d_C_max,
            // Phase 3.3.b moldable enumeration. K1 picks p
            // greedy-earliest from CSR and writes (p, c_min, c_max)
            // into the per-pair side arrays.
            d_costmap_offset, d_costmap_p, d_costmap_cmin, d_costmap_cmax,
            d_pair_p, d_pair_cmin, d_pair_cmax);
    }
    g.sync();

    int num_valid = *d_valid_count;
    if (num_valid > max_valid_pairs) num_valid = max_valid_pairs;
    if (num_valid == 0) return;
    if (*d_unsched_flag != 0) return;
    if (*d_trunc_flag   != 0) return;

    // Phase 3.2 + 3.3.b: K2 with multi-core commit. The wrapper here
    // calls K2BodyV5<false> directly (rather than going through the
    // K2BodyECRTS22 forwarder) so we can pass the new per-pair side
    // arrays for moldable enumeration. K2BodyV5 falls back to
    // d_C_min[j] / d_C_max[j] / d_p_max[j] when the per-pair arrays
    // are null (rigid-gang or non-ECRTS22 paths).
    for (int p = globalWarp; p < num_valid; p += total_warps) {
        sag::v5::K2BodyV5</*ENABLE_IJP=*/false>(
            d_layer_curr, layout, d_valid_pairs, num_valid,
            d_Pred, d_Succ, d_C_min, d_C_max, d_deadline,
            n, m, W,
            d_layer_next, d_output_count, d_unsched_flag,
            layer_capacity,
            d_BCRT, d_WCRT, d_r_min, d_trunc_flag,
            d_max_F_count_observed,
            /*d_ijp_info=*/nullptr,
            p, warpInBlock, lane, smem, layer_dbg,
            d_p_max,
            d_pair_p, d_pair_cmin, d_pair_cmax);
    }
    g.sync();
}

extern "C" __global__ __launch_bounds__(V5_BLOCK_SIZE, 2)
void V5K1K2KernelECRTS22(
    SAGStateLayoutV5              layout,
    int n, int m, int W,
    const uint64_t* __restrict__  d_TC,
    const uint64_t* __restrict__  d_PO,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t*  __restrict__  d_r_min,
    const int32_t*  __restrict__  d_r_max,
    const int32_t*  __restrict__  d_C_min,
    const int32_t*  __restrict__  d_C_max,
    const int32_t*  __restrict__  d_deadline,
    const int32_t*  __restrict__  d_priority,
    const int32_t*  __restrict__  d_prio_order,
    const int32_t*  __restrict__  d_sus_min,
    const int32_t*  __restrict__  d_sus_max,
    const int32_t*  __restrict__  d_p_max,
    const int32_t*  __restrict__  d_p_min,
    const int*      __restrict__  d_candidates,
    int                           num_candidates,
    const char*     __restrict__  d_layer_curr,
    int                           num_states_in,
    char*           __restrict__  d_layer_next,
    int                           layer_capacity,
    ValidPair*      __restrict__  d_valid_pairs,
    int                           max_valid_pairs,
    int*  __restrict__            d_valid_count,
    int*  __restrict__            d_output_count,
    int*  __restrict__            d_unsched_flag,
    int*  __restrict__            d_trunc_flag,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    int*  __restrict__            d_max_F_count_observed,
    // Phase 3.3.b moldable enumeration plumbing.
    const int32_t* __restrict__   d_costmap_offset,
    const int32_t* __restrict__   d_costmap_p,
    const int32_t* __restrict__   d_costmap_cmin,
    const int32_t* __restrict__   d_costmap_cmax,
    int8_t*  __restrict__         d_pair_p,
    int32_t* __restrict__         d_pair_cmin,
    int32_t* __restrict__         d_pair_cmax,
    int                           layer_dbg)
{
    V5K1K2BodyECRTS22(
        layout, n, m, W,
        d_TC, d_PO, d_Pred, d_Succ,
        d_r_min, d_r_max, d_C_min, d_C_max, d_deadline,
        d_priority, d_prio_order, d_sus_min, d_sus_max,
        d_p_max,
        d_p_min,
        d_candidates, num_candidates,
        d_layer_curr, num_states_in,
        d_layer_next, layer_capacity,
        d_valid_pairs, max_valid_pairs,
        d_valid_count, d_output_count,
        d_unsched_flag, d_trunc_flag,
        d_BCRT, d_WCRT, d_max_F_count_observed,
        d_costmap_offset, d_costmap_p, d_costmap_cmin, d_costmap_cmax,
        d_pair_p, d_pair_cmin, d_pair_cmax,
        layer_dbg);
}

extern "C" __global__
void V5ExtractDKeysIotaKernel(
    const char* __restrict__ d_states,
    SAGStateLayoutV5 layout,
    const int* __restrict__  d_num_states_ptr,
    int W,
    uint64_t* __restrict__   d_dkeys_out,
    int* __restrict__        d_idx_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_states = *d_num_states_ptr;
    if (tid >= num_states) return;

    const uint64_t* D = layout.D(d_states, tid);
    for (int w = 0; w < W; w++) {
        d_dkeys_out[(long long)tid * W + w] = D[w];
    }
    d_idx_out[tid] = tid;
}

// iter370: variant that writes keys to d_dkeys_out[(out_offset + tid) * W ..]
// and the iota indices to d_idx_out[out_offset + tid] = idx_offset + tid.
// Used by the dual-source consolidate to extract keys from a SECOND state
// buffer (d_layer_scratch) into the merged keys array (offset after the
// first source's keys).
extern "C" __global__
void V5ExtractDKeysIotaOffsetKernel(
    const char* __restrict__ d_states,
    SAGStateLayoutV5 layout,
    int                       num_states,
    int                       out_offset,
    int                       idx_offset,
    int                       W,
    uint64_t* __restrict__    d_dkeys_out,
    int* __restrict__         d_idx_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_states) return;

    const uint64_t* D = layout.D(d_states, tid);
    int base = out_offset + tid;
    for (int w = 0; w < W; w++) {
        d_dkeys_out[(long long)base * W + w] = D[w];
    }
    d_idx_out[base] = idx_offset + tid;
}

extern "C" __global__
void V5DetectBoundariesKernel(
    const uint64_t* __restrict__ d_dkeys,
    const int*      __restrict__ d_sorted_idx,
    const int*      __restrict__ d_num_states_ptr,
    int W,
    int* __restrict__            d_is_start)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_states = *d_num_states_ptr;
    if (tid >= num_states) return;

    if (tid == 0) {
        d_is_start[0] = 1;
        return;
    }

    const uint64_t* Dp = d_dkeys + (long long)d_sorted_idx[tid - 1] * W;
    const uint64_t* Dc = d_dkeys + (long long)d_sorted_idx[tid    ] * W;
    bool differ = false;
    for (int w = 0; w < W; w++) {
        if (Dp[w] != Dc[w]) { differ = true; break; }
    }
    d_is_start[tid] = differ ? 1 : 0;
}

extern "C" __global__
void V5CompactGroupStartsKernel(
    const int* __restrict__ d_is_start,
    const int* __restrict__ d_group_id,
    const int* __restrict__ d_num_states_ptr,
    int* __restrict__       d_group_starts,
    int* __restrict__       d_num_groups)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_states = *d_num_states_ptr;
    if (tid >= num_states) return;

    if (d_is_start[tid]) {
        d_group_starts[d_group_id[tid]] = tid;
    }
    if (tid == num_states - 1) {
        *d_num_groups = d_group_id[tid] + d_is_start[tid];
    }
}

// iter370/iter390: V5MergeKernel with optional dual/triple-source input
// + batched group processing. Modes:
//   - Single-source: d_pre2 = nullptr. All sorted indices reference d_pre.
//   - Dual-source (iter370 consolidate): d_pre2 != nullptr, d_pre3 = nullptr,
//     pre1_count is the boundary. Indices in [0, pre1_count) read from d_pre.
//     Indices in [pre1_count, ...) read from d_pre2 at (idx - pre1_count).
//   - Triple-source (iter390 consolidate): d_pre2 != nullptr, d_pre3 != nullptr,
//     pre1_count and pre2_end are split points. Indices:
//       [0,           pre1_count) -> d_pre  at (idx)
//       [pre1_count,  pre2_end)   -> d_pre2 at (idx - pre1_count)
//       [pre2_end,    total)      -> d_pre3 at (idx - pre2_end)
//
// Batched group processing: kernel processes groups in
// [group_offset, group_offset + num_groups_in_batch) within
// [0, num_groups_total). For batch g, group g_idx = group_offset + g
// has start = d_group_starts[g_idx]; end = d_group_starts[g_idx+1] for
// g_idx+1 < num_groups_total, else *d_num_states_ptr.
// Per-variant compat-check dispatcher. Compile-time selection between V5's
// default predicate (action codes 0/1/2/3 as defined below) and variant-
// specific predicates that downgrade lossy widening (code 3 -> 0).
//
//   Variant == 0 : NPG_RTSS24 (default V5)             -- compat_and_dominance
//   Variant == 1 : NP_UNI_17  (RTSS 2017, exact)       -- compat_rtss17 (no widen)
//   Variant == 2 : LP_DAG_19  (ECRTS 2019, scaffold)   -- compat_ecrts19 stub (=0)
//   Variant == 3 : GANG_22    (ECRTS 2022, scaffold)   -- compat_ecrts22 stub (=0)
//
// For Phases 3/4 (LP_DAG_19, GANG_22) the body call wires up the variant
// compat predicate; the surrounding kernel structure is unchanged.
template<int VariantId>
__device__ __forceinline__ int v5_merge_compat_dispatch(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    if constexpr (VariantId == 1) {
        return sag::v5::merge_rtss17::dev_check_compat_rtss17(
            buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    } else if constexpr (VariantId == 2) {
        return sag::v5::merge_ecrts19::dev_check_compat_ecrts19(
            buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    } else if constexpr (VariantId == 3) {
        return sag::v5::merge_ecrts22::dev_check_compat_ecrts22(
            buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    } else {
        return sag::v5::merge::dev_check_compat_and_dominance(
            buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    }
}

// Templated merge-kernel body. Two extern "C" __global__ wrappers below pin
// instantiations VariantId=0 (default NPG_RTSS24) and VariantId=1 (RTSS 17).
// VariantId=2/3 wrappers will be added in Phases 3/4 when the variant compat
// predicates are real (Phase 1 stubs return 0 -> no merge -> would not
// produce useful output).
template<int VariantId>
__device__ __forceinline__ void V5MergeKernelBodyTpl(
    const char* __restrict__   d_pre,
    const char* __restrict__   d_pre2,
    int                        pre1_count,
    const char* __restrict__   d_pre3,
    int                        pre2_end,
    const int*  __restrict__   d_sorted_idx,
    const int*  __restrict__   d_group_starts,
    int                        group_offset,
    int                        num_groups_in_batch,
    int                        num_groups_total,
    const int*  __restrict__   d_num_states_ptr,
    SAGStateLayoutV5 layout,
    int n, int m, int W,
    char* __restrict__         d_post,
    int                        max_post_states,
    int*  __restrict__         d_merge_count,
    int*  __restrict__         d_trunc_flag,
    int                        layer_dbg)
{
    const int lane        = threadIdx.x & (WARP_SIZE - 1);
    const int warpInBlock = threadIdx.x / WARP_SIZE;
    const int globalWarp  = blockIdx.x * V5_MERGE_WARPS_PER_BLOCK + warpInBlock;

    if (globalWarp >= num_groups_in_batch) return;
    int g_idx = group_offset + globalWarp;
    if (g_idx >= num_groups_total) return;

    int g_start, g_size;
    if (lane == 0) {
        g_start = d_group_starts[g_idx];
        int g_end = (g_idx + 1 < num_groups_total)
                    ? d_group_starts[g_idx + 1] : *d_num_states_ptr;
        g_size = g_end - g_start;
    }
    g_start = __shfl_sync(0xFFFFFFFF, g_start, 0);
    g_size  = __shfl_sync(0xFFFFFFFF, g_size, 0);
    if (g_size <= 0) return;

    extern __shared__ int smem_merge[];
    int* my_slots = smem_merge + warpInBlock * MAX_SLOTS_PER_GROUP;
    int num_slots = 0;

    for (int s = 0; s < g_size; s++) {
        int orig_idx = d_sorted_idx[g_start + s];

        // iter370/iter390: resolve source buffer + index based on
        // dual/triple-source split.
        const char* src_buf;
        int         src_idx;
        if (d_pre2 == nullptr || orig_idx < pre1_count) {
            src_buf = d_pre;
            src_idx = orig_idx;
        } else if (d_pre3 == nullptr || orig_idx < pre2_end) {
            src_buf = d_pre2;
            src_idx = orig_idx - pre1_count;
        } else {
            src_buf = d_pre3;
            src_idx = orig_idx - pre2_end;
        }

        // Dominance pruning (iter620+): for each slot, check whether the
        // incoming state B (src) is fully dominated by slot A (drop B), or
        // B fully dominates A (replace slot with B), or neither (normal
        // widening merge). Action codes:
        //   0 = no compatible slot at this lane
        //   1 = slot A fully dominates B -> drop B
        //   2 = B fully dominates slot A -> replace slot contents with B
        //   3 = compatible but neither fully dominates -> normal widening merge
        //
        // RTSS 2017 (VariantId=1) downgrades 3->0 inside the dispatcher to
        // preserve the paper's exactness; both states then carry to L+1.
        int merge_action = 0;
        int target_slot_local = -1;

        if (lane < num_slots) {
            int sl_idx = my_slots[lane];
            int code = v5_merge_compat_dispatch<VariantId>(
                d_post, sl_idx,            // A = slot (dst)
                src_buf, src_idx,          // B = incoming (src)
                layout, n, m, W);
            merge_action = code;
            if (code != 0) target_slot_local = sl_idx;
        }

        // Replicate original "first compatible lane" semantics so that
        // byte-identical output is preserved: pick the first lane whose
        // slot is compatible with B, then act on its dominance code.
        unsigned ballot_compat = __ballot_sync(0xFFFFFFFF, merge_action != 0);

        if (ballot_compat != 0) {
            int first_lane    = __ffs(ballot_compat) - 1;
            int chosen_action = __shfl_sync(0xFFFFFFFF, merge_action,      first_lane);
            int target_slot   = __shfl_sync(0xFFFFFFFF, target_slot_local, first_lane);

            if (chosen_action == 1) {
                // Drop B: A>=B fully, merge would produce A unchanged.
            } else if (chosen_action == 2) {
                // Replace slot with B (B fully dominates A; merge result==B).
                sag::v5::merge::dev_copy_full_state_warp(
                    src_buf, src_idx,
                    d_post, target_slot,
                    layout, n, m, W, lane);
            } else {
                // Normal widening merge, lane 0 sequential body matches the
                // pre-dominance path (preserves byte-identical merge output).
                // For VariantId=1 (RTSS 17) chosen_action will never be 3 here
                // because the dispatcher downgrades 3->0; this branch is
                // effectively dead code for that variant.
                if (lane == 0) {
                    // ECRTS19 (VariantId=2) gets the segment-deadline clamp
                    // on widened f_max. Other variants use the default
                    // (no clamp), preserving byte-identity to pre-fix.
                    if constexpr (VariantId == 2) {
                        sag::v5::merge::dev_merge_into_slot<true>(
                            src_buf, src_idx,
                            d_post, target_slot,
                            layout, n, m, W,
                            d_trunc_flag,
                            layer_dbg);
                    } else {
                        sag::v5::merge::dev_merge_into_slot<false>(
                            src_buf, src_idx,
                            d_post, target_slot,
                            layout, n, m, W,
                            d_trunc_flag,
                            layer_dbg);
                    }
                }
            }
        } else {
            int slot_idx;
            if (lane == 0) {
                slot_idx = atomicAdd(d_merge_count, 1);
            }
            slot_idx = __shfl_sync(0xFFFFFFFF, slot_idx, 0);

            // Phase 4.2 bounds check: when a tightening predicate (e.g.
            // ECRTS 2019 disagreement-rejection) refuses many lossy
            // widenings, slot_idx can outrun the d_post buffer. Set
            // d_trunc_flag and skip the write rather than OOB.
            // max_post_states <= 0 means "unbounded" (legacy behaviour
            // when caller cannot supply a tight bound).
            if (max_post_states > 0 && slot_idx >= max_post_states) {
                if (lane == 0) atomicExch(d_trunc_flag, 1);
                return;
            }

            sag::v5::merge::dev_copy_full_state_warp(
                src_buf, src_idx,
                d_post, slot_idx,
                layout, n, m, W, lane);

            if (lane == 0 && num_slots < MAX_SLOTS_PER_GROUP) {
                my_slots[num_slots] = slot_idx;
            }
            __syncwarp(0xFFFFFFFF);
            if (num_slots < MAX_SLOTS_PER_GROUP) num_slots++;
        }

        __syncwarp(0xFFFFFFFF);
    }
}

// Note: launch_bounds(V5_MERGE_BLOCK_SIZE, 4) was tried and REGRESSED
// wall by ~15% (n16_m4_u50/t004: 850ms -> 975ms). The compiler's default
// register allocation already gives optimal occupancy for this workload;
// forcing 4 blocks/SM caused register spills. Don't retry.
extern "C" __global__
void V5MergeKernel(
    const char* __restrict__   d_pre,
    const char* __restrict__   d_pre2,
    int                        pre1_count,
    const char* __restrict__   d_pre3,
    int                        pre2_end,
    const int*  __restrict__   d_sorted_idx,
    const int*  __restrict__   d_group_starts,
    int                        group_offset,
    int                        num_groups_in_batch,
    int                        num_groups_total,
    const int*  __restrict__   d_num_states_ptr,
    SAGStateLayoutV5 layout,
    int n, int m, int W,
    char* __restrict__         d_post,
    int                        max_post_states,
    int*  __restrict__         d_merge_count,
    int*  __restrict__         d_trunc_flag,
    int                        layer_dbg)
{
    V5MergeKernelBodyTpl<0>(
        d_pre, d_pre2, pre1_count, d_pre3, pre2_end,
        d_sorted_idx, d_group_starts,
        group_offset, num_groups_in_batch, num_groups_total,
        d_num_states_ptr, layout, n, m, W,
        d_post, max_post_states, d_merge_count, d_trunc_flag, layer_dbg);
}

extern "C" __global__
void V5MergeKernelRTSS17(
    const char* __restrict__   d_pre,
    const char* __restrict__   d_pre2,
    int                        pre1_count,
    const char* __restrict__   d_pre3,
    int                        pre2_end,
    const int*  __restrict__   d_sorted_idx,
    const int*  __restrict__   d_group_starts,
    int                        group_offset,
    int                        num_groups_in_batch,
    int                        num_groups_total,
    const int*  __restrict__   d_num_states_ptr,
    SAGStateLayoutV5 layout,
    int n, int m, int W,
    char* __restrict__         d_post,
    int                        max_post_states,
    int*  __restrict__         d_merge_count,
    int*  __restrict__         d_trunc_flag,
    int                        layer_dbg)
{
    V5MergeKernelBodyTpl<1>(
        d_pre, d_pre2, pre1_count, d_pre3, pre2_end,
        d_sorted_idx, d_group_starts,
        group_offset, num_groups_in_batch, num_groups_total,
        d_num_states_ptr, layout, n, m, W,
        d_post, max_post_states, d_merge_count, d_trunc_flag, layer_dbg);
}

extern "C" __global__
void V5MergeKernelECRTS19(
    const char* __restrict__   d_pre,
    const char* __restrict__   d_pre2,
    int                        pre1_count,
    const char* __restrict__   d_pre3,
    int                        pre2_end,
    const int*  __restrict__   d_sorted_idx,
    const int*  __restrict__   d_group_starts,
    int                        group_offset,
    int                        num_groups_in_batch,
    int                        num_groups_total,
    const int*  __restrict__   d_num_states_ptr,
    SAGStateLayoutV5 layout,
    int n, int m, int W,
    char* __restrict__         d_post,
    int                        max_post_states,
    int*  __restrict__         d_merge_count,
    int*  __restrict__         d_trunc_flag,
    int                        layer_dbg)
{
    V5MergeKernelBodyTpl<2>(
        d_pre, d_pre2, pre1_count, d_pre3, pre2_end,
        d_sorted_idx, d_group_starts,
        group_offset, num_groups_in_batch, num_groups_total,
        d_num_states_ptr, layout, n, m, W,
        d_post, max_post_states, d_merge_count, d_trunc_flag, layer_dbg);
}

extern "C" __global__
void V5MergeKernelECRTS22(
    const char* __restrict__   d_pre,
    const char* __restrict__   d_pre2,
    int                        pre1_count,
    const char* __restrict__   d_pre3,
    int                        pre2_end,
    const int*  __restrict__   d_sorted_idx,
    const int*  __restrict__   d_group_starts,
    int                        group_offset,
    int                        num_groups_in_batch,
    int                        num_groups_total,
    const int*  __restrict__   d_num_states_ptr,
    SAGStateLayoutV5 layout,
    int n, int m, int W,
    char* __restrict__         d_post,
    int                        max_post_states,
    int*  __restrict__         d_merge_count,
    int*  __restrict__         d_trunc_flag,
    int                        layer_dbg)
{
    V5MergeKernelBodyTpl<3>(
        d_pre, d_pre2, pre1_count, d_pre3, pre2_end,
        d_sorted_idx, d_group_starts,
        group_offset, num_groups_in_batch, num_groups_total,
        d_num_states_ptr, layout, n, m, W,
        d_post, max_post_states, d_merge_count, d_trunc_flag, layer_dbg);
}

// ---------------------------------------------------------------------------
// Variable-width F_entries spill kernels.
//
// V5 normally spills states with the dense per-state stride F_MAX_PER_STATE
// FEntryV5 slots (default 32 -> 384 bytes), even when typical observed
// F_count is 10-21. The unused slots are padding on PCIe.
//
// Variable-width spill packs each chunk to packed_bps bytes/state where
// packed_bps reserves only max_F_used FEntryV5 slots (across all states in the
// chunk). Pack/unpack happens at PCIe boundaries only -- the GPU kernels that
// run on resident buffers continue to use the dense bps for SIMD coalescing.
//
// The packed layout matches the dense layout slot-for-slot up through the
// F_count field, then has only max_F_used FEntryV5 entries (instead of
// F_MAX_PER_STATE), then the ovf int32, then 8-byte alignment padding. The
// math is byte-identical because:
//   * Pack copies bytes 0 .. (header_end + max_F_used*12) verbatim;
//   * The unused F_entries slots [F_count, F_MAX_PER_STATE) are unread by
//     any kernel (sparse_F_lookup walks F_count entries; merge body iterates
//     the F_avail bitmask up to popcount, which equals F_count). Because
//     dense F_count <= max_F_used by chunk-max definition, all VALID entries
//     are preserved -- the discarded slots are only zero/garbage padding.
//
// Generic SM_70+: only basic memory ops, no Hopper-only features.
// ---------------------------------------------------------------------------

// Compute the dense layout's "header bytes" -- everything up to and including
// F_count. State[i]'s header occupies [i*bps + 0, i*bps + header_bytes).
__host__ __device__ static inline int v5_layout_header_bytes(int W, int m) {
    return 3 * W * (int)sizeof(uint64_t)
         + m * (int)sizeof(int2)
         + (int)sizeof(int32_t);
}

// Compute the packed bps for a chunk given `max_F_used` (the chunk-wide max
// F_count). 8-byte aligned to match SAGStateLayoutV5.
__host__ __device__ static inline int v5_packed_bps(int W, int m, int max_F_used) {
    int raw = 3 * W * (int)sizeof(uint64_t)
            + m * (int)sizeof(int2)
            + (int)sizeof(int32_t)
            + max_F_used * (int)sizeof(FEntryV5)
            + (int)sizeof(int32_t);
    return (raw + 7) & ~7;
}

// V5ChunkMaxFCountKernel -- one thread per state, atomicMax into a single
// global counter d_max_out (callers reset to 0 before launch). Coalesced
// reads of the F_count int32 inside each state's dense layout.
extern "C" __global__
void V5ChunkMaxFCountKernel(
    const char* __restrict__ d_states,
    SAGStateLayoutV5 layout,
    int                       count,
    int* __restrict__         d_max_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Block-wide reduce before the global atomic to cut atomic traffic by
    // ~32x for typical block sizes; final atomicMax with the block's max.
    __shared__ int s_max;
    if (threadIdx.x == 0) s_max = 0;
    __syncthreads();
    if (tid < count) {
        int fc = *(layout.F_count((char*)d_states, tid));
        atomicMax(&s_max, fc);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(d_max_out, s_max);
    }
}

// V5PackForSpillKernel -- reads each state from the dense layout and writes
// only its [header + max_F_used FEntries + ovf + padding] bytes into an
// output buffer with stride packed_bps. One warp per state for coalesced
// 32-byte loads/stores; lane = thread byte-offset within the per-state copy.
extern "C" __global__
void V5PackForSpillKernel(
    const char* __restrict__ d_in,
    SAGStateLayoutV5 in_layout,
    char* __restrict__       d_out,       // packed buffer
    int                       packed_bps,
    int                       max_F_used,
    int                       count,
    int                       W,
    int                       m)
{
    const int tid_in_block = threadIdx.x;
    const int lane         = tid_in_block & (WARP_SIZE - 1);
    const int warpInBlock  = tid_in_block / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int globalWarp   = blockIdx.x * warps_per_block + warpInBlock;
    if (globalWarp >= count) return;

    const int dense_bps     = in_layout.bytes_per_state();
    const int header_bytes  = v5_layout_header_bytes(W, m);
    const int entries_bytes = max_F_used * (int)sizeof(FEntryV5);
    // packed_bps reserves header + entries_bytes + ovf_bytes(=4) rounded
    // up to 8; padding is dont-care, left
    // as garbage in d_out (stride only -- never read by any kernel beyond the
    // unpack which restores the dense layout from data_bytes anyway).

    const char* src = d_in  + (long long)globalWarp * dense_bps;
    char*       dst = d_out + (long long)globalWarp * packed_bps;

    // Phase 1: copy header + max_F_used FEntries verbatim, byte-coalesced.
    // We use 4-byte (int32) word strides since header is 4-byte aligned and
    // FEntryV5 is 12 bytes (3 ints).
    int header_words = (header_bytes + entries_bytes) / 4;
    for (int w = lane; w < header_words; w += WARP_SIZE) {
        ((int32_t*)dst)[w] = ((const int32_t*)src)[w];
    }

    // Phase 2: copy the ovf int32 from its dense offset to its packed offset.
    if (lane == 0) {
        int dense_ovf_off  = header_bytes
                           + sag::v5::F_MAX_PER_STATE * (int)sizeof(FEntryV5);
        int packed_ovf_off = header_bytes + entries_bytes;
        *(int32_t*)(dst + packed_ovf_off) =
            *(const int32_t*)(src + dense_ovf_off);
    }
}

// V5UnpackFromSpillKernel -- reverse of pack. Reads each state from a packed
// buffer with stride packed_bps and writes a dense state with stride
// dense_bps. The unused F_entries slots [max_F_used, F_MAX_PER_STATE) are
// zero-filled to preserve byte-identical math (some lookups happen to read
// past F_count via the sparse_F_lookup early-exit, and zero bytes match what
// dense always writes for empty slots).
extern "C" __global__
void V5UnpackFromSpillKernel(
    const char* __restrict__ d_in,        // packed buffer
    int                       packed_bps,
    int                       max_F_used,
    char* __restrict__        d_out,      // dense buffer
    SAGStateLayoutV5 out_layout,
    int                       count,
    int                       W,
    int                       m)
{
    const int tid_in_block = threadIdx.x;
    const int lane         = tid_in_block & (WARP_SIZE - 1);
    const int warpInBlock  = tid_in_block / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int globalWarp   = blockIdx.x * warps_per_block + warpInBlock;
    if (globalWarp >= count) return;

    const int dense_bps     = out_layout.bytes_per_state();
    const int header_bytes  = v5_layout_header_bytes(W, m);
    const int entries_bytes = max_F_used * (int)sizeof(FEntryV5);

    const char* src = d_in  + (long long)globalWarp * packed_bps;
    char*       dst = d_out + (long long)globalWarp * dense_bps;

    // Phase 1: copy header + max_F_used FEntries verbatim into the dense
    // state's [0 .. header+entries_bytes) region.
    int header_words = (header_bytes + entries_bytes) / 4;
    for (int w = lane; w < header_words; w += WARP_SIZE) {
        ((int32_t*)dst)[w] = ((const int32_t*)src)[w];
    }

    // Phase 2: zero out the dense F_entries slots [max_F_used, F_MAX_PER_STATE).
    // These are the slots beyond what the chunk's max_F_used covers; they
    // must read as zeros to match what kernels see in fresh dense states.
    int unused_entry_words =
        (sag::v5::F_MAX_PER_STATE - max_F_used) * (int)sizeof(FEntryV5) / 4;
    int unused_dst_off_words = (header_bytes + entries_bytes) / 4;
    for (int w = lane; w < unused_entry_words; w += WARP_SIZE) {
        ((int32_t*)dst)[unused_dst_off_words + w] = 0;
    }

    // Phase 3: copy ovf from packed offset to dense offset.
    if (lane == 0) {
        int packed_ovf_off = header_bytes + entries_bytes;
        int dense_ovf_off  = header_bytes
                           + sag::v5::F_MAX_PER_STATE * (int)sizeof(FEntryV5);
        *(int32_t*)(dst + dense_ovf_off) =
            *(const int32_t*)(src + packed_ovf_off);
    }
}

extern "C" int V5_layout_header_bytes(int W, int m) {
    return v5_layout_header_bytes(W, m);
}
extern "C" int V5_packed_bps(int W, int m, int max_F_used) {
    return v5_packed_bps(W, m, max_F_used);
}

extern "C" size_t V5_smem_bytes(int n, int W, int m) {
    return v5_smem_bytes(n, W, m);
}
extern "C" size_t V5_merge_smem_bytes() {
    return v5_merge_smem_bytes();
}
extern "C" const void* V5_k1k2_kernel_ptr() {
    return reinterpret_cast<const void*>(&V5K1K2Kernel);
}
extern "C" const void* V5_k1k2_kernel_ijp_ptr() {
    return reinterpret_cast<const void*>(&V5K1K2KernelIJP);
}
extern "C" const void* V5_k1k2_kernel_rtss17_ptr() {
    return reinterpret_cast<const void*>(&V5K1K2KernelRTSS17);
}
extern "C" const void* V5_k1k2_kernel_ecrts19_ptr() {
    return reinterpret_cast<const void*>(&V5K1K2KernelECRTS19);
}
extern "C" const void* V5_k1k2_kernel_ecrts22_ptr() {
    return reinterpret_cast<const void*>(&V5K1K2KernelECRTS22);
}
extern "C" const void* V5_merge_kernel_ptr() {
    return reinterpret_cast<const void*>(&V5MergeKernel);
}
extern "C" const void* V5_merge_kernel_rtss17_ptr() {
    return reinterpret_cast<const void*>(&V5MergeKernelRTSS17);
}
extern "C" const void* V5_merge_kernel_ecrts19_ptr() {
    return reinterpret_cast<const void*>(&V5MergeKernelECRTS19);
}
extern "C" const void* V5_merge_kernel_ecrts22_ptr() {
    return reinterpret_cast<const void*>(&V5MergeKernelECRTS22);
}
extern "C" int V5_block_size() {
    return V5_BLOCK_SIZE;
}
extern "C" int V5_block_warps() {
    return V5_BLOCK_WARPS;
}
extern "C" int V5_merge_block_size() {
    return V5_MERGE_BLOCK_SIZE;
}
extern "C" int V5_merge_warps_per_block() {
    return V5_MERGE_WARPS_PER_BLOCK;
}
