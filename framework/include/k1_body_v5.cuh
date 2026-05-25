// k1_body_v5.cuh -- K1 (FusedEligibility) body for framework sparse layout.
//
// Mirrors include/k1_body.cuh, but reads F_min / F_max from the sparse
// SAGStateLayoutV5 encoding. Hot reads in sub-phase 1 (rho_max via st_F_max[i])
// and sub-phase 2 (rho_min via st_F_min[i]) iterate over `preds[c] & F_mask`,
// which is a subset of F_mask, hence guaranteed to have a sparse F_entry.
//
// The lookup is a linear scan over F_entries; for the typical
// popcount(F_mask|X) <= 30 on n=140 workloads this is fast (1-2 cache lines).
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include "sag_config.h"
#include "sag_types_v5.h"

// POR observation globals (defined in controller.cu).
// Phase B scaffold: K1 lane-0 increments g_por_dbg[0] per state that has
// |in_ready| >= 2 candidates (the P1 predicate). When g_por_observe=0
// (default), the observation branch is constant-folded out at runtime.
extern __device__ int g_por_observe;
extern __device__ int g_por_dbg[8];

namespace sag {
namespace v5 {
namespace k1 {

using namespace sag::config;

__device__ __forceinline__ int v5_k1_ctz64(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ bool v5_k1_bit_test(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

__device__ __forceinline__ bool v5_k1_is_subset(const uint64_t* mask,
                                                const uint64_t* set, int W) {
    for (int w = 0; w < W; w++) {
        if (mask[w] & ~set[w]) return false;
    }
    return true;
}

__device__ __forceinline__ int32_t v5_k1_warp_min(int32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = min(val, other);
    }
    return val;
}

template<int WPB>
__device__ __forceinline__ void K1BodyV5(
    char* smem_base,
    const char* __restrict__     d_input,
    SAGStateLayoutV5             layout,
    const int* __restrict__      d_candidates,
    int num_states,
    int num_candidates,
    const uint64_t* __restrict__ d_TC,
    const uint64_t* __restrict__ d_PO,
    const uint64_t* __restrict__ d_Pred,
    const int32_t* __restrict__  d_r_min,
    const int32_t* __restrict__  d_r_max,
    const int32_t* __restrict__  d_priority,
    const int32_t* __restrict__  d_prio_order,
    const int32_t* __restrict__  d_sus_min,
    const int32_t* __restrict__  d_sus_max,
    int n, int W,
    ValidPair* __restrict__      d_valid_pairs,
    int* __restrict__            d_valid_count,
    int* __restrict__            d_unschedulable_flag,
    int max_valid_pairs,
    int* __restrict__            d_trunc_flag,
    int                          stateIdx,
    int                          lane,
    int                          warpInBlock,
    int                          tid_in_block,
    // K1 deadline pruning (optional; pass nullptrs to disable).
    // When provided, pre-check rho_min(j) + C_min(j) > deadline(j) before
    // emitting a ValidPair; skip the candidate (and let K2's deadline
    // check, which is the authoritative gate, set the unsched flag if
    // appropriate). This is a state-count-reduction optimization;
    // correctness is unchanged because K2 already enforces the deadline.
    const int32_t* __restrict__  d_C_min_dl   = nullptr,
    const int32_t* __restrict__  d_deadline_dl = nullptr)
{
    int32_t* s_rho_max       = (int32_t*)smem_base;
    bool*    s_in_ready      = (bool*)(smem_base + WPB * n * sizeof(int32_t));
    int32_t* s_priority      = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool));
    int32_t* s_prio_order    = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t));
    int32_t* s_rho_min_below = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + n * sizeof(int32_t));
    // iter500: per-warp dense F-pair cache.  V5's sparse F_entries[] is
    // looked up in the K1 inner loop via linear scan (sparse_F_max in sub-
    // phase 1, sparse_F_min in sub-phase 2). For each candidate we walk
    // popcount(preds[c] & F_mask) bits, and each bit triggers a scan of up
    // to F_count entries -- O(num_candidates * popcount * F_count) per state.
    //
    // Once we know st_F_entries / st_F_count, we scatter the (job_idx, f_min,
    // f_max) tuples into a dense per-warp cache and then read s_F_max[i] /
    // s_F_min[i] in O(1). Lookups outside F_avail return 0, matching
    // sparse_F_lookup's miss behaviour.
    //
    // The cache is byte-identical to the sparse lookup because we pre-zero
    // the WPB*n slots and only scatter the entries that are present.
    int32_t* s_F_min         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));
    int32_t* s_F_max         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));
    // Walk-sharing: phase 1 walks `preds[c] & F_mask` to compute rho_max
    // using d_sus_max + my_F_max; phase 2 used to walk the SAME bits for
    // rho_min using d_sus_min + my_F_min. We now compute rho_min in phase
    // 1's walk and cache it here -- phase 2 reads s_rho_min[j] in O(1)
    // instead of redoing the walk. Byte-identical (same bits, same
    // accumulation, same memory reads -- just consolidated).
    int32_t* s_rho_min       = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));

    for (int t = tid_in_block; t < n; t += WPB * WARP_SIZE) {
        s_priority[t]   = d_priority[t];
        s_prio_order[t] = d_prio_order[t];
    }
    for (int j = lane; j < n; j += WARP_SIZE) {
        s_rho_max[warpInBlock * n + j]  = INF_TIME;
        s_rho_min[warpInBlock * n + j]  = INF_TIME;
        s_in_ready[warpInBlock * n + j] = false;
        s_F_min[warpInBlock * n + j]    = 0;
        s_F_max[warpInBlock * n + j]    = 0;
    }
    __syncthreads();

    if (stateIdx < 0 || stateIdx >= num_states) return;

    const uint64_t* st_D      = layout.D(d_input, stateIdx);
    const uint64_t* st_F_mask = layout.F_mask(d_input, stateIdx);
    const int2*     st_A_pair = layout.A_pair(d_input, stateIdx);
    const FEntryV5* st_F_entries = layout.F_entries(d_input, stateIdx);
    int             st_F_count   = *layout.F_count(d_input, stateIdx);

    // iter500: scatter sparse F_entries into the dense cache for this state.
    // F_count <= F_MAX_PER_STATE (default 32), so a lane-strided loop is
    // typically a single warp pass.
    int32_t* my_F_min = s_F_min + warpInBlock * n;
    int32_t* my_F_max = s_F_max + warpInBlock * n;
    for (int e = lane; e < st_F_count; e += WARP_SIZE) {
        FEntryV5 fe = st_F_entries[e];
        my_F_min[fe.job()] = fe.f_min;
        my_F_max[fe.job()] = fe.f_max;
    }
    __syncwarp(0xFFFFFFFF);

    // Sub-phase 1: rho_max + rho_min in one walk (walk-sharing).
    #pragma unroll 2
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int c = d_candidates[ci];
        if (v5_k1_bit_test(st_D, c)) continue;
        const uint64_t* preds = &d_Pred[c * W];
        if (!v5_k1_is_subset(preds, st_D, W)) continue;
        int32_t rho_max_c = d_r_max[c];
        int32_t rho_min_c = d_r_min[c];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds[w] & st_F_mask[w];
            while (pbits) {
                int bit = v5_k1_ctz64(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                int32_t fmax_i = my_F_max[i];
                int32_t fmin_i = my_F_min[i];
                rho_max_c = max(rho_max_c, fmax_i + d_sus_max[c * n + i]);
                rho_min_c = max(rho_min_c, fmin_i + d_sus_min[c * n + i]);
            }
        }
        s_rho_max[warpInBlock * n + c]  = rho_max_c;
        s_rho_min[warpInBlock * n + c]  = rho_min_c;
        s_in_ready[warpInBlock * n + c] = true;
    }
    __syncwarp(0xFFFFFFFF);

    if (lane == 0) {
        // POR Phase B P1+P6 observation. Default-off has zero overhead
        // (g_por_observe is 0 from BSS; the branch is constant after
        // first read).
        // dbg[0] = P1 count: |ES| >= 2 (multi-ready)
        // dbg[1] = total K1 sub-phase-1 invocations
        // dbg[2] = P6 count: 2 <= |ES| <= m (fits on m cores)
        // dbg[3] = states with |ES| > m (P6 fails -- POR blocked)
        // dbg[4] = states with exactly 1 ready candidate (POR vacuous)
        // dbg[5] = states with 0 ready (terminal)
        if (g_por_observe) {
            int n_ready = 0;
            for (int j = 0; j < n; j++) {
                if (s_in_ready[warpInBlock * n + j]) n_ready++;
            }
            atomicAdd(&g_por_dbg[1], 1);
            int m_local = layout.m;
            if (n_ready == 0) atomicAdd(&g_por_dbg[5], 1);
            else if (n_ready == 1) atomicAdd(&g_por_dbg[4], 1);
            else {  // n_ready >= 2
                atomicAdd(&g_por_dbg[0], 1);
                if (n_ready <= m_local) atomicAdd(&g_por_dbg[2], 1);
                else                    atomicAdd(&g_por_dbg[3], 1);
            }
        }

        int32_t running_min = INF_TIME;
        int32_t pending_min = INF_TIME;
        bool have_prio = false;
        int last_prio = 0;
        for (int k = 0; k < n; k++) {
            int j = s_prio_order[k];
            int p = s_priority[j];
            if (!have_prio || p != last_prio) {
                running_min = min(running_min, pending_min);
                pending_min = INF_TIME;
                last_prio = p;
                have_prio = true;
            }
            s_rho_min_below[warpInBlock * n + j] = running_min;
            pending_min = min(pending_min, s_rho_max[warpInBlock * n + j]);
        }
    }
    __syncwarp(0xFFFFFFFF);

    int32_t local_rho_any = INF_TIME;
    for (int j = lane; j < n; j += WARP_SIZE) {
        if (s_in_ready[warpInBlock * n + j])
            local_rho_any = min(local_rho_any, s_rho_max[warpInBlock * n + j]);
    }
    int32_t rho_any = v5_k1_warp_min(local_rho_any);

    // Sub-phase 2: per-lane atomicAdd emission. Tried warp-cooperative
    // batched atomicAdd (one atomic per warp via __ballot_sync + __popc
    // ranking) — REGRESSED +0.5% wall on n16_m4_u50/t004. The ballot/shfl
    // overhead + extra register pressure (j/s_min/s_max kept across the
    // emit boundary) outweighed the atomicAdd contention savings on
    // chain-DAG-typical eligibility densities. Don't retry without first
    // changing the surrounding K1 structure (e.g. compaction-based emit).
    #pragma unroll 2
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int j = d_candidates[ci];
        if (!s_in_ready[warpInBlock * n + j]) continue;

        const uint64_t* tc_j = &d_TC[j * W];
        const uint64_t* po_j = &d_PO[j * W];
        bool guard_ok = true;
        for (int w = 0; w < W; w++) {
            if (((tc_j[w] | po_j[w]) & ~st_D[w]) != 0) { guard_ok = false; break; }
        }
        if (!guard_ok) continue;

        // Walk-sharing: rho_min was computed in phase 1 alongside rho_max.
        // Read it from the cache instead of redoing the walk.
        int32_t rho_min_j = s_rho_min[warpInBlock * n + j];
        int32_t rho_hp_j  = s_rho_min_below[warpInBlock * n + j];
        int32_t s_min_j = max(rho_min_j, st_A_pair[0].x);
        // LST formula: nptest space.hpp:804-805 includes `t_wc` (worst-case
        // certain-ready time of any job). V5 omits t_wc; rho_any (= min over
        // ready jobs of rho_max) is structurally <= t_wc, so V5's bound
        // tightens LST. Conservative -- can miss SCHED branches but never
        // flips UNSCHED to SCHED. Verified safe on n12/n16/n20 (memory
        // v5_correctness_validated_n12_n16_n20.md).
        int32_t s_max_j = min(rho_hp_j - 1, max(st_A_pair[0].y, rho_any));

        // K1 deadline pruning: when an ELIGIBLE candidate's best-case finish
        // would exceed its deadline, the dispatch is statically doomed.
        // Critically, this is exactly what K2 would have detected via its
        // f_min > deadline check; if K1 silently `continue`s without
        // signalling unsched, K2 never runs on this candidate and the unsched
        // is LOST. That caused a verdict-flip (n24/m6/u80/ts5: V5 SCHED,
        // nptest UNSCHED). Set d_unschedulable_flag here so the state is
        // correctly classified even when K1 prunes the only eligible
        // candidate. Soundness: matches K2's f_min check semantics; for
        // rtss24/lossy variants the pruning was rare (wider A_pair → smaller
        // s_min); for rtss17/conservative variants it fired on the boundary
        // cases that exposed the missing flag.
        if (d_C_min_dl != nullptr && d_deadline_dl != nullptr) {
            int32_t f_min_check = s_min_j + d_C_min_dl[j];
            if (f_min_check > d_deadline_dl[j]) {
                if (s_min_j <= s_max_j) {
                    atomicExch(d_unschedulable_flag, 1);
                }
                continue;
            }
        }

        if (s_min_j <= s_max_j) {
            int idx = atomicAdd(d_valid_count, 1);
            if (idx < max_valid_pairs) {
                d_valid_pairs[idx].state_idx = stateIdx;
                d_valid_pairs[idx].job_j     = j;
                d_valid_pairs[idx].s_min     = s_min_j;
                d_valid_pairs[idx].s_max     = s_max_j;
            } else {
                atomicExch(d_trunc_flag, 1);
            }
        }
    }
}

} // namespace k1
} // namespace v5
} // namespace sag
