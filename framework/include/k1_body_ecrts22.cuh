// k1_body_ecrts22.cuh -- K1 (eligibility) body for ECRTS 2022 variant.
//
// Paper: G. Nelissen, J. Marce-i-Igual, M. Nasri, "Response-Time Analysis for
//   Non-Preemptive Periodic Moldable Gang Tasks", ECRTS 2022, pp. 12:1-12:22.
// nptest reference: include/global/state_space_data.hpp:222 ready_times();
//   include/global/space.hpp:733-877 dispatch() with the per-parallelism
//   exec_time loop at :769-798.
//
// HARD INVARIANTS:
//   1. Strict layer-by-layer.
//   2. Order does not matter within a layer.
//   3. No truncation cheats.
//   4. Paper math is authoritative.
//
// PHASE 3 MVP (rigid gang):
//   Each job j has parallelism p = c_p_max[j] (== c_p_min[j] for rigid gang).
//   At dispatch the analyzer commits p simultaneously-available cores. The
//   eligibility check is the same as V5's standard K1, except:
//     - EST(j) = max( R^min(j, s),  A_pair[p-1].x )
//     - LST(j) = min( R^max(j, s),  max(A_pair[p-1].y, rho_any),  rho_hp - 1 )
//
//   where A_pair is sorted ascending by .x (V5 invariant). A_pair[p-1] is the
//   p-th smallest core's availability, i.e. the latest of the p
//   earliest-available cores -- exactly when p cores are simultaneously
//   available.
//
//   For non-gang code paths (p == 1) this reduces to V5's standard K1 with
//   A_pair[0]. The body therefore unifies sequential (p==1) and rigid-gang
//   cases with one extra index lookup per candidate.
//
//   Moldable gang (p_min < p_max) is a Phase 3 follow-up: K1 will iterate
//   p in [p_min, p_max] using the greedy-earliest rule from
//   nptest space.hpp:796-798:
//     p chosen = first p for which A_pair[p-1].x >= EST AND
//                A_pair[p].x >  EST  (i.e. higher p does not push EST further)
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include "sag_config.h"
#include "sag_types_v5.h"

namespace sag {
namespace v5 {
namespace k1_ecrts22 {

using namespace sag::config;

__device__ __forceinline__ int v5_k1g_ctz64(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ bool v5_k1g_bit_test(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

__device__ __forceinline__ bool v5_k1g_is_subset(const uint64_t* mask,
                                                 const uint64_t* set, int W) {
    for (int w = 0; w < W; w++) {
        if (mask[w] & ~set[w]) return false;
    }
    return true;
}

__device__ __forceinline__ int32_t v5_k1g_warp_min(int32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = min(val, other);
    }
    return val;
}

template<int WPB>
__device__ __forceinline__ void K1BodyECRTS22(
    char*                        smem_base,
    const char* __restrict__     d_input,
    SAGStateLayoutV5             layout,
    const int* __restrict__      d_candidates,
    int                          num_states,
    int                          num_candidates,
    const uint64_t* __restrict__ d_TC,
    const uint64_t* __restrict__ d_PO,
    const uint64_t* __restrict__ d_Pred,
    const int32_t* __restrict__  d_r_min,
    const int32_t* __restrict__  d_r_max,
    const int32_t* __restrict__  d_priority,
    // d_prio_order removed: only consumed by the priority-bucketed running-min
    // scan that fed s_rho_min_below, both gone now.
    const int32_t* __restrict__  d_sus_min,
    const int32_t* __restrict__  d_sus_max,
    const int32_t* __restrict__  d_p_max,   // Phase 3 (ECRTS 22): per-job parallelism
    const int32_t* __restrict__  d_p_min,   // Phase 3.5: minimum gang width (Eq. 9 lift)
    int n, int m, int W,
    ValidPair* __restrict__      d_valid_pairs,
    int* __restrict__            d_valid_count,
    int* __restrict__            d_unschedulable_flag,
    int                          max_valid_pairs,
    int* __restrict__            d_trunc_flag,
    int                          stateIdx,
    int                          lane,
    int                          warpInBlock,
    int                          tid_in_block,
    // K1 deadline pruning (optional; pass nullptrs to disable). Same as
    // K1BodyV5: skip emission when s_min_j + C_min(j) > deadline(j); K2 is
    // the authoritative gate for unschedulability.
    // d_C_max_dl is consumed by the Phase 3.3.b fallback path (sparse iid
    // with no CSR entries) to write to per-pair side arrays.
    const int32_t* __restrict__  d_C_min_dl   = nullptr,
    const int32_t* __restrict__  d_deadline_dl = nullptr,
    const int32_t* __restrict__  d_C_max_dl   = nullptr,
    // Phase 3.3.b moldable enumeration (optional; pass nullptrs to use
    // single-p rigid mode). When all of d_costmap_offset/p/cmin/cmax AND
    // d_pair_p_out/d_pair_cmin_out/d_pair_cmax_out are non-null, K1 picks
    // p greedy-earliest from the CSR cost map and writes (p, c_min, c_max)
    // into the per-pair side arrays for K2. nptest space.hpp:796-798.
    const int32_t* __restrict__  d_costmap_offset = nullptr,
    const int32_t* __restrict__  d_costmap_p_arr  = nullptr,
    const int32_t* __restrict__  d_costmap_cmin   = nullptr,
    const int32_t* __restrict__  d_costmap_cmax   = nullptr,
    int8_t*  __restrict__        d_pair_p_out     = nullptr,
    int32_t* __restrict__        d_pair_cmin_out  = nullptr,
    int32_t* __restrict__        d_pair_cmax_out  = nullptr)
{
    // Same shmem layout as K1BodyV5 (we do not need extra storage for rigid
    // gang -- only the per-candidate p lookup changes inside the inner loop).
    int32_t* s_rho_max       = (int32_t*)smem_base;
    bool*    s_in_ready      = (bool*)(smem_base + WPB * n * sizeof(int32_t));
    int32_t* s_priority      = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool));
    // s_prio_order removed: was only consumed by the priority-bucketed
    // running-min lane-0 scan, which is now gone (replaced by per-(state, j)
    // gang-aware rho_hp walk). Saves n*sizeof(int32_t) shmem per block.
    // s_rho_min_below removed: gang-aware rho_hp now computed per-(state, j)
    // inline in sub-phase 2 (ECRTS 2022 Eq. 9). Saves WPB*n*sizeof(int32_t)
    // shmem per block.
    int32_t* s_F_min         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t));
    int32_t* s_F_max         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));
    // Walk-sharing (parity with K1BodyV5): cache rho_min computed during
    // phase 1's walk so phase 2 reads it in O(1) instead of redoing the
    // same `preds[c] & F_mask` walk.
    int32_t* s_rho_min       = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));
    // Pre-stage d_p_max + d_p_min into shmem. ECRTS 22 Eq. 9 gang-aware
    // rho_hp walk reads p_h for all n jobs per candidate per state; staging
    // once at block init keeps reads in shmem instead of repeated global
    // hits. Block-shared (not per-warp) -- size 2*n*sizeof(int32_t).
    int32_t* s_p_max         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t));
    int32_t* s_p_min         = (int32_t*)(smem_base + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(bool)
                                          + n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + WPB * n * sizeof(int32_t)
                                          + n * sizeof(int32_t));

    for (int t = tid_in_block; t < n; t += WPB * WARP_SIZE) {
        s_priority[t] = d_priority[t];
        s_p_max[t]    = d_p_max[t];
        s_p_min[t]    = (d_p_min != nullptr) ? d_p_min[t] : d_p_max[t];
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
        if (v5_k1g_bit_test(st_D, c)) continue;
        const uint64_t* preds = &d_Pred[c * W];
        if (!v5_k1g_is_subset(preds, st_D, W)) continue;
        int32_t rho_max_c = d_r_max[c];
        int32_t rho_min_c = d_r_min[c];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds[w] & st_F_mask[w];
            while (pbits) {
                int bit = v5_k1g_ctz64(pbits);
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

    // Phase 3.5 (deferred): Gang-aware rho_hp tried but loosened the
    // constraint enough to cause UNSCHED on the gang.cpp test. The simple
    // "p_i + p_j > m means block" rule misses scenarios where the cumulative
    // commitment of multiple higher-priority gang jobs jointly exceeds m.
    // A correct gang-aware rho_hp needs the full parallelism-stacking
    // analysis from ECRTS 2022 paper, deferred to a future session.
    //
    // The lane-0 priority-bucketed running-min computation that previously
    // populated s_rho_min_below has been removed: sub-phase 2's per-(state, j)
    // walk computes rho_hp_p directly from s_rho_max + s_priority + d_p_min,
    // making the running-min cache unnecessary. Saves an O(n) sequential
    // lane-0 scan per state plus WPB*n int32 shmem writes.

    int32_t local_rho_any = INF_TIME;
    for (int j = lane; j < n; j += WARP_SIZE) {
        if (s_in_ready[warpInBlock * n + j])
            local_rho_any = min(local_rho_any, s_rho_max[warpInBlock * n + j]);
    }
    int32_t rho_any = v5_k1g_warp_min(local_rho_any);

    // Sub-phase 2: eligibility per (state, j) with parallelism-aware
    // A_pair lookup. Phase 3.3.b: when CSR cost map is provided, K1
    // picks p greedy-earliest (the smallest p that minimizes EST_p =
    // max(rho_min_j, A_pair[p-1].x)) and writes (chosen_p, c_min, c_max)
    // to per-pair side arrays for K2. This matches the moldable
    // scheduler-decision semantics from ECRTS 2022 -- the scheduler
    // picks one p per dispatch; multi-emit (one ValidPair per p) would
    // treat p adversarially and over-report UNSCHED.
    bool moldable_active = (d_costmap_offset != nullptr &&
                            d_costmap_p_arr  != nullptr &&
                            d_costmap_cmin   != nullptr &&
                            d_costmap_cmax   != nullptr);

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

        // Greedy-earliest p selection. Multi-emit + reject-all-code-3 also
        // tried; verdict still flipped because V5's K1+K2 reach A_pair states
        // that nptest's tighter abstraction does not (Job 2 hits f_max=21 at
        // layer 4 with s_max=17). Issue is in the per-state interval-envelope
        // computation, not the merge predicate. Multi-day paper-level fix
        // needed; deferred. See v5_ecrts22_multi_emit_per_p_negative.md.
        int p = d_p_max[j];
        if (p < 1) p = 1;
        if (p > m) p = m;
        int32_t chosen_cmin = (d_C_min_dl != nullptr) ? d_C_min_dl[j] : 0;
        int32_t chosen_cmax = (d_C_max_dl != nullptr) ? d_C_max_dl[j] : chosen_cmin;
        if (moldable_active) {
            int off = d_costmap_offset[j];
            int n_p = d_costmap_offset[j + 1] - off;
            if (n_p == 1) {
                int p_cand = d_costmap_p_arr[off];
                if (p_cand >= 1 && p_cand <= m) {
                    p = p_cand;
                    chosen_cmin = d_costmap_cmin[off];
                    chosen_cmax = d_costmap_cmax[off];
                }
            } else {
                int32_t best_EST = INF_TIME;
                int best_p = -1;
                for (int k = 0; k < n_p; k++) {
                    int p_cand = d_costmap_p_arr[off + k];
                    if (p_cand < 1 || p_cand > m) continue;
                    int32_t EST_p = max(rho_min_j, st_A_pair[p_cand - 1].x);
                    if (EST_p < best_EST ||
                        (EST_p == best_EST && p_cand < best_p)) {
                        best_EST = EST_p;
                        best_p = p_cand;
                        chosen_cmin = d_costmap_cmin[off + k];
                        chosen_cmax = d_costmap_cmax[off + k];
                    }
                }
                if (best_p >= 1) p = best_p;
            }
        }
        int p_idx = p - 1;

        // ECRTS 2022 Eq. 9: gang-aware rho_hp at this specific p.
        int32_t prio_j = s_priority[j];
        int32_t rho_hp_p = INF_TIME;
        for (int jh = 0; jh < n; jh++) {
            if (jh == j) continue;
            if (s_priority[jh] >= prio_j) continue;
            int32_t rho_max_h;
            if (s_in_ready[warpInBlock * n + jh]) {
                rho_max_h = s_rho_max[warpInBlock * n + jh];
            } else {
                if (v5_k1g_bit_test(st_D, jh)) continue;
                const uint64_t* preds_h = &d_Pred[jh * W];
                bool has_preds = false;
                for (int w = 0; w < W; w++) {
                    if (preds_h[w]) { has_preds = true; break; }
                }
                if (has_preds) continue;
                rho_max_h = d_r_max[jh];
            }
            // Read pre-staged p_h from block-shared shmem (resolved at K1
            // entry against d_p_min/d_p_max nullptr fallback).
            int p_h = s_p_min[jh];
            if (p_h < 1) p_h = 1;
            if (p_h > m) p_h = m;
            int32_t blocked;
            if (p_h > p) {
                blocked = max(rho_max_h, st_A_pair[p_h - 1].y);
            } else {
                blocked = rho_max_h;
            }
            if (blocked < rho_hp_p) rho_hp_p = blocked;
        }

        int32_t s_min_p = max(rho_min_j, st_A_pair[p_idx].x);
        int32_t s_max_p = min(rho_hp_p - 1,
                              max(st_A_pair[p_idx].y, rho_any));

        // K1 deadline pruning at this p. Sets d_unschedulable_flag if the
        // candidate is eligible but its f_min exceeds deadline -- otherwise
        // the unsched would be lost (K2 doesn't run on pruned candidates).
        // See k1_body_v5.cuh for the lossy-variant verdict-flip diagnosis.
        if (d_deadline_dl != nullptr) {
            if (s_min_p + chosen_cmin > d_deadline_dl[j]) {
                if (s_min_p <= s_max_p) {
                    atomicExch(d_unschedulable_flag, 1);
                }
                continue;
            }
        }

        if (s_min_p <= s_max_p) {
            int idx = atomicAdd(d_valid_count, 1);
            if (idx < max_valid_pairs) {
                d_valid_pairs[idx].state_idx = stateIdx;
                d_valid_pairs[idx].job_j     = j;
                d_valid_pairs[idx].s_min     = s_min_p;
                d_valid_pairs[idx].s_max     = s_max_p;
                if (d_pair_p_out != nullptr) {
                    d_pair_p_out[idx] = (int8_t)p;
                }
                if (d_pair_cmin_out != nullptr &&
                    d_pair_cmax_out != nullptr) {
                    d_pair_cmin_out[idx] = chosen_cmin;
                    d_pair_cmax_out[idx] = chosen_cmax;
                }
            } else {
                atomicExch(d_trunc_flag, 1);
            }
        }
    }
}

} // namespace k1_ecrts22
} // namespace v5
} // namespace sag
