// ijp_detect_v5.cuh -- Independent-Job Parallelism (IJP) detection for V5.
//
// Mirrors the legacy IJPDetectKernel from src/successor_creation.cu:93 but:
//   * Reads from the V5 sparse F-pair encoding (sag_types_v5.h)
//   * Supports W >= 1 via a multi-word mask (legacy was W==1 only)
//   * Runs as a __device__ body callable from inside V5K1K2Kernel after K1
//
// Per-state output: IJPInfoV5{has_ijp, k, leader, t_disp, mask[MAX_BITSET_WORDS]}.
// When has_ijp == 1 (k >= 2 and the J' set passed all soundness gates), the
// leader pair (job_j == leader) emits a SINGLE multi-job successor with
// D' = D | mask, and the other (k-1) pairs are skipped. This collapses k!
// permutations of the same J' set into 1 child state.
//
// Per iter180 in legacy: on UNSCHED-determining workloads with high core
// utilization, this gives 80%+ state-count reduction.
//
// Soundness conditions (Algorithm 4-style greedy interferer absorption):
//   For each candidate J' set member jj:
//     1. f_max[jj] <= deadline[jj]   (each member must be schedulable)
//     2. s_min(jj) <= s_max(jj; J')  (J' member feasibility under worst-slot bound)
//     3. No "sneak" higher-priority job h not in J' with rho_max[h] <= t_disp
//        (else h would displace a J' member under JLFP)
//     4. parent A_max[k-1] <= new_t_disp (slot k-1 must accept the newest member)
//
// Generic across A/H/B-series NV GPUs (no Hopper-only intrinsics).

#pragma once
#include <cstdio>
#include "sag_config.h"
#include "sag_types_v5.h"
#include "const_job_arrays_v5.cuh"  // c_C_max/c_deadline/c_r_min/c_priority

namespace sag {
namespace v5 {
namespace ijp {

using namespace sag::config;

// Compile-time bounds. IJP_V5_MAX_K bounds the IJP set size; legacy used 16.
// Bounded by m at runtime, so for m <= 20 a 20-element fixed-size array
// dominates the per-warp register footprint.
constexpr int IJP_V5_INF_TIME = INT32_MAX / 2;
constexpr int IJP_V5_MAX_K    = 16;
// Max n we support per layer for IJP. Higher n forces a sparse fallback;
// IJP detection is O(n^2) in lane-0 sequential code so very large n is
// already uneconomic. Match legacy IJP_MAX_N.
constexpr int IJP_V5_MAX_N    = 256;

// Per-state IJP detection result. Multi-word mask supports W > 1 (legacy
// IJP was uint64_t-only / N <= 64). For N > MAX_BITSET_WORDS*64 the
// detection falls through with has_ijp=0.
//
// iter430: member_jobs[] / member_s_max[] propagate the per-member
// tentative s_max from detection (the `tent[t]` array, RTNS '24 Eq. 7
// per-member LFT) into K2 so f_max[jj] can use member_s_max[jj] + C_max[jj]
// instead of the looser t_disp + C_max[jj] (max across J'). Indexed by
// job-index ascending order matching the bit-walk in K2's emission loop.
struct IJPInfoV5 {
    int32_t  has_ijp;          // 0 or 1
    int32_t  k;                // |J'|
    int32_t  leader;           // smallest job_idx in J'
    int32_t  t_disp;           // worst-slot dispatch time t_disp = max_{jj in J'} s_max(jj)
    uint64_t mask[MAX_BITSET_WORDS];
    // iter430: per-member tentative s_max (final accepted `tent[]` from detection),
    // ordered by ascending job index (matching the bit-walk in K2 emit). Slot t
    // corresponds to the t-th set bit (low-to-high) of mask.
    int32_t  member_s_max[IJP_V5_MAX_K];
    // iter480 bug #4 fix: per-member s_min (paper EST(j', v) lower bound),
    // ordered by ascending job index (parallel to member_s_max). Used by K2
    // STEP 7 as `est_max = max over members of member_s_min` per Algorithm 2
    // line 13 (max EST not max LST).
    int32_t  member_s_min[IJP_V5_MAX_K];
    // iter480 bug #4 fix: max over J' members of member_s_min (= max EST over
    // J', the paper's Algorithm 2 line 13 EST). Replaces ijp_t_disp (= max LST)
    // for the K2 STEP 7 parent-slot bump. Sound and tighter than max LST.
    int32_t  est_max_jp;
};

__device__ __forceinline__ int v5_ijp_ctz64(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ bool v5_ijp_bit_test(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

__device__ __forceinline__ bool v5_ijp_is_subset(const uint64_t* a,
                                                  const uint64_t* b, int W) {
    for (int w = 0; w < W; w++) {
        if (a[w] & ~b[w]) return false;
    }
    return true;
}

// IJP detect body. Single warp per state; lane 0 does the sequential scan,
// other lanes idle. (Per legacy IJP: parallelizing the n^2 inner loops adds
// register pressure with little gain at typical n <= 80.)
//
// out: d_ijp_info[state_idx] = {has_ijp, k, leader, t_disp, mask[W]}.
//
// d_ijp_info is sized per WAVE INPUT count (num_states_in). Caller must
// initialize d_ijp_info[stateIdx].has_ijp = 0 before calling (or rely on
// the explicit reset at the top of this body).
__device__ __forceinline__ void IJPDetectBodyV5(
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
    const int32_t* __restrict__  d_C_max,
    const int32_t* __restrict__  d_deadline,
    const int32_t* __restrict__  d_sus_min,
    const int32_t* __restrict__  d_sus_max,
    int                          n,
    int                          m,
    int                          W,
    IJPInfoV5* __restrict__      d_ijp_info,
    int                          stateIdx,
    int                          lane)
{
    if (stateIdx < 0 || stateIdx >= num_states) return;

    // Default: no IJP for this state. Lane 0 writes; other lanes return.
    if (lane == 0) {
        d_ijp_info[stateIdx].has_ijp = 0;
        d_ijp_info[stateIdx].k       = 0;
        d_ijp_info[stateIdx].leader  = 0;
        d_ijp_info[stateIdx].t_disp  = 0;
        d_ijp_info[stateIdx].est_max_jp = 0;
        for (int w = 0; w < W; w++) d_ijp_info[stateIdx].mask[w] = 0ULL;
    }
    if (lane != 0) return;

    // Bounds: m >= 2 (need >= 2 cores for IJP), n <= IJP_V5_MAX_N.
    if (m < 2 || n > IJP_V5_MAX_N) return;
    if (W > MAX_BITSET_WORDS) return;

    const uint64_t* st_D       = layout.D(d_input, stateIdx);
    const uint64_t* st_F_mask  = layout.F_mask(d_input, stateIdx);
    const int2*     st_A_pair  = layout.A_pair(d_input, stateIdx);
    const FEntryV5* st_F_entries = layout.F_entries(d_input, stateIdx);
    int             st_F_count   = *layout.F_count(d_input, stateIdx);

    // Per-job state arrays. Storage allocated in registers / local memory.
    // For typical n <= 100 this is comfortable.
    // iter480 bug #2 fix: also keep rho_min_arr[] for tighter per-slot EST
    // computation in the no-sneak gate (paper Eq. 12).
    int32_t rho_max_arr  [IJP_V5_MAX_N];
    int32_t rho_min_arr  [IJP_V5_MAX_N];
    bool    in_ready_arr [IJP_V5_MAX_N];
    int32_t s_min_arr    [IJP_V5_MAX_N];
    bool    is_eligible  [IJP_V5_MAX_N];

    for (int j = 0; j < n; j++) {
        rho_max_arr [j] = IJP_V5_INF_TIME;
        rho_min_arr [j] = IJP_V5_INF_TIME;
        in_ready_arr[j] = false;
        s_min_arr   [j] = IJP_V5_INF_TIME;
        is_eligible [j] = false;
    }

    // Sub-Phase 1: rho_max for each ready candidate.
    int32_t rho_any = IJP_V5_INF_TIME;
    for (int ci = 0; ci < num_candidates; ci++) {
        int c = d_candidates[ci];
        if (c >= n) continue;
        if (v5_ijp_bit_test(st_D, c)) continue;
        const uint64_t* preds = &d_Pred[c * W];
        if (!v5_ijp_is_subset(preds, st_D, W)) continue;

        int32_t rho_max_c = d_r_max[c];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds[w] & st_F_mask[w];
            while (pbits) {
                int bit = v5_ijp_ctz64(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                int32_t fmax_i = sparse_F_max(st_F_entries, st_F_count, i);
                int32_t v = fmax_i + d_sus_max[c * n + i];
                if (v > rho_max_c) rho_max_c = v;
            }
        }
        rho_max_arr[c]  = rho_max_c;
        in_ready_arr[c] = true;
        if (rho_max_c < rho_any) rho_any = rho_max_c;
    }

    // Sub-Phase 2: per-job rho_min, eligibility, single-job s_max bound.
    for (int ci = 0; ci < num_candidates; ci++) {
        int j = d_candidates[ci];
        if (j >= n) continue;
        if (!in_ready_arr[j]) continue;
        if (v5_ijp_bit_test(st_D, j)) continue;

        const uint64_t* tc_j = &d_TC[j * W];
        const uint64_t* po_j = &d_PO[j * W];
        bool guard = true;
        for (int w = 0; w < W; w++) {
            if (((tc_j[w] | po_j[w]) & ~st_D[w]) != 0) { guard = false; break; }
        }
        if (!guard) continue;

        int32_t rho_min_j = c_r_min[j];  // iter530: cmem
        const uint64_t* preds_j = &d_Pred[j * W];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds_j[w] & st_F_mask[w];
            while (pbits) {
                int bit = v5_ijp_ctz64(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                int32_t fmin_i = sparse_F_min(st_F_entries, st_F_count, i);
                int32_t v = fmin_i + d_sus_min[j * n + i];
                if (v > rho_min_j) rho_min_j = v;
            }
        }

        int32_t prio_j = c_priority[j];  // iter530: cmem
        int32_t hp = IJP_V5_INF_TIME;
        for (int h = 0; h < n; h++) {
            if (!in_ready_arr[h] || h == j) continue;
            if (c_priority[h] < prio_j) {  // iter530: cmem
                if (rho_max_arr[h] < hp) hp = rho_max_arr[h];
            }
        }
        int32_t sm_min = max(rho_min_j, (int32_t)st_A_pair[0].x);
        int32_t hpm1   = (hp >= IJP_V5_INF_TIME) ? IJP_V5_INF_TIME : (hp - 1);
        int32_t sm_max = min(hpm1, max((int32_t)st_A_pair[0].y, rho_any));
        if (sm_min > sm_max) continue;
        s_min_arr  [j] = sm_min;
        rho_min_arr[j] = rho_min_j;
        is_eligible[j] = true;
    }

    // Sort eligible jobs by priority asc (lower prio number = higher pri),
    // tie-breaking by job index asc.
    int rank[IJP_V5_MAX_N];
    int rank_cnt = 0;
    for (int j = 0; j < n; j++) {
        if (is_eligible[j]) rank[rank_cnt++] = j;
    }
    // Insertion sort.
    for (int i = 1; i < rank_cnt; i++) {
        int x = rank[i]; int xp = c_priority[x];  // iter530: cmem
        int p = i - 1;
        while (p >= 0) {
            int yp = c_priority[rank[p]];  // iter530: cmem
            if (yp < xp || (yp == xp && rank[p] < x)) break;
            rank[p + 1] = rank[p]; p--;
        }
        rank[p + 1] = x;
    }

    // Greedy J' construction. set_arr holds the current J' (sorted by
    // priority by construction since we walk rank in priority order).
    int     set_arr[IJP_V5_MAX_K];
    int     set_cnt = 0;
    int32_t cur_t_disp = -1;

    // iter430: final accepted per-member tentative s_max, indexed by
    // acceptance order (parallel to set_arr[]). Updated each time we
    // accept a new member (which can re-tighten existing members' s_max
    // because new_k grows the worst-slot index `a_slot`). Captured here so
    // K2 can use the per-member LFT instead of t_disp = max(tent[]).
    int32_t final_tent[IJP_V5_MAX_K];

    int set_cap = (m < IJP_V5_MAX_K) ? m : IJP_V5_MAX_K;
    for (int ri = 0; ri < rank_cnt && set_cnt < set_cap; ri++) {
        int rj = rank[ri];

        // Pairwise non-conflict: rj must not be in TC(ex) and ex must not
        // be in TC(rj) for any ex in set_arr.
        bool conflict = false;
        const uint64_t* tc_rj = &d_TC[rj * W];
        for (int t = 0; t < set_cnt; t++) {
            int ex = set_arr[t];
            if ((tc_rj[ex / 64] >> (ex % 64)) & 1ULL) { conflict = true; break; }
            const uint64_t* tc_ex = &d_TC[ex * W];
            if ((tc_ex[rj / 64] >> (rj % 64)) & 1ULL) { conflict = true; break; }
        }
        if (conflict) continue;

        int new_k = set_cnt + 1;
        int worst_slot = new_k - 1;
        int32_t a_slot = (worst_slot < m) ? (int32_t)st_A_pair[worst_slot].y
                                          : IJP_V5_INF_TIME;

        // Per-member tentative s_max. We compute TWO bounds:
        //   - tent[t]      : "could-be-at-worst-slot" bound, using a_slot
        //                    (= A_pair[k-1].y). Used for the sound feasibility
        //                    and sched-deadline checks (any member could end up
        //                    at the worst slot under FIFO-tied dispatch).
        //   - slot_tent[t] : "JLFP-slot-aware" bound, using A_pair[t].y where
        //                    t is the member's priority rank in J'. This is
        //                    strictly tighter for non-worst-slot members and
        //                    matches RTNS '24 / Eq. 7's per-member LFT used in
        //                    the K2 emission for f_max[jj] (iter430).
        // rho_hp_ijp uses the others J' \ {jj} excluded from the hp-ready
        // set so they don't artificially block jj.
        int32_t tent[IJP_V5_MAX_K];
        int32_t slot_tent[IJP_V5_MAX_K];
        bool feas = true;
        // Soundness fix (paper Eq. 11): t_high(j, v) = cert_ready_high(j, v, ψ(v))
        // where ψ(v) is the J' size after this acceptance. The previous single
        // MIN over higher-priority outsiders' rho_max was cert_ready_high(...,1)
        // (the FIRST hp-outsider's certain-ready time), making sm artificially
        // tight. Replace with the ψ-th smallest rho_max via a small
        // max-heap-of-size-ψ. ψ <= IJP_V5_MAX_K so heap fits in registers.
        const int psi = new_k;  // ψ(v) at this acceptance moment
        for (int t = 0; t < new_k; t++) {
            int jj = (t < set_cnt) ? set_arr[t] : rj;
            int32_t a_slot_t = (t < m) ? (int32_t)st_A_pair[t].y
                                       : IJP_V5_INF_TIME;

            int32_t prio_jj = c_priority[jj];  // iter530: cmem
            int32_t any_ijp = IJP_V5_INF_TIME;
            // Max-heap of size ψ holding the ψ smallest hp rho_max values.
            // After the scan its root (hp_top[0]) IS the ψ-th smallest.
            int32_t hp_top[IJP_V5_MAX_K + 1];
            int     hp_count = 0;
            for (int h = 0; h < n; h++) {
                if (!in_ready_arr[h] || h == jj) continue;
                bool in_others = false;
                for (int t2 = 0; t2 < set_cnt; t2++) {
                    if (set_arr[t2] == h) { in_others = true; break; }
                }
                if (!in_others && h == rj) in_others = true;
                if (in_others) continue;
                if (rho_max_arr[h] < any_ijp) any_ijp = rho_max_arr[h];
                if (c_priority[h] < prio_jj) {  // iter530: cmem
                    int32_t v = rho_max_arr[h];
                    if (hp_count < psi) {
                        // Insert v; sift-up to restore max-heap property.
                        int idx = hp_count++;
                        hp_top[idx] = v;
                        while (idx > 0) {
                            int p = (idx - 1) >> 1;
                            if (hp_top[p] < hp_top[idx]) {
                                int32_t tmp = hp_top[p];
                                hp_top[p] = hp_top[idx];
                                hp_top[idx] = tmp;
                                idx = p;
                            } else break;
                        }
                    } else if (v < hp_top[0]) {
                        // Replace max with v; sift-down to restore.
                        hp_top[0] = v;
                        int idx = 0;
                        while (true) {
                            int l = 2 * idx + 1;
                            int r = 2 * idx + 2;
                            int largest = idx;
                            if (l < psi && hp_top[l] > hp_top[largest]) largest = l;
                            if (r < psi && hp_top[r] > hp_top[largest]) largest = r;
                            if (largest == idx) break;
                            int32_t tmp = hp_top[idx];
                            hp_top[idx] = hp_top[largest];
                            hp_top[largest] = tmp;
                            idx = largest;
                        }
                    }
                }
            }
            // If fewer than ψ HP outsiders exist, there is no ψ-th HP-ready
            // event ⇒ no upper-bound displacement deadline ⇒ INF.
            int32_t hp_ijp = (hp_count < psi) ? IJP_V5_INF_TIME : hp_top[0];
            int32_t hpm1 = (hp_ijp >= IJP_V5_INF_TIME) ? IJP_V5_INF_TIME : (hp_ijp - 1);
            int32_t sm        = min(hpm1, max(a_slot,    any_ijp));
            int32_t sm_slot_t = min(hpm1, max(a_slot_t,  any_ijp));
            if (s_min_arr[jj] > sm) { feas = false; break; }
            tent[t]      = sm;
            slot_tent[t] = sm_slot_t;
        }
        if (!feas) continue;

        int32_t new_t = -IJP_V5_INF_TIME;
        for (int t = 0; t < new_k; t++) {
            if (tent[t] > new_t) new_t = tent[t];
        }

        // Slot k-1 must be available by new_t (else the kth dispatch can't
        // happen at the worst-slot bound).
        if ((int32_t)st_A_pair[new_k - 1].y > new_t) continue;

        // iter480 bug #2 fix: paper Eq. 12 no-sneak condition.
        //
        // Paper: J' is admitted iff
        //   forall h in E(v) \ J',  exists j' in J',  R_max(h, v) < EST(j', v)
        //
        // Previously V5 used a priority-filter form
        //   (rho_max[h] <= new_t && d_priority[h] < prio_max_in_set) -> reject
        // which (a) added a priority filter the paper does NOT have, and
        // (b) compared against new_t = max LST instead of per-member EST,
        // both of which made the gate LOOSER (admitted more J' sets that
        // shouldn't qualify, leading to false-SCHED on cases like t005).
        //
        // EST(j', v) per paper Eq. 5 = max(A_max_t(v), pot_ready(t, v)) for
        // the t-th J' member at slot t. We use the sound LOWER bound
        //     EST_LB(j' at slot t, v) = max(rho_min[j'], A_pair[t].y)
        // since EST_t(v) >= A_max_t(v) = A_pair[t].y by Eq. 5, and j' cannot
        // start before its own minimum certain-ready time rho_min[j'].
        // (Using A_pair[t].y at the slot rather than A_pair[0].x as in
        // s_min_arr is strictly tighter and avoids over-rejection.)
        // For 'rj' (the candidate being added), its slot is set_cnt.
        bool sneak_ok = true;
        for (int h = 0; h < n; h++) {
            if (!in_ready_arr[h]) continue;
            if (h == rj) continue;
            bool in_set = false;
            for (int t = 0; t < set_cnt; t++) {
                if (set_arr[t] == h) { in_set = true; break; }
            }
            if (in_set) continue;
            // Per Eq. 12: exists j' in J' s.t. R_max(h) < EST(j').
            // Equivalent: J' invalid iff R_max(h) >= EST(j') for ALL j' in J'.
            int32_t rmax_h = rho_max_arr[h];
            bool found_jprime = false;
            for (int t = 0; t < new_k; t++) {
                int jprime = (t < set_cnt) ? set_arr[t] : rj;
                int32_t a_max_t = (t < m) ? (int32_t)st_A_pair[t].y
                                          : IJP_V5_INF_TIME;
                int32_t est_jprime = max(rho_min_arr[jprime], a_max_t);
                if (rmax_h < est_jprime) { found_jprime = true; break; }
            }
            if (!found_jprime) { sneak_ok = false; break; }
        }
        if (!sneak_ok) continue;

        // Per-member deadline check at the worst-case f_max under IJP.
        bool sched_ok = true;
        for (int t = 0; t < new_k; t++) {
            int jj = (t < set_cnt) ? set_arr[t] : rj;
            int32_t fm = tent[t] + c_C_max[jj];  // iter530: cmem
            if (fm > c_deadline[jj]) { sched_ok = false; break; }  // iter530: cmem
        }
        if (!sched_ok) continue;

        // Accept rj.
        set_arr[set_cnt++] = rj;
        cur_t_disp = new_t;
        // iter430: snapshot the per-member JLFP-slot-aware s_max for the
        // newly-accepted set. Indexed by acceptance order (= priority rank,
        // same as set_arr[]). Uses slot_tent (with A_pair[t].y) which is
        // strictly tighter than tent (with A_pair[k-1].y) for non-worst-slot
        // members. tent is used for the worst-slot feasibility/sched checks.
        for (int t = 0; t < set_cnt; t++) {
            final_tent[t] = slot_tent[t];
        }
    }

    if (set_cnt >= 2) {
        // Build mask + leader.
        uint64_t mask[MAX_BITSET_WORDS];
        for (int w = 0; w < MAX_BITSET_WORDS; w++) mask[w] = 0ULL;
        int leader = INT_MAX;
        for (int t = 0; t < set_cnt; t++) {
            int jj = set_arr[t];
            mask[jj / 64] |= (1ULL << (jj % 64));
            if (jj < leader) leader = jj;
        }
        d_ijp_info[stateIdx].has_ijp = 1;
        d_ijp_info[stateIdx].k       = set_cnt;
        d_ijp_info[stateIdx].leader  = leader;
        d_ijp_info[stateIdx].t_disp  = cur_t_disp;
        for (int w = 0; w < W; w++) {
            d_ijp_info[stateIdx].mask[w] = mask[w];
        }

        // iter430: write member_s_max[] indexed by ascending job index so K2's
        // bit-walk emit (low-to-high) reads tents in matching order. Insertion
        // sort the (job_idx, tent) pairs from acceptance order to job order.
        // iter480 bug #4 fix: also write member_s_min[] (per-member slot-aware
        // EST lower bound) and est_max_jp (= max EST over J' members), used
        // by K2 STEP 7 instead of t_disp (= max LST) per Algorithm 2 line 13.
        // Slot-aware EST_LB at slot t = max(rho_min[set_arr[t]], A_pair[t].y),
        // strictly tighter than s_min_arr[j'] which uses A_pair[0].x.
        int     ord_jobs[IJP_V5_MAX_K];
        int32_t ord_smax[IJP_V5_MAX_K];
        int32_t ord_smin[IJP_V5_MAX_K];
        for (int t = 0; t < set_cnt; t++) {
            ord_jobs[t] = set_arr[t];
            ord_smax[t] = final_tent[t];
            int32_t a_max_t = (t < m) ? (int32_t)st_A_pair[t].y
                                      : IJP_V5_INF_TIME;
            ord_smin[t] = max(rho_min_arr[set_arr[t]], a_max_t);
        }
        for (int i = 1; i < set_cnt; i++) {
            int xj = ord_jobs[i]; int32_t xs = ord_smax[i]; int32_t xn = ord_smin[i];
            int p = i - 1;
            while (p >= 0 && ord_jobs[p] > xj) {
                ord_jobs[p+1] = ord_jobs[p];
                ord_smax[p+1] = ord_smax[p];
                ord_smin[p+1] = ord_smin[p];
                p--;
            }
            ord_jobs[p+1] = xj; ord_smax[p+1] = xs; ord_smin[p+1] = xn;
        }
        int32_t est_max_jp = -IJP_V5_INF_TIME;
        for (int t = 0; t < set_cnt; t++) {
            d_ijp_info[stateIdx].member_s_max[t] = ord_smax[t];
            d_ijp_info[stateIdx].member_s_min[t] = ord_smin[t];
            if (ord_smin[t] > est_max_jp) est_max_jp = ord_smin[t];
        }
        d_ijp_info[stateIdx].est_max_jp = est_max_jp;
        // Zero unused slots (deterministic device memory).
        for (int t = set_cnt; t < IJP_V5_MAX_K; t++) {
            d_ijp_info[stateIdx].member_s_max[t] = 0;
            d_ijp_info[stateIdx].member_s_min[t] = 0;
        }
    }
}

} // namespace ijp
} // namespace v5
} // namespace sag
