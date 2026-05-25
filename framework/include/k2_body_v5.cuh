// k2_body_v5.cuh -- Inline K2 (successor creation) body for framework,
// using the SPARSE F-pair layout (sag_types_v5.h).
//
// Differences from the dense K2 body:
//   - Reads parent F_min[k] via sparse_F_min lookup over par_F_entries
//   - Writes child F_entries indexed by the new F_avail = F_mask | X
//     (sorted ascending by job_idx)
//   - Per-state F_count is bounded by F_MAX_PER_STATE; on overflow we set
//     a sticky d_trunc_flag, printf the violation, and abort (NEVER drop
//     a valid state silently)
//
// iter410: templated on ENABLE_IJP. With ENABLE_IJP=false the IJP path is
// dead-code-eliminated by NVCC, restoring iter310's register footprint and
// kernel performance. With ENABLE_IJP=true, the leader pair of an IJP set J'
// emits a single multi-job successor with D' = D | mask, and non-leader
// members of J' are skipped (collapsing k! permutations into 1).
//
// Generic across A/H/B-series NV GPUs (no Hopper-only intrinsics).

#pragma once
#include <cstdio>
#include "sag_config.h"
#include "sag_types_v5.h"
#include "ijp_detect_v5.cuh"

namespace sag {
namespace v5 {

using namespace sag::config;

__device__ __forceinline__ int v5_k2_ctz64(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

__device__ __forceinline__ bool v5_k2_bit_test(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

// Single-job STEP 7: in-place insertion of (eft_val, lft_val) into the
// sorted s_A_min_new / s_A_max_new arrays plus Lemma 6 clamp. Used by
// BOTH the (ENABLE_IJP && !ijp_path) arm and the !ENABLE_IJP p_commit==1
// fast path -- both are byte-identical (the legacy invariant per
// k2_body_v5.cuh comments). Extracting eliminates ~50 lines of
// duplicated insertion + clamp code.
//
// Caller must already hold lane==0 context. Uses MAX_CORES bound on m.
__device__ __forceinline__ void v5_step7_insert_single_and_clamp(
    int32_t* s_A_min_new,
    int32_t* s_A_max_new,
    int32_t  est,
    int32_t  eft_val,
    int32_t  lft_val,
    int32_t  p_val,
    int32_t  s_max_val,
    int      m)
{
    int32_t pa_idx = 0, ca_idx = 0;
    bool eft_added = false, lft_added = false;

    for (int i = 1; i < m; i++) {
        int32_t pi = s_A_min_new[i];
        if (!eft_added && eft_val < pi) {
            s_A_min_new[pa_idx++] = eft_val;
            eft_added = true;
        }
        s_A_min_new[pa_idx++] = max(est, pi);
    }
    if (!eft_added) s_A_min_new[pa_idx++] = eft_val;

    for (int i = 1; i < m; i++) {
        int32_t ci = s_A_max_new[i];
        if (!lft_added && lft_val < ci) {
            s_A_max_new[ca_idx++] = lft_val;
            lft_added = true;
        }
        s_A_max_new[ca_idx++] = max(est, ci);
    }
    if (!lft_added) s_A_max_new[ca_idx++] = lft_val;

    // Lemma 6 clamp: clamp at most m-1 slots (dispatched core slot stays).
    int32_t clamp_max = m - 1;
    if (clamp_max < 0) clamp_max = 0;
    int32_t clamp_n = (p_val < m) ? p_val : clamp_max;
    if (clamp_n > clamp_max) clamp_n = clamp_max;
    for (int i = 0; i < clamp_n; i++) {
        if (s_A_max_new[i] > s_max_val) s_A_max_new[i] = s_max_val;
    }
}

// Carry parent F-entries forward into the new state, intersected with
// new F_avail = (s_F_mask_new | s_X_new). Then overwrite/insert (j, f_min, f_max)
// for the dispatched job. The output array is sorted ascending by job_idx.
//
// On overflow (popcount(new_F_avail) > F_MAX_PER_STATE), set d_trunc_flag,
// printf, and abort the warp. NEVER silently drop a valid entry.
//
// Single-job variant (one j_disp). For IJP path see v5_build_child_F_entries_ijp.
__device__ __forceinline__ void v5_build_child_F_entries(
    const FEntryV5* __restrict__ par_entries,
    int                          par_F_count,
    const uint64_t* __restrict__ s_F_mask_new,
    const uint64_t* __restrict__ s_X_new,
    int W,
    int n,
    int j_disp,
    int32_t f_min_disp,
    int32_t f_max_disp,
    FEntryV5*                    out_entries,
    int*                         out_F_count,
    int*                         d_trunc_flag,
    int*                         d_max_F_count_observed,
    int                          layer_dbg,
    int                          state_dbg,
    int                          p_commit_disp = 0)
{
    int out_count = 0;
    int par_pos = 0;
    bool j_emitted = false;

    // Walk new F_avail bits (low to high) and emit entries.
    for (int w = 0; w < W; w++) {
        uint64_t bits = s_F_mask_new[w] | s_X_new[w];
        while (bits) {
            int bit = v5_k2_ctz64(bits);
            bits &= bits - 1;
            int k = w * 64 + bit;
            if (k >= n) continue;

            int32_t fmin = 0, fmax = 0;
            int32_t parent_packed = 0;  // preserve parent's packed (job, p) bits
            // Advance par_pos to find par entry with job_idx == k (or skip if absent).
            while (par_pos < par_F_count && par_entries[par_pos].job() < k) par_pos++;
            bool par_has_k = (par_pos < par_F_count && par_entries[par_pos].job() == k);
            if (par_has_k) {
                fmin = par_entries[par_pos].f_min;
                fmax = par_entries[par_pos].f_max;
                parent_packed = par_entries[par_pos].job_idx;  // packed (job, p)
            }
            // Override if this is the dispatched job.
            if (k == j_disp) {
                fmin = f_min_disp;
                fmax = f_max_disp;
                j_emitted = true;
            }

            if (out_count >= F_MAX_PER_STATE) {
                // OVERFLOW. Loud failure -- never silently truncate.
                printf("[V5 OVERFLOW] F_MAX_PER_STATE=%d exceeded at "
                       "layer=%d state=%d (popcount(F_mask|X) > %d). "
                       "Increase SAG_V5_F_MAX_PER_STATE.\n",
                       F_MAX_PER_STATE, layer_dbg, state_dbg, F_MAX_PER_STATE);
                atomicExch(d_trunc_flag, 1);
                return;
            }
            // Pack p_commit into upper 16 bits for ECRTS 2022's parallelism-
            // aware merge predicate. For dispatched job: pack p_commit_disp.
            // For forwarded parent entry: preserve parent's packed bits.
            // For new (non-parent, non-dispatched) bits in s_X_new: write
            // p=0 (sentinel "unset", read as no-parallelism-info).
            if (k == j_disp) {
                out_entries[out_count].set_job_p(k, p_commit_disp);
            } else if (par_has_k) {
                out_entries[out_count].job_idx = parent_packed;
            } else {
                out_entries[out_count].job_idx = k;  // upper bits 0
            }
            out_entries[out_count].f_min   = fmin;
            out_entries[out_count].f_max   = fmax;
            out_count++;
        }
    }

    // Defensive: j_disp must be in F_avail (j is always in s_X_new). If not,
    // it's a logic bug -- emit an error.
    if (!j_emitted) {
        printf("[V5 BUG] j_disp=%d not in F_avail at layer=%d state=%d\n",
               j_disp, layer_dbg, state_dbg);
        atomicExch(d_trunc_flag, 1);
        return;
    }

    *out_F_count = out_count;
    if (d_max_F_count_observed != nullptr) {
        atomicMax(d_max_F_count_observed, out_count);
    }
}

// IJP-aware variant: jobs in J' (mask) override (f_min, f_max) per-job using
// the per-member arrays job_fmin[k] / job_fmax[k] keyed by member rank.
//
// member_jobs[k] holds the J' members ordered low-to-high in job_idx, so the
// per-job sound bounds are looked up in O(k) when walking F_avail bits.
__device__ __forceinline__ void v5_build_child_F_entries_ijp(
    const FEntryV5* __restrict__ par_entries,
    int                          par_F_count,
    const uint64_t* __restrict__ s_F_mask_new,
    const uint64_t* __restrict__ s_X_new,
    int W,
    int n,
    const int*                   member_jobs,
    const int32_t*               member_fmin,
    const int32_t*               member_fmax,
    int                          k_members,
    FEntryV5*                    out_entries,
    int*                         out_F_count,
    int*                         d_trunc_flag,
    int*                         d_max_F_count_observed,
    int                          layer_dbg,
    int                          state_dbg)
{
    int out_count = 0;
    int par_pos = 0;
    int mem_pos = 0;
    int members_emitted = 0;

    for (int w = 0; w < W; w++) {
        uint64_t bits = s_F_mask_new[w] | s_X_new[w];
        while (bits) {
            int bit = v5_k2_ctz64(bits);
            bits &= bits - 1;
            int k = w * 64 + bit;
            if (k >= n) continue;

            int32_t fmin = 0, fmax = 0;
            while (par_pos < par_F_count && par_entries[par_pos].job() < k) par_pos++;
            if (par_pos < par_F_count && par_entries[par_pos].job() == k) {
                fmin = par_entries[par_pos].f_min;
                fmax = par_entries[par_pos].f_max;
            }
            // member_jobs is sorted ascending; advance mem_pos.
            while (mem_pos < k_members && member_jobs[mem_pos] < k) mem_pos++;
            if (mem_pos < k_members && member_jobs[mem_pos] == k) {
                fmin = member_fmin[mem_pos];
                fmax = member_fmax[mem_pos];
                members_emitted++;
            }

            if (out_count >= F_MAX_PER_STATE) {
                printf("[V5 OVERFLOW IJP] F_MAX_PER_STATE=%d exceeded at "
                       "layer=%d state=%d.\n",
                       F_MAX_PER_STATE, layer_dbg, state_dbg);
                atomicExch(d_trunc_flag, 1);
                return;
            }
            out_entries[out_count].job_idx = k;
            out_entries[out_count].f_min   = fmin;
            out_entries[out_count].f_max   = fmax;
            out_count++;
        }
    }

    if (members_emitted != k_members) {
        printf("[V5 BUG IJP] member emit %d/%d at layer=%d state=%d\n",
               members_emitted, k_members, layer_dbg, state_dbg);
        atomicExch(d_trunc_flag, 1);
        return;
    }

    *out_F_count = out_count;
    if (d_max_F_count_observed != nullptr) {
        atomicMax(d_max_F_count_observed, out_count);
    }
}

template<bool ENABLE_IJP>
__device__ __forceinline__ void K2BodyV5(
    const char* __restrict__      d_input,
    SAGStateLayoutV5              layout,
    const ValidPair* __restrict__ d_valid_pairs,
    int                           num_valid,
    const uint64_t* __restrict__  d_Pred,
    const uint64_t* __restrict__  d_Succ,
    const int32_t* __restrict__   d_C_min,
    const int32_t* __restrict__   d_C_max,
    const int32_t* __restrict__   d_deadline,
    int n, int m, int W,
    char* __restrict__            d_output,
    int* __restrict__             d_output_count,
    int* __restrict__             d_unschedulable_flag,
    int max_output_states,
    int32_t* __restrict__         d_BCRT,
    int32_t* __restrict__         d_WCRT,
    const int32_t* __restrict__   d_r_min,
    int* __restrict__             d_trunc_flag,
    int* __restrict__             d_max_F_count_observed,
    const sag::v5::ijp::IJPInfoV5* __restrict__ d_ijp_info,
    int   globalWarp,
    int   warpInBlock,
    int   lane,
    char* smem_k2_base,
    int   layer_dbg,
    // Phase 3.2 (ECRTS 2022) gang-aware multi-core commit. When non-null,
    // the non-IJP STEP 7 reads d_p_max_arg[j] and commits p cores from the
    // sorted A_pair prefix instead of just slot 0. Pass nullptr (default)
    // for single-core dispatch (rtss24, rtss17, ecrts19); pass d_p_max for
    // ecrts22.
    const int32_t* __restrict__   d_p_max_arg = nullptr,
    // Phase 3.3.b (ECRTS 2022) moldable per-pair p / cost overrides. When
    // non-null, the dispatched-job f_min/f_max use d_pair_cmin[globalWarp]
    // / d_pair_cmax[globalWarp] (looked up per-pair by K1's greedy-earliest
    // choice) instead of the per-job d_C_min[j] / d_C_max[j]. The multi-
    // core commit count uses d_pair_p_arg[globalWarp] (per-pair) in
    // priority over d_p_max_arg[j] (per-job). Pass nullptrs (default) for
    // non-moldable variants.
    const int8_t*  __restrict__   d_pair_p_arg     = nullptr,
    const int32_t* __restrict__   d_pair_cmin_arg  = nullptr,
    const int32_t* __restrict__   d_pair_cmax_arg  = nullptr)
{
    if (globalWarp >= num_valid) return;

    // Per-warp shared-memory slice. Layout below mirrors dense K2:
    //   uint64_t s_D_new     [W]
    //   uint64_t s_X_new     [W]
    //   uint64_t s_F_mask_new[W]
    //   int32_t  s_A_min_new [m]
    //   int32_t  s_A_max_new [m]
    //   int32_t  s_par_F_min [n]   (iter500: dense parent F_min cache)
    // Round up to 8 bytes so each warp's slice starts at a uint64_t-aligned
    // address (s_D_new is uint64_t*; for odd N or m the raw size is only
    // 4-aligned, which causes warp 1's first 8-byte write to misalign on
    // small-N inputs).
    int smem_raw = 3 * W * (int)sizeof(uint64_t)
                 + 2 * m * (int)sizeof(int32_t)
                 + n * (int)sizeof(int32_t);
    const int smem_per_warp = (smem_raw + 7) & ~7;
    char* my_smem = smem_k2_base + warpInBlock * smem_per_warp;

    uint64_t* s_D_new      = reinterpret_cast<uint64_t*>(my_smem);
    uint64_t* s_X_new      = s_D_new + W;
    uint64_t* s_F_mask_new = s_X_new + W;
    int32_t*  s_A_min_new  = reinterpret_cast<int32_t*>(s_F_mask_new + W);
    int32_t*  s_A_max_new  = s_A_min_new + m;
    // iter500: dense parent F_min cache used by K2 STEP 5.  V5's sparse
    // F_entries[] is otherwise looked up via linear scan (sparse_F_min)
    // for each bit in par_X. We scatter par_entries into s_par_F_min[k]
    // ONCE per warp, then STEP 5 reads s_par_F_min[k] in O(1).
    // Lookups outside F_avail return 0, matching sparse_F_min's miss
    // behaviour.  Cost: F_count ALU writes (warp-cooperative); savings:
    // popcount(par_X) * F_count ALU per warp.
    int32_t*  s_par_F_min  = s_A_max_new + m;

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

    // -----------------------------------------------------------------
    // iter410: IJP dispatch.
    //   Case A: d_ijp_info == nullptr OR has_ijp == 0 -> standard single-job path
    //   Case B: has_ijp == 1, j != leader, j in mask -> SKIP (leader handles all)
    //   Case C: has_ijp == 1, j == leader -> emit ONE multi-job successor
    //
    // With ENABLE_IJP=false, all five vars are compile-time 0 / false and
    // every subsequent `if (ijp_path)` branch is dead-code-eliminated by NVCC.
    // -----------------------------------------------------------------
    int      ijp_has    = 0;
    int      ijp_k      = 0;
    int      ijp_leader = 0;
    int32_t  ijp_t_disp = 0;
    int32_t  ijp_est_max_jp = 0;  // iter480 bug #4 fix: max EST over J' members
    uint64_t ijp_mask_w[MAX_BITSET_WORDS];
    #pragma unroll
    for (int w = 0; w < MAX_BITSET_WORDS; w++) ijp_mask_w[w] = 0ULL;

    bool ijp_path = false;
    if constexpr (ENABLE_IJP) {
        if (d_ijp_info != nullptr) {
            if (lane == 0) {
                ijp_has    = d_ijp_info[si].has_ijp;
                ijp_k      = d_ijp_info[si].k;
                ijp_leader = d_ijp_info[si].leader;
                ijp_t_disp = d_ijp_info[si].t_disp;
                ijp_est_max_jp = d_ijp_info[si].est_max_jp;
            }
            ijp_has    = __shfl_sync(0xFFFFFFFF, ijp_has, 0);
            ijp_k      = __shfl_sync(0xFFFFFFFF, ijp_k, 0);
            ijp_leader = __shfl_sync(0xFFFFFFFF, ijp_leader, 0);
            ijp_t_disp = __shfl_sync(0xFFFFFFFF, ijp_t_disp, 0);
            ijp_est_max_jp = __shfl_sync(0xFFFFFFFF, ijp_est_max_jp, 0);
            // Each lane fetches its own word of mask via lane 0; then broadcast.
            if (ijp_has != 0 && lane < W) {
                ijp_mask_w[lane] = d_ijp_info[si].mask[lane];
            }
            // Broadcast each word to all lanes.
            #pragma unroll
            for (int w = 0; w < MAX_BITSET_WORDS; w++) {
                if (w < W) {
                    ijp_mask_w[w] = __shfl_sync(0xFFFFFFFF, ijp_mask_w[w], w);
                }
            }
            // Did we hit Case B? Job j is in mask but j != leader -> skip.
            bool job_in_mask = (ijp_has != 0)
                && ((ijp_mask_w[j / 64] >> (j % 64)) & 1ULL);
            if (job_in_mask && j != ijp_leader) {
                return;  // non-leader pair: leader will handle the J' set
            }
            ijp_path = job_in_mask && (j == ijp_leader);
        }
    }

    const uint64_t* par_D       = layout.D(d_input, si);
    const uint64_t* par_X       = layout.X(d_input, si);
    const uint64_t* par_F_mask  = layout.F_mask(d_input, si);
    const int2*     par_A_pair  = layout.A_pair(d_input, si);
    const FEntryV5* par_entries = layout.F_entries(d_input, si);
    int             par_F_count = *layout.F_count(d_input, si);
    const int32_t*  par_ovf     = layout.ovf(d_input, si);

    // iter500: scatter par_entries into the per-warp dense F_min cache.
    // STEP 5's sparse_F_min(par_entries, par_F_count, k) calls below become
    // O(1) reads from s_par_F_min[k]. Pre-zero so absent entries return 0
    // (matching sparse_F_min's miss behaviour).
    for (int t = lane; t < n; t += WARP_SIZE) s_par_F_min[t] = 0;
    __syncwarp(0xFFFFFFFF);
    for (int e = lane; e < par_F_count; e += WARP_SIZE) {
        FEntryV5 fe = par_entries[e];
        s_par_F_min[fe.job()] = fe.f_min;
    }
    __syncwarp(0xFFFFFFFF);

    // Phase 3.3.b: per-pair cost when moldable enumeration is active.
    int32_t cmin_for_j = (d_pair_cmin_arg != nullptr)
                            ? d_pair_cmin_arg[globalWarp] : d_C_min[j];
    int32_t cmax_for_j = (d_pair_cmax_arg != nullptr)
                            ? d_pair_cmax_arg[globalWarp] : d_C_max[j];
    // p_commit (also computed inside STEP 7 multi-core branch). Hoisted
    // here for the diagnostic printf in the deadline check.
    int p_commit = 1;
    if (d_pair_p_arg != nullptr) {
        p_commit = (int)d_pair_p_arg[globalWarp];
        if (p_commit < 1) p_commit = 1;
        if (p_commit > m) p_commit = m;
    } else if (d_p_max_arg != nullptr) {
        p_commit = d_p_max_arg[j];
        if (p_commit < 1) p_commit = 1;
        if (p_commit > m) p_commit = m;
    }
    int32_t f_min = s_min_val + cmin_for_j;
    int32_t f_max = (ijp_path ? ijp_t_disp : s_max_val) + cmax_for_j;

    if (!ijp_path) {
        if (f_min > d_deadline[j]) {
            if (lane == 0) atomicExch(d_unschedulable_flag, 1);
            return;
        }
    }
    int ovf_witness = (f_max > d_deadline[j]) ? 1 : 0;
    if (!ijp_path && ovf_witness && lane == 0)
        atomicExch(d_unschedulable_flag, 1);

    if (!ijp_path && lane == 0 && d_BCRT != nullptr) {
        atomicMin(&d_BCRT[j], f_min - d_r_min[j]);
        atomicMax(&d_WCRT[j], f_max - d_r_min[j]);
    }

    // STEP 3: D' = D | bit(j) (or D | mask for IJP).
    if (lane < W) {
        uint64_t dw = par_D[lane];
        if constexpr (ENABLE_IJP) {
            if (ijp_path) {
                dw |= ijp_mask_w[lane];
            } else {
                if (lane == (j / 64)) dw |= (1ULL << (j % 64));
            }
        } else {
            if (lane == (j / 64)) dw |= (1ULL << (j % 64));
        }
        s_D_new[lane] = dw;
    }

    // STEP 4: F_mask' init from parent, conditionally add j (or each member of J').
    if (lane < W) {
        s_F_mask_new[lane] = par_F_mask[lane];
    }
    if constexpr (ENABLE_IJP) {
        if (ijp_path) {
            // For each jj in J' (lane 0 walks bits): if Succ[jj] not subset of D',
            // add jj to F_mask.
            if (lane == 0) {
                for (int w = 0; w < W; w++) {
                    uint64_t bits = ijp_mask_w[w];
                    while (bits) {
                        int bit = __ffsll(bits) - 1;
                        bits &= bits - 1;
                        int jj = w * 64 + bit;
                        const uint64_t* succ_jj = &d_Succ[jj * W];
                        bool not_subset = false;
                        for (int ww = 0; ww < W; ww++) {
                            if (succ_jj[ww] & ~s_D_new[ww]) { not_subset = true; break; }
                        }
                        if (not_subset) {
                            s_F_mask_new[jj / 64] |= (1ULL << (jj % 64));
                        }
                    }
                }
            }
        } else {
            const uint64_t* succ_j = &d_Succ[j * W];
            bool local_ok = true;
            if (lane < W) {
                local_ok = ((succ_j[lane] & ~s_D_new[lane]) == 0);
            }
            unsigned ballot = __ballot_sync(0xFFFFFFFF, local_ok || lane >= W);
            if (ballot != 0xFFFFFFFF) {
                if (lane == (j / 64)) {
                    s_F_mask_new[lane] |= (1ULL << (j % 64));
                }
            }
        }
    } else {
        const uint64_t* succ_j = &d_Succ[j * W];
        bool local_ok = true;
        if (lane < W) {
            local_ok = ((succ_j[lane] & ~s_D_new[lane]) == 0);
        }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, local_ok || lane >= W);
        if (ballot != 0xFFFFFFFF) {
            if (lane == (j / 64)) {
                s_F_mask_new[lane] |= (1ULL << (j % 64));
            }
        }
    }
    __syncwarp(0xFFFFFFFF);

    // STEP 4b: parallel GC. All 32 lanes participate; no divergent __shfl_sync.
    // Each lane handles one bit within the current word w; warp-reduces a clear-mask.
    {
        // Hoist s_D_new[0..W-1] into per-lane registers. STEP 3 finalizes
        // s_D_new before the syncwarp at line 482; STEP 4/4b only READ
        // it. The inner subset check (line 513) reads s_D_new[ww] up to
        // W times per assigned lane per outer w iteration; register
        // caching collapses that to W shmem loads at the top.
        // Distinct from `v5_fmask_hoist_negative.md`: that anti-pattern
        // hoisted UNIFORM reads to per-warp shmem (488B/warp); this is
        // hoisting shmem-resident data into registers (W*8 bytes/lane,
        // W<=8). No shmem write; no broadcast pre-prefetch waste.
        uint64_t d_reg[MAX_BITSET_WORDS];
        #pragma unroll
        for (int ww = 0; ww < MAX_BITSET_WORDS; ww++) {
            d_reg[ww] = (ww < W) ? s_D_new[ww] : 0ULL;
        }
        for (int w = 0; w < W; w++) {
            // Uniform broadcast load: all 32 lanes hit the same shmem
            // address; modern CUDA shmem broadcasts in one cycle.
            // Replaces the prior lane-0 conditional load + shfl_sync
            // pair (which was the same broadcast at higher cost).
            uint64_t fcheck = s_F_mask_new[w];

            int popcnt = __popcll(fcheck);
            if (popcnt == 0) continue;

            // Lane l (l < popcnt) finds the l-th set bit in fcheck.
            int my_bit = -1;
            if (lane < popcnt) {
                uint64_t walk = fcheck;
                for (int i = 0; i < lane; i++) {
                    walk &= walk - 1;
                }
                my_bit = __ffsll(walk) - 1;
            }

            // Each assigned lane checks its bit's subset status.
            bool is_subset = false;
            if (my_bit >= 0) {
                int k = w * 64 + my_bit;
                const uint64_t* succ_k = &d_Succ[k * W];
                is_subset = true;
                #pragma unroll
                for (int ww = 0; ww < W; ww++) {
                    if (succ_k[ww] & ~d_reg[ww]) {
                        is_subset = false;
                        break;
                    }
                }
            }

            // Lanes that determine subset contribute their bit to the clear mask.
            uint64_t clear_mask = (is_subset && my_bit >= 0) ? (1ULL << my_bit) : 0ULL;

            // Warp-reduce via __shfl_xor_sync. ALL 32 lanes execute each shfl.
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                uint64_t other = __shfl_xor_sync(0xFFFFFFFF, clear_mask, offset);
                clear_mask |= other;
            }

            // Lane 0 applies the clear; ensure shmem write visible before next iter.
            if (lane == 0 && clear_mask != 0ULL) {
                s_F_mask_new[w] &= ~clear_mask;
            }
            __syncwarp(0xFFFFFFFF);
        }
    }

    // STEP 5: X' = {j} or J' (IJP) plus carry-overs from par_X.
    if constexpr (ENABLE_IJP) {
        if (ijp_path) {
            if (lane < W) s_X_new[lane] = ijp_mask_w[lane];
        } else {
            if (lane < W) s_X_new[lane] = (lane == (j / 64)) ? (1ULL << (j % 64)) : 0ULL;
        }
    } else {
        if (lane < W) s_X_new[lane] = (lane == (j / 64)) ? (1ULL << (j % 64)) : 0ULL;
    }
    __syncwarp(0xFFFFFFFF);

    {
        bool did_ijp_step5 = false;
        if constexpr (ENABLE_IJP) {
            if (ijp_path) {
                // Lane 0 walks par_X bits and tests against union of Pred over J'.
                if (lane == 0) {
                    uint64_t pred_union[MAX_BITSET_WORDS];
                    for (int w = 0; w < MAX_BITSET_WORDS; w++) pred_union[w] = 0ULL;
                    for (int w = 0; w < W; w++) {
                        uint64_t bits = ijp_mask_w[w];
                        while (bits) {
                            int bit = __ffsll(bits) - 1;
                            bits &= bits - 1;
                            int jj = w * 64 + bit;
                            const uint64_t* pj = &d_Pred[jj * W];
                            for (int ww = 0; ww < W; ww++) pred_union[ww] |= pj[ww];
                        }
                    }
                    for (int w = 0; w < W; w++) {
                        uint64_t xbits = par_X[w];
                        while (xbits) {
                            int bit = __ffsll(xbits) - 1;
                            xbits &= xbits - 1;
                            int k = w * 64 + bit;
                            bool in_pred = ((pred_union[k / 64] >> (k % 64)) & 1ULL);
                            if (!in_pred && s_par_F_min[k] > ijp_t_disp) {  // iter500: dense cache
                                s_X_new[k / 64] |= (1ULL << (k % 64));
                            }
                        }
                    }
                }
                did_ijp_step5 = true;
            }
        }
        if (!did_ijp_step5) {
            // standard single-job path
            const uint64_t* preds_j = &d_Pred[j * W];
            if (W == 1) {
                const uint64_t xword = par_X[0];
                bool my_bit_set = false;
                if (lane < n && lane < 32) {
                    int k = lane;
                    if (((xword >> k) & 1ULL)
                        && !v5_k2_bit_test(preds_j, k)
                        && s_par_F_min[k] > s_max_val) {  // iter500: dense cache
                        my_bit_set = true;
                    }
                }
                unsigned ballot1 = __ballot_sync(0xFFFFFFFF, my_bit_set);
                unsigned ballot2 = 0;
                if (n > 32) {
                    bool my_bit_set2 = false;
                    int k2 = lane + 32;
                    if (k2 < n) {
                        if (((xword >> k2) & 1ULL)
                            && !v5_k2_bit_test(preds_j, k2)
                            && s_par_F_min[k2] > s_max_val) {  // iter500: dense cache
                            my_bit_set2 = true;
                        }
                    }
                    ballot2 = __ballot_sync(0xFFFFFFFF, my_bit_set2);
                }
                if (lane == 0) {
                    s_X_new[0] |= ((uint64_t)ballot1) | (((uint64_t)ballot2) << 32);
                }
            } else if (W == 2) {
                // W=2 ballot extension (n in 65..128). Mirrors the W==1
                // idiom: 4 ballots cover 128 bits, each lane handles its
                // own k=lane+32*r position. Replaces the sequential
                // atomicOr-to-shmem loop in the generic W>1 fall-through.
                // Pre-stage par_X words + preds_j words into registers.
                const uint64_t xw0 = par_X[0];
                const uint64_t xw1 = par_X[1];
                const uint64_t pw0 = preds_j[0];
                const uint64_t pw1 = preds_j[1];
                uint64_t out_w0 = 0, out_w1 = 0;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int k = lane + 32 * r;
                    bool bit_set = false;
                    if (k < n) {
                        const uint64_t xw  = (r < 2) ? xw0 : xw1;
                        const uint64_t pw  = (r < 2) ? pw0 : pw1;
                        const int local_b  = (r & 1) ? (k - 64) : (k - 32 * (r >> 1));
                        // Compute local bit position robustly: local_b = k - (k/64)*64.
                        const int b = k - ((k >> 6) << 6);
                        if (((xw >> b) & 1ULL)
                            && !((pw >> b) & 1ULL)
                            && s_par_F_min[k] > s_max_val) {
                            bit_set = true;
                        }
                        (void)local_b;
                    }
                    unsigned bal = __ballot_sync(0xFFFFFFFF, bit_set);
                    if (r == 0) out_w0 |= (uint64_t)bal;
                    else if (r == 1) out_w0 |= ((uint64_t)bal) << 32;
                    else if (r == 2) out_w1 |= (uint64_t)bal;
                    else /* r == 3 */ out_w1 |= ((uint64_t)bal) << 32;
                }
                if (lane == 0) {
                    s_X_new[0] |= out_w0;
                    s_X_new[1] |= out_w1;
                }
            } else {
                int bit_idx = 0;
                for (int w = 0; w < W; w++) {
                    uint64_t xbits = par_X[w];
                    while (xbits) {
                        int bit = __ffsll(xbits) - 1;
                        xbits &= xbits - 1;
                        int k = w * 64 + bit;
                        if (bit_idx % WARP_SIZE == lane) {
                            if (!v5_k2_bit_test(preds_j, k)
                                && s_par_F_min[k] > s_max_val) {  // iter500: dense cache
                                atomicOr(reinterpret_cast<unsigned long long*>(&s_X_new[k / 64]),
                                         1ULL << (k % 64));
                            }
                        }
                        bit_idx++;
                    }
                }
            }
        }
    }

    // STEP 7: insertion-based core availability update.
    for (int i = lane; i < m; i += WARP_SIZE) {
        s_A_min_new[i] = par_A_pair[i].x;
        s_A_max_new[i] = par_A_pair[i].y;
    }
    __syncwarp(0xFFFFFFFF);

    // Lemma 6: p = popcount(par_X & Pred[j])
    int32_t p_val = 0;
    if constexpr (!ENABLE_IJP) {
        const uint64_t* preds_j = &d_Pred[j * W];
        int local_pc = 0;
        if (lane < W) {
            uint64_t pwx = par_X[lane] & preds_j[lane];
            local_pc = __popcll(pwx);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            int other = __shfl_xor_sync(0xFFFFFFFF, local_pc, offset);
            local_pc += other;
        }
        p_val = local_pc;
    } else {
        if (!ijp_path) {
            const uint64_t* preds_j = &d_Pred[j * W];
            int local_pc = 0;
            if (lane < W) {
                uint64_t pwx = par_X[lane] & preds_j[lane];
                local_pc = __popcll(pwx);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                int other = __shfl_xor_sync(0xFFFFFFFF, local_pc, offset);
                local_pc += other;
            }
            p_val = local_pc;
        }
    }

    // p_commit was already hoisted above (line ~389 region) for diagnostic
    // use in the deadline check; reuse it here for v5_build_child_F_entries.

    if (lane == 0) {
        if constexpr (ENABLE_IJP) {
            if (ijp_path) {
                // === IJP STEP 7: multi-job core insertion (sound bound) ===
                // Build per-member f_min/f_max arrays. f_min[jj] uses par_A_pair[0].x
                // as a sound lower bound; f_max[jj] uses the per-member tentative
                // s_max from IJP detection (RTNS '24 Eq. 7) when available, falling
                // back to ijp_t_disp (legacy uniprocessor-pessimistic bound) only if
                // we somehow have no member_s_max plumbed.
                int32_t job_emin[ijp::IJP_V5_MAX_K];
                int32_t job_emax[ijp::IJP_V5_MAX_K];
                int kk = 0;
                int32_t s_min_jj_lb = par_A_pair[0].x;
                for (int w = 0; w < W && kk < ijp::IJP_V5_MAX_K; w++) {
                    uint64_t bits = ijp_mask_w[w];
                    while (bits && kk < ijp::IJP_V5_MAX_K) {
                        int bit = __ffsll(bits) - 1;
                        bits &= bits - 1;
                        int jj = w * 64 + bit;
                        int32_t fm = s_min_jj_lb + d_C_min[jj];
                        // iter430: per-member tighter f_max via tent[t] from
                        // detection. member_s_max[] is indexed by ascending job
                        // index, matching this bit-walk order.
                        int32_t s_max_jj = d_ijp_info[si].member_s_max[kk];
                        int32_t fM = s_max_jj + d_C_max[jj];
                        job_emin[kk] = fm;
                        job_emax[kk] = fM;
                        kk++;
                    }
                }
                // Sort job_emin/job_emax independently.
                for (int i = 1; i < kk; i++) {
                    int32_t x = job_emin[i]; int p = i - 1;
                    while (p >= 0 && job_emin[p] > x) { job_emin[p+1] = job_emin[p]; p--; }
                    job_emin[p+1] = x;
                }
                for (int i = 1; i < kk; i++) {
                    int32_t x = job_emax[i]; int p = i - 1;
                    while (p >= 0 && job_emax[p] > x) { job_emax[p+1] = job_emax[p]; p--; }
                    job_emax[p+1] = x;
                }

                // Snapshot parent A.
                int32_t par_amin_local[MAX_CORES];
                int32_t par_amax_local[MAX_CORES];
                for (int i = 0; i < m; i++) {
                    par_amin_local[i] = s_A_min_new[i];
                    par_amax_local[i] = s_A_max_new[i];
                }

                // iter480 bug #4 fix: paper Algorithm 2 line 13 says
                //   EST = largest earliest start time of jobs in J_B
                // = max EST over J' members. V5 was using ijp_t_disp which is
                // max LST = max member_s_max. LST >= EST so the old value
                // over-bounds (sound but imprecise; can over-tighten parent
                // slots and produce false-SCHED on edge cases). Use
                // est_max_jp = max EST over J' members (computed in detection
                // from s_min_arr -- a sound EST lower bound). Strictly tighter
                // than ijp_t_disp.
                int32_t est_max = ijp_est_max_jp;

                // iter450 paired-merge fix: bump BOTH par_amin AND par_amax
                // by est_max for slots idx_par >= kk. The independent
                // A_min/A_max sorts below preserve per-slot pair invariant
                // (sorted_min[i] <= sorted_max[i]) by the standard "sorted
                // pairs lemma": if every input pair has min <= max, then
                // independent sorts produce per-slot pairings that also
                // satisfy min <= max. Without bumping par_amax along with
                // par_amin, est_max can push the bumped min above the parent's
                // max, breaking the input pair invariant and (when iter430
                // per-member member_s_max produces job_emax < est_max)
                // breaking the output per-slot invariant -- which leads to
                // unsound abstractions and false UNSCHED verdicts (e.g.,
                // n16_m4_u50/t006 layer 46 in iter440 attempts).
                //
                // Bumping par_amax UP by est_max is sound (loosens LFT, never
                // tightens it) and it only fires when par_amax[idx_par] <
                // est_max -- i.e., precisely the case that would otherwise
                // break the invariant.
                int idx_job = 0, idx_par = kk, out_i = 0;
                while (idx_job < kk && idx_par < m) {
                    int32_t emin_p = max(est_max, par_amin_local[idx_par]);
                    if (job_emin[idx_job] <= emin_p) {
                        s_A_min_new[out_i++] = job_emin[idx_job++];
                    } else {
                        s_A_min_new[out_i++] = emin_p; idx_par++;
                    }
                }
                while (idx_job < kk) s_A_min_new[out_i++] = job_emin[idx_job++];
                while (idx_par < m)  { s_A_min_new[out_i++] = max(est_max, par_amin_local[idx_par]); idx_par++; }

                idx_job = 0; idx_par = kk; out_i = 0;
                while (idx_job < kk && idx_par < m) {
                    // iter450: clamp parent's A_max from below by est_max so
                    // the input pair (max(est_max, par_amin), max(est_max, par_amax))
                    // still satisfies min <= max. Equivalent to par_amax when
                    // par_amax >= est_max (the common case), and a sound bump
                    // upward when par_amax < est_max (rare; preserves invariant).
                    int32_t emax_p = max(est_max, par_amax_local[idx_par]);
                    if (job_emax[idx_job] <= emax_p) {
                        s_A_max_new[out_i++] = job_emax[idx_job++];
                    } else {
                        s_A_max_new[out_i++] = emax_p; idx_par++;
                    }
                }
                while (idx_job < kk) s_A_max_new[out_i++] = job_emax[idx_job++];
                while (idx_par < m)  { s_A_max_new[out_i++] = max(est_max, par_amax_local[idx_par]); idx_par++; }
            } else {
                // Single-job standard insertion + Lemma 6 clamp via shared
                // helper (byte-identical to the !ENABLE_IJP p_commit==1
                // fast path; both arms now route through the helper).
                v5_step7_insert_single_and_clamp(
                    s_A_min_new, s_A_max_new,
                    s_min_val, f_min, f_max,
                    p_val, s_max_val, m);
            }
        } else {
            // Multi-core commit support (Phase 3.2 ECRTS 2022). p_commit was
            // hoisted to function scope (above STEP 7 lane==0 block) for the
            // F_entries packing call site. Reuse it here instead of recomputing.
            int32_t est = s_min_val;
            int32_t eft_val = f_min;
            int32_t lft_val = f_max;

            // ECRTS22 perf: p_commit=1 fast path. The multi-core sort path
            // below is byte-identical to V5's standard single-core in-place
            // insertion when p_commit==1 (per the legacy invariant). For
            // rtss24/rtss17/ecrts19 (always p_commit=1) and ecrts22-rigid-p=1
            // (the dominant chain-DAG case), use the simpler in-place path
            // and skip the m-element snapshot. Saves 2m memory ops per K2
            // invocation.
            if (p_commit == 1) {
                // Single-job in-place insertion via shared helper (also
                // used by the IJP-true && !ijp_path arm; both byte-identical).
                v5_step7_insert_single_and_clamp(
                    s_A_min_new, s_A_max_new,
                    est, eft_val, lft_val,
                    p_val, s_max_val, m);
            } else {
            // p_commit >= 2: full multi-core sort path with snapshot.
            int32_t par_amin_snap[MAX_CORES];
            int32_t par_amax_snap[MAX_CORES];
            for (int i = 0; i < m; i++) {
                par_amin_snap[i] = s_A_min_new[i];
                par_amax_snap[i] = s_A_max_new[i];
            }

            // Multi-core sorted insertion for A_min: drop par[0..p_commit-1],
            // insert p_commit copies of eft_val, bump par[p_commit..m-1] up
            // to max(est, par[i]). Output is sorted ascending.
            //
            // For p_commit==1 this reduces to V5's standard single-core
            // insertion (verified byte-output identical).
            {
                int pa_idx = 0;
                int p_rem = p_commit;
                int i = p_commit;
                while (i < m && p_rem > 0) {
                    int32_t pi_bumped = max(est, par_amin_snap[i]);
                    if (eft_val < pi_bumped) {
                        s_A_min_new[pa_idx++] = eft_val;
                        p_rem--;
                    } else {
                        s_A_min_new[pa_idx++] = pi_bumped;
                        i++;
                    }
                }
                while (p_rem > 0) {
                    s_A_min_new[pa_idx++] = eft_val;
                    p_rem--;
                }
                while (i < m) {
                    s_A_min_new[pa_idx++] = max(est, par_amin_snap[i++]);
                }
            }

            {
                int ca_idx = 0;
                int p_rem = p_commit;
                int i = p_commit;
                while (i < m && p_rem > 0) {
                    int32_t ci_bumped = max(est, par_amax_snap[i]);
                    if (lft_val < ci_bumped) {
                        s_A_max_new[ca_idx++] = lft_val;
                        p_rem--;
                    } else {
                        s_A_max_new[ca_idx++] = ci_bumped;
                        i++;
                    }
                }
                while (p_rem > 0) {
                    s_A_max_new[ca_idx++] = lft_val;
                    p_rem--;
                }
                while (i < m) {
                    s_A_max_new[ca_idx++] = max(est, par_amax_snap[i++]);
                }
            }

            // Lemma 6 clamp: cores blocked by j's predecessors in par_X
            // have A_max <= s_max_val. The dispatched-job block occupies
            // p_commit slots in the sorted output; we must never clamp
            // those (the EFT/LFT pair belongs to j's actual finish, not
            // a phantom predecessor block).
            //
            // For chain-DAG with eft >= max(par) (the common case), the
            // dispatched block is at the END of the sorted output, so
            // clamping the FIRST min(p_val, m - p_commit) slots is safe.
            // For p_commit==1 this matches the previous m-1 logic;
            // for gang p_commit>=2 (Phase 3.2 ECRTS 2022) it correctly
            // leaves the gang block unclamped.
            int32_t clamp_max_np = m - p_commit;
            if (clamp_max_np < 0) clamp_max_np = 0;
            int32_t clamp_n = (p_val < m) ? p_val : clamp_max_np;
            if (clamp_n > clamp_max_np) clamp_n = clamp_max_np;
            for (int i = 0; i < clamp_n; i++) {
                if (s_A_max_new[i] > s_max_val) s_A_max_new[i] = s_max_val;
            }
            }  // end p_commit >= 2 branch
        }
    }
    __syncwarp(0xFFFFFFFF);

    // STEP 9: Reserve output slot.
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
    int2*     out_A_pair = layout.A_pair(d_output, out_idx);
    FEntryV5* out_entries = layout.F_entries(d_output, out_idx);
    int*      out_F_count_p = layout.F_count(d_output, out_idx);
    int32_t*  out_ovf    = layout.ovf(d_output, out_idx);

    if (lane < W) {
        out_D[lane]      = s_D_new[lane];
        out_X[lane]      = s_X_new[lane];
        out_F_mask[lane] = s_F_mask_new[lane];
    }
    for (int i = lane; i < m; i += WARP_SIZE) {
        out_A_pair[i] = make_int2(s_A_min_new[i], s_A_max_new[i]);
    }

    if (lane == 0) {
        if constexpr (ENABLE_IJP) {
            if (ijp_path) {
                // Build per-member f_min/f_max arrays in ascending member order.
                // iter430: f_max[jj] uses the per-member tentative s_max from
                // IJP detection (RTNS '24 Eq. 7), tighter than ijp_t_disp =
                // max_{j' in J'} s_max(j') which over-approximates non-worst-slot
                // members.
                int     member_jobs[ijp::IJP_V5_MAX_K];
                int32_t member_fmin[ijp::IJP_V5_MAX_K];
                int32_t member_fmax[ijp::IJP_V5_MAX_K];
                int kk = 0;
                int32_t s_min_jj_lb = par_A_pair[0].x;
                int any_ovf = 0;
                for (int w = 0; w < W && kk < ijp::IJP_V5_MAX_K; w++) {
                    uint64_t bits = ijp_mask_w[w];
                    while (bits && kk < ijp::IJP_V5_MAX_K) {
                        int bit = __ffsll(bits) - 1;
                        bits &= bits - 1;
                        int jj = w * 64 + bit;
                        int32_t fm = s_min_jj_lb + d_C_min[jj];
                        int32_t s_max_jj = d_ijp_info[si].member_s_max[kk];
                        int32_t fM = s_max_jj + d_C_max[jj];
                        member_jobs[kk] = jj;
                        member_fmin[kk] = fm;
                        member_fmax[kk] = fM;
                        if (fM > d_deadline[jj]) any_ovf = 1;
                        if (fm > d_deadline[jj]) atomicExch(d_unschedulable_flag, 1);
                        if (d_BCRT != nullptr) {
                            atomicMin(&d_BCRT[jj], fm - d_r_min[jj]);
                            atomicMax(&d_WCRT[jj], fM - d_r_min[jj]);
                        }
                        kk++;
                    }
                }
                v5_build_child_F_entries_ijp(
                    par_entries, par_F_count,
                    s_F_mask_new, s_X_new, W, n,
                    member_jobs, member_fmin, member_fmax, kk,
                    out_entries, out_F_count_p,
                    d_trunc_flag,
                    d_max_F_count_observed,
                    layer_dbg, out_idx);
                if (any_ovf) atomicExch(d_unschedulable_flag, 1);
                *out_ovf = *par_ovf | any_ovf;
            } else {
                v5_build_child_F_entries(
                    par_entries, par_F_count,
                    s_F_mask_new, s_X_new, W, n,
                    j, f_min, f_max,
                    out_entries, out_F_count_p,
                    d_trunc_flag,
                    d_max_F_count_observed,
                    layer_dbg, out_idx,
                    p_commit);  // ECRTS 2022 parallelism-aware merge plumbing
                *out_ovf = *par_ovf | ovf_witness;
            }
        } else {
            v5_build_child_F_entries(
                par_entries, par_F_count,
                s_F_mask_new, s_X_new, W, n,
                j, f_min, f_max,
                out_entries, out_F_count_p,
                d_trunc_flag,
                d_max_F_count_observed,
                layer_dbg, out_idx,
                p_commit);  // ECRTS 2022 parallelism-aware merge plumbing
            *out_ovf = *par_ovf | ovf_witness;
        }
    }
}

} // namespace v5
} // namespace sag
