// merge_body_v5.cuh -- Device helpers for the framework merge,
// using the SPARSE F-pair layout (sag_types_v5.h).
//
// Mirrors src/merge_kernels.cu's dev_check_range_compatible /
// dev_merge_into_slot, but indexes F via sparse F_entries (sorted by
// job_idx) instead of dense F_pair[n].
//
// On any merge whose resulting popcount(F_mask|X) > F_MAX_PER_STATE,
// FAIL LOUDLY (printf, set d_trunc_flag, return). Never silently drop.
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include <cstdio>
#include "sag_config.h"
#include "sag_types_v5.h"
#include "const_job_arrays_v5.cuh"  // for c_deadline (ECRTS19 segment-deadline clamp)

namespace sag {
namespace v5 {
namespace merge {

using namespace sag::config;

__device__ __forceinline__ void dev_copy_full_state_warp(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayoutV5 layout, int n, int m, int W,
    int lane)
{
    {
        const uint64_t* sD = layout.D(src_buf, src_idx);
        uint64_t*       dD = layout.D(dst_buf, dst_idx);
        if (lane < W) dD[lane] = sD[lane];

        const uint64_t* sX = layout.X(src_buf, src_idx);
        uint64_t*       dX = layout.X(dst_buf, dst_idx);
        if (lane < W) dX[lane] = sX[lane];

        const uint64_t* sF = layout.F_mask(src_buf, src_idx);
        uint64_t*       dF = layout.F_mask(dst_buf, dst_idx);
        if (lane < W) dF[lane] = sF[lane];
    }
    {
        const int2* sA = layout.A_pair(src_buf, src_idx);
        int2*       dA = layout.A_pair(dst_buf, dst_idx);
        for (int i = lane; i < m; i += WARP_SIZE) {
            dA[i] = sA[i];
        }
    }
    {
        // Copy F_count + F_entries[0..F_count). Lane 0 reads/writes F_count.
        const FEntryV5* sE = layout.F_entries(src_buf, src_idx);
        FEntryV5*       dE = layout.F_entries(dst_buf, dst_idx);
        int sf = *layout.F_count(src_buf, src_idx);
        sf = __shfl_sync(0xFFFFFFFF, sf, 0);
        if (lane == 0) {
            *layout.F_count(dst_buf, dst_idx) = sf;
        }
        // Each FEntryV5 is 12 bytes (3 int32). Copy as 3-int32 stride per
        // entry; spread across 32 lanes.
        const int32_t* sE32 = reinterpret_cast<const int32_t*>(sE);
        int32_t*       dE32 = reinterpret_cast<int32_t*>(dE);
        int total_int32 = sf * 3;
        for (int i = lane; i < total_int32; i += WARP_SIZE) {
            dE32[i] = sE32[i];
        }
    }
    if (lane == 0) {
        *layout.ovf(dst_buf, dst_idx) = *layout.ovf(src_buf, src_idx);
    }
}

__device__ __forceinline__ void dev_copy_full_state(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayoutV5 layout, int n, int m, int W)
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

    const int2* sA = layout.A_pair(src_buf, src_idx);
    int2*       dA = layout.A_pair(dst_buf, dst_idx);
    for (int i = 0; i < m; i++) dA[i] = sA[i];

    const FEntryV5* sE = layout.F_entries(src_buf, src_idx);
    FEntryV5*       dE = layout.F_entries(dst_buf, dst_idx);
    int sf = *layout.F_count(src_buf, src_idx);
    *layout.F_count(dst_buf, dst_idx) = sf;
    for (int e = 0; e < sf; e++) dE[e] = sE[e];

    *layout.ovf(dst_buf, dst_idx) = *layout.ovf(src_buf, src_idx);
}

// dev_check_state_dominates: returns a 2-bit code indicating whether
// state A "covers" state B (or vice versa) on the merge-affected fields
// other than F-entry intervals.
//
// "A dominates B" here means dev_merge_into_slot(B, A) would produce A
// unchanged, IGNORING F-entry interval widening (which is checked separately
// in dev_check_full_dominates):
//   bit 0 set: A.A_pair >= B's componentwise AND A.F_mask >= B.F_mask AND
//              A.X <= B.X (since merge ANDs X, A.X = A.X & B.X requires
//              B.X superset of A.X) AND A.ovf >= B.ovf (OR retains)
//   bit 1 set: symmetric
//
// Both bits set => byte-exact match on these dimensions (still need F-interval
// check for full dominance, plus F-entries for X-only jobs).
__device__ __forceinline__ int dev_check_state_dominates(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    bool a_dom_b = true;
    bool b_dom_a = true;

    // Per-core A_pair containment
    const int2* aA = layout.A_pair(buf_a, idx_a);
    const int2* bA = layout.A_pair(buf_b, idx_b);
    #pragma unroll 8
    for (int i = 0; i < m; i++) {
        int2 ai = aA[i];
        int2 bi = bA[i];
        // a covers b iff ai.x <= bi.x && ai.y >= bi.y
        if (ai.x > bi.x || ai.y < bi.y) a_dom_b = false;
        // b covers a iff bi.x <= ai.x && bi.y >= ai.y
        if (bi.x > ai.x || bi.y < ai.y) b_dom_a = false;
        if (!a_dom_b && !b_dom_a) return 0;
    }

    // F_mask superset: A>=B iff (B & ~A) == 0; A.X <= B.X iff (A.X & ~B.X) == 0
    const uint64_t* aFm = layout.F_mask(buf_a, idx_a);
    const uint64_t* bFm = layout.F_mask(buf_b, idx_b);
    const uint64_t* aX  = layout.X(buf_a, idx_a);
    const uint64_t* bX  = layout.X(buf_b, idx_b);
    for (int w = 0; w < W; w++) {
        uint64_t af = aFm[w];
        uint64_t bf = bFm[w];
        uint64_t ax = aX[w];
        uint64_t bx = bX[w];
        // A.F_mask superset of B requires (B.F_mask & ~A.F_mask) == 0
        if (a_dom_b && (bf & ~af) != 0ULL) a_dom_b = false;
        if (b_dom_a && (af & ~bf) != 0ULL) b_dom_a = false;
        // A.X <= B.X (X gets ANDed in merge; for a_dom_b the post-merge
        // X must equal A.X, so B.X superset of A.X). Equivalent: (ax & ~bx)==0
        if (a_dom_b && (ax & ~bx) != 0ULL) a_dom_b = false;
        if (b_dom_a && (bx & ~ax) != 0ULL) b_dom_a = false;
        if (!a_dom_b && !b_dom_a) return 0;
    }

    // ovf: post-merge ovf = A.ovf | B.ovf. For a_dom_b: A.ovf >= B.ovf.
    int32_t ao = *layout.ovf(buf_a, idx_a);
    int32_t bo = *layout.ovf(buf_b, idx_b);
    if (a_dom_b && ((bo & ~ao) != 0)) a_dom_b = false;
    if (b_dom_a && ((ao & ~bo) != 0)) b_dom_a = false;

    int result = 0;
    if (a_dom_b) result |= 1;
    if (b_dom_a) result |= 2;
    return result;
}

// dev_check_full_dominates: extends dev_check_state_dominates with F-interval
// containment. For A to fully dominate B, in addition to the partial check,
// for every job j in B.F_mask:
//   A.F_min[j] <= B.F_min[j] AND A.F_max[j] >= B.F_max[j]
// (A must have an entry for j too, which is implied by A.F_mask >= B.F_mask
// plus the layout invariant that F_entries covers F_mask.)
//
// Returns same bitmask as dev_check_state_dominates (0=neither, 1=A>=B fully,
// 2=B>=A fully, 3=byte-exact equal on these dims).
__device__ __forceinline__ int dev_check_full_dominates(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    int code = dev_check_state_dominates(buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    if (code == 0) return 0;

    bool a_dom_b = (code & 1);
    bool b_dom_a = (code & 2);

    const FEntryV5* aE = layout.F_entries(buf_a, idx_a);
    int aF = *layout.F_count(buf_a, idx_a);
    const FEntryV5* bE = layout.F_entries(buf_b, idx_b);
    int bF = *layout.F_count(buf_b, idx_b);

    // Both lists are sorted ascending by job_idx (V5 invariant). Walk B's
    // entries and look up the matching entry in A using a forward cursor.
    if (a_dom_b) {
        int ai = 0;
        for (int bi = 0; bi < bF; bi++) {
            int bj = bE[bi].job();
            while (ai < aF && aE[ai].job() < bj) ai++;
            if (ai >= aF || aE[ai].job() != bj) {
                // F_mask said A includes bj but no F-entry. Per layout
                // invariant this shouldn't happen for j in F_mask, but the
                // mask superset check above only ensured F_mask containment;
                // entries may be present for X-only jobs too. We require an
                // F-entry to compare intervals.
                a_dom_b = false; break;
            }
            if (aE[ai].f_min > bE[bi].f_min || aE[ai].f_max < bE[bi].f_max) {
                a_dom_b = false; break;
            }
        }
    }

    if (b_dom_a) {
        int bi = 0;
        for (int ai = 0; ai < aF; ai++) {
            int aj = aE[ai].job();
            while (bi < bF && bE[bi].job() < aj) bi++;
            if (bi >= bF || bE[bi].job() != aj) {
                b_dom_a = false; break;
            }
            if (bE[bi].f_min > aE[ai].f_min || bE[bi].f_max < aE[ai].f_max) {
                b_dom_a = false; break;
            }
        }
    }

    int result = 0;
    if (a_dom_b) result |= 1;
    if (b_dom_a) result |= 2;
    return result;
}

// dev_check_compat_and_dominance (iter620+ fusion of dev_check_range_compatible
// + dev_check_full_dominates).  Single forward pass over A_pair / F_mask / X /
// ovf / F_entries, computing Cond 2 + Cond 3 (range compat) AND dominance bits
// in lockstep.  Byte-identical to the chained two-call sequence:
//
//   if (dev_check_range_compatible(A, B, ..., true)) {
//       int dom = dev_check_full_dominates(A, B, ...);
//       merge_action = (dom & 1) ? 1 : (dom & 2) ? 2 : 3;
//   } else {
//       merge_action = 0;
//   }
//
// Returns:
//   0 = incompatible (Cond 2 or Cond 3 failed)
//   1 = compatible AND A fully dominates B    -> drop B
//   2 = compatible AND B fully dominates A    -> replace slot with B
//   3 = compatible, neither fully dominates   -> normal widening merge
//
// Parameter ordering matches dev_check_full_dominates (A, B), NOT
// dev_check_range_compatible.  The original call site swapped (src, slot) to
// (slot, src) between the two calls; here we use (A=slot, B=src) consistently.
// All overlap predicates are symmetric in (A, B) so this is byte-identical
// for the Cond 2 / Cond 3 / dominance checks.  The F_mask we walk for
// Cond 3 is A.F_mask (matches "B.F_mask" in the original call where B was
// slot; with arg-swap that becomes A.F_mask).
// Internal templated helper. EMIT_FIELDS_EQ=true: also captures whether
// A and B agree bit-identically on F_mask + X + ovf (snapshot taken before
// the F_entries pass which may break dominance flags). rtss17's hybrid
// widening predicate consumes this flag to avoid a second 2W+1 global read.
template<bool EMIT_FIELDS_EQ>
__device__ __forceinline__ int dev_check_compat_and_dominance_impl(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W,
    bool* out_fields_eq)
{
    bool overlap   = true;
    bool a_dom_b   = true;
    bool b_dom_a   = true;
    bool fields_eq = true;  // F_mask + X + ovf bit-equal across all words

    // ---- Pass 1: A_pair (m elements). Combines Cond 2 + dominance. ---------
    {
        const int2* aA = layout.A_pair(buf_a, idx_a);
        const int2* bA = layout.A_pair(buf_b, idx_b);
        #pragma unroll 8
        for (int i = 0; i < m; i++) {
            int2 ai = aA[i];
            int2 bi = bA[i];
            // Cond 2 overlap: NOT (ai.x > bi.y+1 || bi.x > ai.y+1)
            if (ai.x > bi.y + 1) overlap = false;
            if (bi.x > ai.y + 1) overlap = false;
            // Early-exit when overlap fails -- the function returns 0
            // immediately regardless of dominance flags, so remaining
            // A_pair iterations are wasted work.  Sound: overlap is
            // monotonic-false, dominance updates after this point would
            // be discarded by the `return 0` below.  EMIT_FIELDS_EQ
            // callers (rtss17) initialize *out_fields_eq=false; they
            // only consult it on code-3 returns, so leaving it unwritten
            // when we return 0 is safe.
            if (!overlap) return 0;
            // a_dom_b: ai.x <= bi.x && ai.y >= bi.y
            if (ai.x > bi.x || ai.y < bi.y) a_dom_b = false;
            // b_dom_a: bi.x <= ai.x && bi.y >= ai.y
            if (bi.x > ai.x || bi.y < ai.y) b_dom_a = false;
        }
    }
    if (!overlap) return 0;

    // ---- Pass 2: F_mask + X (W elements). Dominance only, no Cond effect. --
    {
        const uint64_t* aFm = layout.F_mask(buf_a, idx_a);
        const uint64_t* bFm = layout.F_mask(buf_b, idx_b);
        const uint64_t* aX  = layout.X(buf_a, idx_a);
        const uint64_t* bX  = layout.X(buf_b, idx_b);
        for (int w = 0; w < W; w++) {
            uint64_t af = aFm[w];
            uint64_t bf = bFm[w];
            uint64_t ax = aX[w];
            uint64_t bx = bX[w];
            if (a_dom_b && (bf & ~af) != 0ULL) a_dom_b = false;
            if (b_dom_a && (af & ~bf) != 0ULL) b_dom_a = false;
            if (a_dom_b && (ax & ~bx) != 0ULL) a_dom_b = false;
            if (b_dom_a && (bx & ~ax) != 0ULL) b_dom_a = false;
            if constexpr (EMIT_FIELDS_EQ) {
                if (fields_eq && (af != bf || ax != bx)) fields_eq = false;
            }
        }
    }

    // ---- Pass 3: ovf. ------------------------------------------------------
    {
        int32_t ao = *layout.ovf(buf_a, idx_a);
        int32_t bo = *layout.ovf(buf_b, idx_b);
        if (a_dom_b && ((bo & ~ao) != 0)) a_dom_b = false;
        if (b_dom_a && ((ao & ~bo) != 0)) b_dom_a = false;
        if constexpr (EMIT_FIELDS_EQ) {
            if (fields_eq && ao != bo) fields_eq = false;
        }
    }

    // Snapshot field-equality state BEFORE Pass 4. fields_eq accumulator
    // tracks F_mask + X + ovf bit-equality across ALL W words, INDEPENDENT
    // of A_pair. Pass 4 walks F_entries which does NOT touch F_mask/X/ovf,
    // so freezing the flag here is sound for rtss17's hybrid widening
    // predicate (which admits widening only when these structural fields
    // are bit-identical, regardless of A_pair).
    if constexpr (EMIT_FIELDS_EQ) {
        *out_fields_eq = fields_eq;
    }

    // ---- Pass 4: F_entries dual-cursor merge walk. -------------------------
    // Original work split:
    //   range_compatible (over A.F_mask):
    //     for J in A.F_mask, lookup A.F_entry / B.F_entry; check
    //     a_min > b_max+1 || b_min > a_max+1.
    //   full_dominates (a_dom_b):
    //     walk B.F_entries; for each J, find A.F_entry; check
    //     A.f_min <= B.f_min && A.f_max >= B.f_max.
    //   full_dominates (b_dom_a): symmetric (walk A.F_entries).
    //
    // Fused: single dual-cursor pass over both lists.  At each position we
    // see (A entry only) | (B entry only) | (both for same J).  We dispatch
    // overlap + dominance updates inline.
    {
        const FEntryV5* aE = layout.F_entries(buf_a, idx_a);
        int aF = *layout.F_count(buf_a, idx_a);
        const FEntryV5* bE = layout.F_entries(buf_b, idx_b);
        int bF = *layout.F_count(buf_b, idx_b);
        const uint64_t* aFm = layout.F_mask(buf_a, idx_a);

        // Hoist aFm[0..W-1] into registers. Pass 4's inAFm bit-test runs
        // up to (aF + bF) iterations (~14 typical, max 64). Reading from
        // an unrolled register table avoids the indirect global-memory
        // address resolution per iteration. MAX_BITSET_WORDS = 8 covers
        // n <= 512 with comfortable register budget.
        uint64_t aFm_reg[MAX_BITSET_WORDS];
        #pragma unroll
        for (int w = 0; w < MAX_BITSET_WORDS; w++) {
            aFm_reg[w] = (w < W) ? aFm[w] : 0ULL;
        }

        int ai = 0;
        int bi = 0;
        while (ai < aF || bi < bF) {
            int32_t aj = (ai < aF) ? aE[ai].job() : INT32_MAX;
            int32_t bj = (bi < bF) ? bE[bi].job() : INT32_MAX;
            // Once both dominance flags are dead, only the Cond-3 overlap
            // check matters (the function returns 3 if no overlap-fail
            // is found). Skip the dominance-update branches.
            const bool any_dom = (a_dom_b || b_dom_a);
            if (aj == bj) {
                int32_t a_min = aE[ai].f_min;
                int32_t a_max = aE[ai].f_max;
                int32_t b_min = bE[bi].f_min;
                int32_t b_max = bE[bi].f_max;
                int J = aj;
                // Cond 3 overlap iff J is in A.F_mask (matches original
                // range_compatible(src,slot,...) walking slot.F_mask = A.F_mask).
                int w = J >> 6;
                int bit = J & 63;
                bool inAFm = ((aFm_reg[w] >> bit) & 1ULL) != 0ULL;
                if (inAFm) {
                    if (a_min > b_max + 1) return 0;
                    if (b_min > a_max + 1) return 0;
                }
                if (any_dom) {
                    // Dominance: full containment of intervals.
                    if (a_dom_b && (a_min > b_min || a_max < b_max)) a_dom_b = false;
                    if (b_dom_a && (b_min > a_min || b_max < a_max)) b_dom_a = false;
                }
                ai++;
                bi++;
            } else if (aj < bj) {
                // Only A has entry for aj.
                int32_t a_min = aE[ai].f_min;
                int J = aj;
                int w = J >> 6;
                int bit = J & 63;
                bool inAFm = ((aFm_reg[w] >> bit) & 1ULL) != 0ULL;
                // Cond 3 overlap with phantom B (b_min=0, b_max=0): need
                // !(a_min > 0+1) && !(0 > a_max+1).  The second is always
                // false (since a_max>=0).  Reduces to a_min <= 1 always
                // OK if a_min could be 0 or 1.  If a_min > 1, fail.
                if (inAFm) {
                    if (a_min > 1) return 0;
                    // 0 > a_max+1 is impossible.
                }
                // Dominance: B has no entry for aj; matches the original
                // full_dominates walk over B.F_entries (a_dom_b unaffected
                // since aj is not in B's entry list) but b_dom_a walks A's
                // entries -- A has aj, B doesn't -> b_dom_a fails.
                if (b_dom_a) b_dom_a = false;
                ai++;
            } else { // bj < aj
                // Only B has entry for bj.
                int32_t b_min = bE[bi].f_min;
                int J = bj;
                int w = J >> 6;
                int bit = J & 63;
                bool inAFm = ((aFm_reg[w] >> bit) & 1ULL) != 0ULL;
                // Cond 3 overlap iff J in A.F_mask.  In the original code,
                // walking A.F_mask only fires range_compat tests for J that
                // are in A.F_mask AND have an entry in either A.F_entries or
                // B.F_entries (otherwise both lookups return (0,0) and overlap
                // trivially holds).  Here B has the entry; check overlap with
                // A's phantom (a_min=0, a_max=0).
                if (inAFm) {
                    if (b_min > 1) return 0;
                }
                // Dominance: A has no entry for bj; a_dom_b walks B.F_entries
                // -> A missing this entry -> a_dom_b fails.
                if (a_dom_b) a_dom_b = false;
                bi++;
            }
        }
    }

    if (a_dom_b) return 1;
    if (b_dom_a) return 2;
    return 3;
}

// Default predicate: no fields-eq output, byte-identical to the historical
// signature used by V5/ECRTS19/ECRTS22 callers.
__device__ __forceinline__ int dev_check_compat_and_dominance(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    return dev_check_compat_and_dominance_impl<false>(
        buf_a, idx_a, buf_b, idx_b, layout, n, m, W, nullptr);
}

// Variant for rtss17 hybrid merge: emits F_mask+X+ovf bit-equality flag.
// When the flag is set on a code-3 return, sound widening is admitted; the
// caller does NOT need to re-read the field arrays.
__device__ __forceinline__ int dev_check_compat_and_dominance_with_fields_eq(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W,
    bool* out_fields_eq)
{
    return dev_check_compat_and_dominance_impl<true>(
        buf_a, idx_a, buf_b, idx_b, layout, n, m, W, out_fields_eq);
}

// dev_check_range_compatible (RTSS Eq. 24-25)
//   Cond 2: per-core A_min/A_max intervals overlap.
//   Cond 3 (optional): per-job F_min/F_max intervals overlap (only for jobs
//                      in B's F_mask).
__device__ __forceinline__ bool dev_check_range_compatible(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W,
    bool useJobFinishTimes = false)
{
    const int2* aA = layout.A_pair(buf_a, idx_a);
    const int2* bA = layout.A_pair(buf_b, idx_b);

    #pragma unroll 8
    for (int i = 0; i < m; i++) {
        int2 ai = aA[i];
        int2 bi = bA[i];
        if (ai.x > bi.y + 1) return false;
        if (bi.x > ai.y + 1) return false;
    }

    if (useJobFinishTimes) {
        const uint64_t* fm = layout.F_mask(buf_b, idx_b);
        const FEntryV5* aE = layout.F_entries(buf_a, idx_a);
        int aF = *layout.F_count(buf_a, idx_a);
        const FEntryV5* bE = layout.F_entries(buf_b, idx_b);
        int bF = *layout.F_count(buf_b, idx_b);

        // iter501 (Candidate B-prime): dual-cursor sparse-F lookup.  aE[]
        // and bE[] are sorted ascending by job_idx; the bit walk over fm[]
        // is also ascending, so cursors that only advance never need to
        // restart. This collapses two O(F_count) linear scans per bit into
        // O(1) amortised. Byte-identical: same (a_min, a_max, b_min, b_max)
        // values as before for any given J, just looked up faster.
        int cur_a = 0;
        int cur_b = 0;
        for (int w = 0; w < W; w++) {
            uint64_t bits = fm[w];
            while (bits) {
                int bit = __ffsll(bits) - 1;
                int J = w * 64 + bit;
                if (J >= n) break;

                while (cur_a < aF && aE[cur_a].job() < J) cur_a++;
                int32_t a_min = 0, a_max = 0;
                if (cur_a < aF && aE[cur_a].job() == J) {
                    a_min = aE[cur_a].f_min;
                    a_max = aE[cur_a].f_max;
                }

                while (cur_b < bF && bE[cur_b].job() < J) cur_b++;
                int32_t b_min = 0, b_max = 0;
                if (cur_b < bF && bE[cur_b].job() == J) {
                    b_min = bE[cur_b].f_min;
                    b_max = bE[cur_b].f_max;
                }

                if (a_min > b_max + 1) return false;
                if (b_min > a_max + 1) return false;
                bits &= bits - 1;
            }
        }
    }
    return true;
}

// dev_merge_into_slot (RTSS Eq. 26-29)
//   X:   AND
//   A:   componentwise (min, max)
//   F_min/F_max widening over (src.F_mask | dst.F_mask)
//   F_mask: OR
//   ovf:    OR
//
// Sparse-aware: builds a new F_entries array indexed by the post-merge
// F_avail = (newF_mask | newX) where newF_mask = src.F_mask|dst.F_mask
// and newX = src.X & dst.X.
//
// Output goes into dst's F_entries / F_count. On overflow > F_MAX_PER_STATE,
// FAIL LOUDLY.
//
// Template param `CLAMP_FMAX_TO_DEADLINE` (default false): when true,
// the per-job widened f_max is clamped to that job's deadline (ECRTS 2019
// Lemma 4 / §V segment-deadline). Soundness: f_max is a per-segment
// "latest finish time" upper bound. K2's deadline check lets f_max
// exceed deadline (it's flag-of-doom but not the gate); however the
// clamp tightens the merge-induced over-approximation without losing
// any reachable schedule (every input state already had f_max <= max
// possible reachable finish time, which is itself bounded by deadline
// for any in-flight schedule).
template<bool CLAMP_FMAX_TO_DEADLINE = false>
__device__ __forceinline__ void dev_merge_into_slot(
    const char* src_buf, int src_idx,
    char*       dst_buf, int dst_idx,
    SAGStateLayoutV5 layout, int n, int m, int W,
    int* d_trunc_flag,
    int  layer_dbg)
{
    // X: AND
    const uint64_t* sX = layout.X(src_buf, src_idx);
    uint64_t*       dX = layout.X(dst_buf, dst_idx);
    for (int w = 0; w < W; w++) dX[w] &= sX[w];

    // A: min/max
    const int2* sA = layout.A_pair(src_buf, src_idx);
    int2*       dA = layout.A_pair(dst_buf, dst_idx);
    for (int i = 0; i < m; i++) {
        int2 si_pair = sA[i];
        int2 di_pair = dA[i];
        if (si_pair.x < di_pair.x) di_pair.x = si_pair.x;
        if (si_pair.y > di_pair.y) di_pair.y = si_pair.y;
        dA[i] = di_pair;
    }

    // F_mask union (compute the new F_mask separately so we can build
    // F_entries before overwriting dst's F_mask).
    const uint64_t* sFm = layout.F_mask(src_buf, src_idx);
    uint64_t*       dFm = layout.F_mask(dst_buf, dst_idx);

    // Pre-fetch src F_entries / count BEFORE we modify dst (we'll write
    // back to a temp area then commit to dst at the end).
    const FEntryV5* sE = layout.F_entries(src_buf, src_idx);
    int sF = *layout.F_count(src_buf, src_idx);
    const FEntryV5* dE_in = layout.F_entries(dst_buf, dst_idx);
    int dF = *layout.F_count(dst_buf, dst_idx);

    // Build the merged entries into a small register-stack buffer; bounded by
    // F_MAX_PER_STATE. We need:
    //   new F_avail = (sFm | dFm) | (sX & dX_post)
    // For each k in new F_avail (ascending):
    //   - if k in sFm and k in dFm: F = (min(s.fmin,d.fmin), max(s.fmax,d.fmax))
    //   - else if k in sFm only:    F = src's
    //   - else if k in dFm only:    F = dst's
    //   - else (k in newX only):    F = src or dst's stale value (RTSS 26-29
    //                                doesn't widen non-F_mask jobs); we use
    //                                whichever has an entry, falling back to
    //                                (0,0) only if neither has it.
    //
    // Direct-write into stack temp, then memcpy to dst.

    FEntryV5 tmp[F_MAX_PER_STATE];
    int tmp_count = 0;

    // dst.X has already been AND-ed; reread.
    const uint64_t* dX_post = layout.X(dst_buf, dst_idx);

    // iter501 (Candidate B-prime): dual-cursor sparse-F lookup.
    //
    // sE[] and dE_in[] are both sorted ascending by job_idx (invariant from
    // v5_build_child_F_entries / dev_merge_into_slot itself). We iterate
    // F_avail bits in ascending k order. The previous code did
    // sparse_F_lookup(sE, sF, k) and sparse_F_lookup(dE_in, dF, k) per bit,
    // each an O(F_count) linear scan -- total O(popcount * F_count) per
    // merge.
    //
    // With dual cursors that only ever advance, each lookup becomes O(1)
    // amortised, cutting F-entry build cost from O(p*F) to O(p+F). On n28
    // workloads this trims ~70-80% of merge body work and closes the bulk
    // of the 52% kernel time the merge consumes.
    //
    // Byte-identical: both sE/dE_in are sorted, so cursor lookups produce
    // exactly the same (s_min, s_max, d_min, d_max) values that the old
    // linear scans returned. The output ordering / overflow semantics /
    // FAIL-LOUD path are unchanged.
    int cur_s = 0;
    int cur_d = 0;

    for (int w = 0; w < W; w++) {
        uint64_t fmask_union = sFm[w] | dFm[w];
        uint64_t xint = dX_post[w];  // dst.X &= src.X already done above
        uint64_t avail = fmask_union | xint;
        while (avail) {
            int bit = __ffsll(avail) - 1;
            avail &= avail - 1;
            int k = w * 64 + bit;
            if (k >= n) continue;

            bool in_sFm = (sFm[w] >> bit) & 1ULL;
            bool in_dFm = (dFm[w] >> bit) & 1ULL;

            // Dual-cursor lookup in sE for k.
            while (cur_s < sF && sE[cur_s].job() < k) cur_s++;
            int32_t s_min = 0, s_max = 0;
            bool has_s = (cur_s < sF && sE[cur_s].job() == k);
            if (has_s) {
                s_min = sE[cur_s].f_min;
                s_max = sE[cur_s].f_max;
            }
            // Dual-cursor lookup in dE_in for k.
            while (cur_d < dF && dE_in[cur_d].job() < k) cur_d++;
            int32_t d_min = 0, d_max = 0;
            bool has_d = (cur_d < dF && dE_in[cur_d].job() == k);
            if (has_d) {
                d_min = dE_in[cur_d].f_min;
                d_max = dE_in[cur_d].f_max;
            }

            int32_t out_min, out_max;
            if (in_sFm && in_dFm) {
                out_min = (has_s && (!has_d || s_min < d_min)) ? s_min : (has_d ? d_min : s_min);
                out_max = (has_s && (!has_d || s_max > d_max)) ? s_max : (has_d ? d_max : s_max);
            } else if (in_sFm) {
                out_min = has_s ? s_min : 0;
                out_max = has_s ? s_max : 0;
            } else if (in_dFm) {
                out_min = has_d ? d_min : 0;
                out_max = has_d ? d_max : 0;
            } else {
                // k in newX only (no F_mask). Carry whichever side has data;
                // dst's takes priority if both have, matching the dense
                // legacy semantics (dst's F_pair[k] is left untouched for
                // non-F_mask k in legacy merge_kernels.cu line 184-209).
                if (has_d) { out_min = d_min; out_max = d_max; }
                else if (has_s) { out_min = s_min; out_max = s_max; }
                else           { out_min = 0;     out_max = 0;     }
            }

            // ECRTS 2019 segment-deadline clamp ATTEMPTED (Lemma 4 §V).
            // Result: TRUNCATED at layer 66/81 on n16_m4_u20/t000 ecrts19
            // because K2's deadline check only flags f_min > deadline, NOT
            // f_max > deadline. f_max can validly exceed deadline; clamping
            // it loses information about wider finish-time intervals,
            // making merges less compatible (containment strict-er) and
            // state count explode. The agent plan's soundness argument
            // was incorrect on this point. Disabled. See
            // IMPROVEMENT_BACKLOG.md for the lesson.
            if constexpr (CLAMP_FMAX_TO_DEADLINE) {
                // Intentionally a no-op until a sound formulation exists.
                // (Plan: only clamp f_max for FRESH F-entries from jobs
                // dispatched in current layer, not INHERITED ones from
                // prior layers. Future iteration.)
            }
            if (tmp_count >= F_MAX_PER_STATE) {
                printf("[V5 MERGE OVERFLOW] F_MAX_PER_STATE=%d exceeded at "
                       "layer=%d slot=%d. Increase SAG_V5_F_MAX_PER_STATE.\n",
                       F_MAX_PER_STATE, layer_dbg, dst_idx);
                if (d_trunc_flag) atomicExch(d_trunc_flag, 1);
                return;
            }
            // Preserve packed (job, p_committed) bits when one source has
            // an entry. ECRTS 2022's parallelism-aware merge predicate
            // would have already rejected this merge if p differs across
            // common-job entries, so picking either side's packed bits
            // is consistent. For non-gang variants both sides have p=0
            // (upper bits zero), so the packed copy reduces to plain k.
            int32_t packed_jp = k;
            if (has_s && has_d) {
                // Common job: both should have same p (else merge would
                // have been rejected for ECRTS22). Take src's bits.
                packed_jp = sE[cur_s].job_idx;
            } else if (has_s) {
                packed_jp = sE[cur_s].job_idx;
            } else if (has_d) {
                packed_jp = dE_in[cur_d].job_idx;
            }
            tmp[tmp_count].job_idx = packed_jp;
            tmp[tmp_count].f_min   = out_min;
            tmp[tmp_count].f_max   = out_max;
            tmp_count++;
        }
    }

    // Commit tmp -> dst (sequential; lane 0 only context typically).
    FEntryV5* dE_out = layout.F_entries(dst_buf, dst_idx);
    for (int e = 0; e < tmp_count; e++) dE_out[e] = tmp[e];
    *layout.F_count(dst_buf, dst_idx) = tmp_count;

    // F_mask: OR (commit AFTER F_entries so the read above used pre-OR dFm).
    for (int w = 0; w < W; w++) dFm[w] |= sFm[w];

    // ovf: OR
    *layout.ovf(dst_buf, dst_idx) |= *layout.ovf(src_buf, src_idx);
}

} // namespace merge
} // namespace v5
} // namespace sag
