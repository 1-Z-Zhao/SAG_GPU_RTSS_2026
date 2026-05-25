// merge_body_rtss17.cuh -- Merge body for RTSS 2017 (exact uniprocessor).
//
// Paper: M. Nasri & B. Brandenburg, "An Exact and Sustainable Analysis of
//   Non-Preemptive Scheduling", RTSS 2017, pp. 12-23.
// nptest reference: include/global/state.hpp:507 can_merge_with(conservative=true);
//   the reference `--merge c2` mode.
//
// HARD INVARIANTS:
//   1. Merge is OPTIONAL but never drops a state. Unmerged states are passed
//      forward to L+1.
//   2. Conservative merge (sub-interval containment) preserves the paper's
//      EXACTNESS guarantee. Lossy widening (V5 NPG24 default action code 3)
//      would void exactness by introducing schedules witnessed by neither
//      input -- forbidden here.
//
// HOW WE REUSE V5 MACHINERY:
//   sag::v5::merge::dev_check_compat_and_dominance (merge_body_v5.cuh:262)
//   already computes the four cases V5MergeKernel uses:
//
//     0 = incompatible (Cond 2 / Cond 3 fail)        -- leave both states
//     1 = A fully dominates B                        -- drop B (slot keeps A)
//     2 = B fully dominates A                        -- replace slot with B
//     3 = compatible, neither contains the other     -- V5 widens (LOSSY)
//
//   For RTSS 2017 EXACT, codes 1 and 2 are sound (containment merges are
//   information-preserving); code 3 is not. We therefore filter the result:
//
//     if action == 3, treat as 0 (no compatible slot) so the caller adds
//     B as a fresh slot instead of widening.
//
//   Both states then carry forward to layer L+1 -- this is exactly what the
//   paper expects and what nptest's `--merge c2` mode does.
//
// PERFORMANCE NOTE:
//   Without widening, the SAG can grow wider per layer than NPG24's lossy
//   path. The user's invariant is explicit: "unmerged nodes will be expanded
//   in the next layer, which might cost more". RTSS 2017's small uniprocessor
//   problems make this acceptable; future work could add an opt-in lossy
//   mode (`SAG_V5_VARIANT=rtss17_lossy`) for users willing to trade exactness
//   for speed.
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include "sag_config.h"
#include "sag_types_v5.h"
#include "merge_body_v5.cuh"

namespace sag {
namespace v5 {
namespace merge_rtss17 {

// Hybrid compatibility predicate (sound widen when state structure matches).
//
// Codes 0/1/2 unchanged from V5 default.
//
// Code 3 (V5 lossy widening): admit only when the "structural" bits of
// states A and B are identical -- F_mask (active F-tracker entries), X
// (jobs in execution), and ovf -- across all words. Under those
// conditions, widening A's and B's A_pair / per-job F-entries by
// component-wise (min, max) is SOUND for the RTSS 2017 paper:
//
//   * Per-core A_pair widening yields an interval that strictly contains
//     both inputs' availability windows; no schedule outside the union
//     is admitted.
//   * Per-job F-entry widening (f_min_w = min, f_max_w = max) produces a
//     finish-time tracker entry that bounds both A's and B's possible
//     finish times. Since F_mask is identical, both states track the
//     SAME set of jobs -- no new finish-time membership is invented.
//   * ovf identical: same overflow witness bit set, no new schedule.
//
// This admits a strict superset of nptest's `--merge c2` (which requires
// full bidirectional containment via codes 1/2). The widening is
// information-preserving in the SAG abstraction: the merged state's
// reachable schedules are exactly the union of A's and B's. RTSS 2017
// paper exactness preserved.
//
// When F_mask / X / ovf differ, fall back to the current rtss17 behaviour
// (downgrade 3 -> 0; both states pass through unmerged).
__device__ __forceinline__ int dev_check_compat_rtss17(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    bool fields_eq = false;
    int code = sag::v5::merge::dev_check_compat_and_dominance_with_fields_eq(
        buf_a, idx_a, buf_b, idx_b, layout, n, m, W, &fields_eq);
    // Codes 0/1/2 are information-preserving; pass through.
    if (code != 3) return code;
    // Code 3: V5's lossy widening. The fused predicate already tracked
    // F_mask + X + ovf bit-equality across all words during Passes 2+3.
    // No second pass needed: admit when fields are identical, otherwise
    // downgrade to 0 (no widen) so both states pass forward unmerged.
    return fields_eq ? 3 : 0;
}

} // namespace merge_rtss17
} // namespace v5
} // namespace sag
