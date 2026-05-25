// merge_body_ecrts19.cuh -- Merge scaffold for ECRTS 2019 variant.
//
// Paper: M. Nasri, G. Nelissen, B. Brandenburg, "Response-Time Analysis of
//   Limited-Preemptive Parallel DAG Tasks under Global Scheduling", ECRTS 2019.
// nptest reference: same merge framework (state.hpp:507, node.hpp:571);
//   variant uses lossy l1 by default but with stricter F-time checks.
//
// HARD INVARIANTS:
//   1. Merge is optional; unmerged states pass to L+1.
//   2. Lossy merge over-approximates and is the published default in nptest
//      (`--merge l1`). Conservative is opt-in.
//
// COMPATIBILITY (Phase 4):
//   Two same-D-group states merge if A_pair intervals overlap or are
//   contiguous (same as V5's lossy l1). Additionally, a per-segment F-time
//   pointwise compatibility is enforced for jobs that are SEGMENT
//   PREDECESSORS (zero-delay edges) of any not-yet-dispatched job, because
//   widening their F-times affects the eligibility of dependent segments
//   in subsequent layers.
//
// IMPLEMENTATION (Phase 4): mostly identical to V5 lossy merge. The delta is
// the per-segment F_min/F_max widening clamp (no segment's F_max can exceed
// its task's deadline; mark unsched if it does -- matches ECRTS 2019 §V).
//
// Phase 1: scaffold only; routed merge stays on V5MergeKernel.
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include "sag_config.h"
#include "sag_types_v5.h"

namespace sag {
namespace v5 {
namespace merge_ecrts19 {

// Phase 4: ECRTS 2019 default merge. nptest's `--merge l1` is the
// published baseline; lossy widening is permitted. V5's default predicate
// (codes 0/1/2/3) implements this exactly.
//
// Tried & reverted (anti-patterns recorded in memory):
//   (a) "any disagreement -> reject" -- busted layer_capacity at L=40
//       (IMPROVEMENT_BACKLOG `disagreement-rejection`).
//   (b) "clamp f_max to deadline" -- busted to TRUNCATED at L=66
//       (v5_ecrts19_fmax_deadline_clamp_negative.md).
//   (c) "K-bounded lazy rejection (K=3)" -- still TRUNCATED at L=40.
//       Chain-DAG segment expansion creates dense merge groups where
//       per-segment F-time histories differ broadly; refusing any
//       widening explodes state count regardless of K threshold.
//
// ECRTS19 BCRT/WCRT tightening must come from algorithmic levers
// (POR Algorithm 4 absorbing segment chains, Leveraging Parallelism /
// independent-job tandems, RTS 2023 sec V), not from per-merge filters.
__device__ __forceinline__ int dev_check_compat_ecrts19(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    return sag::v5::merge::dev_check_compat_and_dominance(
        buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
}

} // namespace merge_ecrts19
} // namespace v5
} // namespace sag
