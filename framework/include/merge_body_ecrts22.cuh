// merge_body_ecrts22.cuh -- Merge scaffold for ECRTS 2022 variant.
//
// Paper: G. Nelissen, J. Marce-i-Igual, M. Nasri, "Response-Time Analysis for
//   Non-Preemptive Periodic Moldable Gang Tasks", ECRTS 2022.
// nptest reference: include/global/state.hpp:507 can_merge_with(); ECRTS 2022
//   §IV uses lossy l1 with the addition that two states with the same D-key
//   that differ in the parallelism choice for any j in F_mask must NOT merge
//   (their job_finish_times are tagged with p, see K2 notes).
//
// HARD INVARIANTS:
//   1. Merge is optional; unmerged states pass forward.
//   2. Two states with different parallelism for any common dispatched job
//      MUST NOT merge -- their finish-time semantics differ.
//
// COMPATIBILITY (Phase 3):
//   Same as V5 lossy l1, plus: for every job j in (F_mask_A & F_mask_B), the
//   parallelism encoded in F_entries[j] must match. If parallelism differs,
//   incompatible (return 0). This guarantees sound widening.
//
// IMPLEMENTATION NOTES (Phase 3):
//   - When parallelism is encoded in f_min's high bits (option (ii) above),
//     extracting and comparing is a few ALU ops per F_entry compare.
//   - V5MergeKernel's outer loop (groups, lanes, slots) is unchanged; only
//     dev_check_compat_and_dominance gets a parallelism-aware overload.
//
// Phase 1: scaffold only.
//
// Generic across A/H/B-series NV GPUs.

#pragma once
#include "sag_config.h"
#include "sag_types_v5.h"
#include "merge_body_v5.cuh"

namespace sag {
namespace v5 {
namespace merge_ecrts22 {

__device__ __forceinline__ int dev_check_compat_ecrts22(
    const char* buf_a, int idx_a,
    const char* buf_b, int idx_b,
    SAGStateLayoutV5 layout, int n, int m, int W)
{
    int code = sag::v5::merge::dev_check_compat_and_dominance(
        buf_a, idx_a, buf_b, idx_b, layout, n, m, W);
    if (code != 3) return code;
    // ECRTS 2022 paper requirement: states with different gang parallelism
    // for any common F_mask job MUST NOT merge. F_entries packs p_committed
    // in upper 16 bits (sag_types_v5.h::FEntryV5). Walk both sides via
    // dual cursor and reject widening on common-job p mismatch. Sentinel
    // p=0 is treated as "no info"; only reject when both sides have nonzero
    // p AND they disagree.
    const FEntryV5* aE = layout.F_entries(buf_a, idx_a);
    const FEntryV5* bE = layout.F_entries(buf_b, idx_b);
    int aF = *layout.F_count(buf_a, idx_a);
    int bF = *layout.F_count(buf_b, idx_b);
    int ai = 0, bi = 0;
    while (ai < aF && bi < bF) {
        int aj = aE[ai].job();
        int bj = bE[bi].job();
        if (aj == bj) {
            int ap = aE[ai].p_committed();
            int bp = bE[bi].p_committed();
            if (ap != 0 && bp != 0 && ap != bp) return 0;
            ai++; bi++;
        } else if (aj < bj) {
            ai++;
        } else {
            bi++;
        }
    }
    return 3;
}

} // namespace merge_ecrts22
} // namespace v5
} // namespace sag
