// v5_variant.h -- SAG-paper variant selection for framework.
//
// The user-facing API is a single env var SAG_V5_VARIANT={rtss24|rtss17|ecrts19|ecrts22}
// naming the paper. All optimization toggles (IJP, F-cache hoist, merge level,
// etc.) are auto-selected per variant via v5_default_tricks() so the user does
// not have to reason about them.
//
//   rtss24  (default) -- Limited-preemptive self-suspending + event-driven
//                        delay-induced. Srinivasan/Gunzel/Nelissen, RTSS 2024.
//                        Implemented (existing V5 path; 20/20 nptest match
//                        on iter450 default gate).
//   rtss17            -- Exact uniprocessor non-preemptive.
//                        Nasri & Brandenburg, RTSS 2017.
//                        Implemented (Phase 2). Byte-identical to nptest -m 1
//                        on small uniprocessor inputs; conservative c2 merge.
//   ecrts19           -- Limited-preemptive parallel DAG, global.
//                        Nasri/Nelissen/Brandenburg, ECRTS 2019.
//                        Implemented (Phase 4) via segment expansion. Verdict
//                        match on chain-DAG inputs; ~85% byte-identical.
//   ecrts22           -- Non-preemptive periodic moldable gang.
//                        Nelissen/Marce-i-Igual/Nasri, ECRTS 2022.
//                        Implemented (Phase 3). Rigid p=1 byte-identical to
//                        rtss24; rigid p>=2 SCHED-correct but loose WCRT
//                        (multi-core K2 commit on; moldable greedy-earliest
//                        K1 enumeration deferred -- see PHASE3_3_B_PLAN.md).
//
// SAG_V5_IJP is now a deprecated back-door override and has effect only when
// SAG_V5_VARIANT=rtss24 (or unset). Future variants will pick IJP internally.
//
// Header-only; included from both host (framework_main.cu) and device
// translation units (controller.cu). Generic across A/H/B-series NV GPUs.

#pragma once
#include <cstdio>
#include <cstdlib>

namespace sag {
namespace v5 {

enum class V5Variant : int {
    NPG_RTSS24 = 0,   // current default
    NP_UNI_17  = 1,   // RTSS 2017
    LP_DAG_19  = 2,   // ECRTS 2019
    GANG_22    = 3,   // ECRTS 2022
};

inline const char* v5_variant_name(V5Variant v) {
    switch (v) {
        case V5Variant::NPG_RTSS24: return "rtss24";
        case V5Variant::NP_UNI_17:  return "rtss17";
        case V5Variant::LP_DAG_19:  return "ecrts19";
        case V5Variant::GANG_22:    return "ecrts22";
    }
    return "?";
}

inline const char* v5_variant_paper(V5Variant v) {
    switch (v) {
        case V5Variant::NPG_RTSS24:
            return "Srinivasan, Gunzel, Nelissen -- RTSS 2024 "
                   "(Limited-Preemptive Self-Suspending and Event-Driven "
                   "Delay-Induced Tasks)";
        case V5Variant::NP_UNI_17:
            return "Nasri, Brandenburg -- RTSS 2017 "
                   "(Exact and Sustainable Analysis of Non-Preemptive "
                   "Scheduling, uniprocessor)";
        case V5Variant::LP_DAG_19:
            return "Nasri, Nelissen, Brandenburg -- ECRTS 2019 "
                   "(Response-Time Analysis of Limited-Preemptive "
                   "Parallel DAG Tasks under Global Scheduling)";
        case V5Variant::GANG_22:
            return "Nelissen, Marce-i-Igual, Nasri -- ECRTS 2022 "
                   "(Response-Time Analysis for Non-Preemptive Periodic "
                   "Moldable Gang Tasks)";
    }
    return "?";
}

// Parse SAG_V5_VARIANT (case-insensitive; '-' and '_' interchangeable).
//
//   null/empty   -> NPG_RTSS24, *ok = true
//   known alias  -> matching variant, *ok = true
//   unknown      -> NPG_RTSS24, *ok = false (caller prints + exits)
//
// Aliases:
//   rtss24, rtss_24, npg_rtss24, np_global_24, ssed24, default
//   rtss17, rtss_17, np_uni_17, uni17, uni
//   ecrts19, ecrts_19, lp_dag_19, lpdag, lp
//   ecrts22, ecrts_22, gang_22, gang, moldable
inline V5Variant v5_parse_variant(const char* s, bool* ok) {
    *ok = true;
    if (!s || s[0] == '\0') return V5Variant::NPG_RTSS24;

    auto eq = [](const char* a, const char* b) -> bool {
        for (;; a++, b++) {
            unsigned char ca = (unsigned char)*a;
            unsigned char cb = (unsigned char)*b;
            if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca + 32);
            if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb + 32);
            if (ca == '-') ca = '_';
            if (cb == '-') cb = '_';
            if (ca != cb) return false;
            if (ca == '\0') return true;
        }
    };

    if (eq(s, "rtss24") || eq(s, "rtss_24") || eq(s, "npg_rtss24") ||
        eq(s, "np_global_24") || eq(s, "ssed24") || eq(s, "default")) {
        return V5Variant::NPG_RTSS24;
    }
    if (eq(s, "rtss17") || eq(s, "rtss_17") || eq(s, "np_uni_17") ||
        eq(s, "uni17") || eq(s, "uni")) {
        return V5Variant::NP_UNI_17;
    }
    if (eq(s, "ecrts19") || eq(s, "ecrts_19") || eq(s, "lp_dag_19") ||
        eq(s, "lpdag") || eq(s, "lp")) {
        return V5Variant::LP_DAG_19;
    }
    if (eq(s, "ecrts22") || eq(s, "ecrts_22") || eq(s, "gang_22") ||
        eq(s, "gang") || eq(s, "moldable")) {
        return V5Variant::GANG_22;
    }

    *ok = false;
    return V5Variant::NPG_RTSS24;
}

inline void v5_print_variant_choices(FILE* f) {
    std::fprintf(f,
        "Valid SAG_V5_VARIANT choices:\n"
        "  rtss24    (default) -- limited-preemptive self-suspending + event-driven\n"
        "                         (Srinivasan/Gunzel/Nelissen, RTSS 2024)\n"
        "  rtss17              -- exact uniprocessor non-preemptive\n"
        "                         (Nasri/Brandenburg, RTSS 2017)\n"
        "  ecrts19             -- limited-preemptive parallel DAG (global)\n"
        "                         (Nasri/Nelissen/Brandenburg, ECRTS 2019)\n"
        "  ecrts22             -- non-preemptive periodic moldable gang\n"
        "                         (Nelissen/Marce-i-Igual/Nasri, ECRTS 2022)\n"
        "Aliases (case-insensitive): rtss_24/npg_rtss24/np_global_24/default,\n"
        "  rtss_17/np_uni_17/uni17/uni, ecrts_19/lp_dag_19/lpdag/lp,\n"
        "  ecrts_22/gang_22/gang/moldable.\n");
}

// Per-variant tricks table. Hidden from users; controller picks based on
// variant + parsed-input shape. Rationale captured in framework/VARIANTS.md.
//
//   ijp_default     : NPG24 default; controller may override per-input.
//                     Other variants always force false (paper-incompatible
//                     or redundant after segment-expansion).
//   merge_level     : 0 = conservative c2 (sub-interval containment, exact);
//                     1 = lossy l1 (current V5 default).
//   f_max_per_state : 0 = use compile-time SAG_V5_F_MAX_PER_STATE; >0 overrides
//                     for variants whose state population is wider.
//   m_eq_one        : true => host code forces m=1 regardless of -m flag (UNI).
struct V5VariantTricks {
    bool ijp_default;
    int  merge_level;
    int  f_max_per_state;
    bool m_eq_one;
    const char* notes;
};

inline V5VariantTricks v5_default_tricks(V5Variant v) {
    switch (v) {
        case V5Variant::NPG_RTSS24:
            return { /*ijp_default=*/false, /*merge_level=*/1,
                     /*f_max=*/0, /*m_eq_one=*/false,
                     "lossy l1 merge; IJP off by default (override via SAG_V5_IJP=1)" };
        case V5Variant::NP_UNI_17:
            return { /*ijp_default=*/false, /*merge_level=*/0,
                     /*f_max=*/0, /*m_eq_one=*/true,
                     "exact analysis: conservative c2 merge; m forced to 1" };
        case V5Variant::LP_DAG_19:
            return { /*ijp_default=*/false, /*merge_level=*/1,
                     /*f_max=*/48, /*m_eq_one=*/false,
                     "lossy l1 merge; IJP off (segment expansion makes it redundant)" };
        case V5Variant::GANG_22:
            return { /*ijp_default=*/false, /*merge_level=*/1,
                     /*f_max=*/0, /*m_eq_one=*/false,
                     "lossy l1 merge with parallelism check; IJP off (asymmetric costs)" };
    }
    return { false, 1, 0, false, "" };
}

} // namespace v5
} // namespace sag
