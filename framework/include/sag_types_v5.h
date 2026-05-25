// sag_types_v5.h -- Sparse F-pair state layout for framework.
//
// Replaces the dense `F_pair[n]` (n × int2 = n × 8 bytes) of legacy
// SAGStateLayout with a sparse encoding:
//
//   D[W] | X[W] | F_mask[W] | A_pair[m] | F_count(int32)
//        | F_entries[F_MAX_PER_STATE]   | ovf(int32)
//
// where F_entries[k] = struct { int32 job_idx; int32 f_min; int32 f_max }
// (12 bytes per active job).
//
// Why F_mask | X?
//   The K2 STEP 5 "X' = {j} ∪ {k ∈ par_X : k ∉ Pred(j) ∧ par_F_min[k] > s_max}"
//   reads par_F_min[k] for k in par_X. Bits can be in X but not in F_mask
//   (when a dispatched job's Succ ⊆ D' at dispatch time, it goes to X but
//   not F_mask, per K2 step 4 / step 5). To match legacy behavior, we must
//   preserve F values for the union of F_mask AND X. We therefore index
//   F_entries by `F_avail = F_mask | X`. Lookups walk F_avail's bits in
//   sorted order; entries are stored in ascending job_idx order.
//
// Sized at runtime:
//   bytes_per_state = 3*W*8 + 2*m*4 + 4 + F_MAX_PER_STATE*12 + 4
//
// F_MAX_PER_STATE is a build-time toggle (-DSAG_V5_F_MAX_PER_STATE=N) so we
// can set it generously without changing the layout descriptor. Default 96
// covers periodic n=140-class workloads where popcount(F_mask|X) is bounded
// by the number of jobs simultaneously "in execution" -- typically << n.
//
// On any state where popcount(F_mask|X) > F_MAX_PER_STATE, kernels MUST fail
// loudly (printf the popcount and the layer/state, set a sticky error flag,
// return). NEVER silently truncate.
//
// Generic across A/H/B-series NV GPUs (no Hopper-only intrinsics).

#pragma once
#include "sag_config.h"
#include <cstdint>
#include <type_traits>

// F_MAX_PER_STATE: build-time cap on per-state F_entries.
//
// Empirical measurements (max popcount(F_mask | X) across all V5 runs):
//   n=4 chain   :  3
//   n8/t000  N=40, m=2 : 5
//   n12/t000 N=60, m=3 : 10
//   n16/t000 N=80, m=4 : 15
//   n20/t000 N=100, m=5: 12
//   n24/t000 N=120, m=6: 18
//   n28/t000 N=140, m=7: 19   (u20)
//   n28/t000 N=140, m=7: 21   (u50)
//   n28/t000 N=140, m=7: 16   (u80)
//
// Default F_MAX_PER_STATE = 32 covers all observed cases up to N=140 with
// >=11 headroom, which is comfortable for higher-utilization variants of the
// same N. It gives bps_v5 / bps_dense = 0.41x on n=140 (dense=1256, sparse=520).
//
// On overflow, kernels FAIL LOUDLY (printf the violation, set d_trunc_flag,
// abort). NEVER silently truncate.
//
// Override via -DSAG_V5_F_MAX_PER_STATE=N if the workload requires more.
#ifndef SAG_V5_F_MAX_PER_STATE
#define SAG_V5_F_MAX_PER_STATE 32
#endif

namespace sag {
namespace v5 {

constexpr int F_MAX_PER_STATE = SAG_V5_F_MAX_PER_STATE;

// One sparse F-pair entry.  Tightly packed; total 12 bytes (no padding when
// laid out in a flat int32-aligned arena, which we guarantee via the layout
// descriptor's bytes_per_state being 8-byte aligned and F_count being a
// 4-byte int32 immediately preceding the entry array).
// FEntryV5 layout (12 bytes total):
//   - job_idx: low 16 bits = job index (n <= 512); upper 16 bits = p_committed
//     (committed gang parallelism for this dispatch; ECRTS 2022 only, 0 = unset
//     interpreted as p=1 by readers that ignore parallelism).
//   - f_min, f_max: as before.
//
// Helper macros below (V5_FE_JOB_IDX, V5_FE_P_COMMITTED, V5_FE_PACK_JOB_P)
// abstract the bit packing so non-gang variants read job_idx exactly as
// before (mask with V5_FE_JOB_MASK). Default-constructed entries have
// p_committed == 0; ECRTS 2022's K2 commits set it to the chosen p in
// [1..m]; ECRTS 2022's merge predicate compares them on common F_mask
// jobs to enforce the paper's parallelism-disagree invariant.
// Job indices fit in 16 bits (n <= 512 = 2^9), p_committed in 16 bits
// (m <= 64 = 2^6). Both have ample headroom. Bit layout shared between
// host and device.
#define V5_FE_JOB_MASK    0xFFFF
#define V5_FE_P_SHIFT     16

struct FEntryV5 {
    int32_t job_idx;  // low 16 bits = job index; high 16 bits = p_committed
    int32_t f_min;
    int32_t f_max;

    // Read job index, masking out p_committed in the upper bits. All
    // existing call sites that did `entry.job_idx` for comparisons,
    // arithmetic, or array indexing should NOT be modified -- they
    // would break when ECRTS 2022's K2 packs p into the upper bits.
    // Use this accessor instead.
    __host__ __device__ __forceinline__ int32_t job() const {
        return (int32_t)((uint32_t)job_idx & V5_FE_JOB_MASK);
    }
    // Read committed parallelism (0 if unset; ECRTS 2022's K2 packs >=1).
    __host__ __device__ __forceinline__ int32_t p_committed() const {
        return (int32_t)(((uint32_t)job_idx >> V5_FE_P_SHIFT));
    }
    __host__ __device__ __forceinline__ void set_job_p(int32_t j, int32_t p) {
        job_idx = (int32_t)(((uint32_t)j & V5_FE_JOB_MASK)
                            | (((uint32_t)p) << V5_FE_P_SHIFT));
    }
};
static_assert(sizeof(FEntryV5) == 12, "FEntryV5 must be exactly 12 bytes");

#define V5_FE_JOB_IDX(e)        ((e).job())
#define V5_FE_P_COMMITTED(e)    ((e).p_committed())
#define V5_FE_PACK_JOB_P(j, p)  ((int32_t)(((uint32_t)(j) & V5_FE_JOB_MASK) | (((uint32_t)(p)) << V5_FE_P_SHIFT)))

// ---------------------------------------------------------------------------
// SAGStateLayoutV5 -- sparse F-pair layout descriptor.
//
// Each state is laid out as:
//   uint64_t D[W];                                                    (8W)
//   uint64_t X[W];                                                    (8W)
//   uint64_t F_mask[W];                                               (8W)
//   int2     A_pair[m];                                              (8m)
//   int32_t  F_count;                                                 (4)
//   FEntryV5 F_entries[F_MAX_PER_STATE];                            (12*Fmax)
//   int32_t  ovf;                                                    (4)
//
// Total raw = 24*W + 8*m + 4 + 12*F_MAX_PER_STATE + 4
// Padded up to next multiple of 8 so D[] of next state is uint64_t aligned.
// ---------------------------------------------------------------------------
struct SAGStateLayoutV5 {
    int W;
    int n;
    int m;

    __host__ __device__ int bytes_per_state() const {
        int raw = 3 * W * (int)sizeof(uint64_t)
                + m * (int)sizeof(int2)
                + (int)sizeof(int32_t)
                + F_MAX_PER_STATE * (int)sizeof(FEntryV5)
                + (int)sizeof(int32_t);
        return (raw + 7) & ~7;
    }

    __host__ __device__ uint64_t* D(char* base, int idx) const {
        return (uint64_t*)(base + (long long)idx * bytes_per_state());
    }
    __host__ __device__ const uint64_t* D(const char* base, int idx) const {
        return (const uint64_t*)(base + (long long)idx * bytes_per_state());
    }

    __host__ __device__ uint64_t* X(char* base, int idx) const {
        return (uint64_t*)(base + (long long)idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* X(const char* base, int idx) const {
        return (const uint64_t*)(base + (long long)idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }

    __host__ __device__ uint64_t* F_mask(char* base, int idx) const {
        return (uint64_t*)(base + (long long)idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* F_mask(const char* base, int idx) const {
        return (const uint64_t*)(base + (long long)idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }

    // A_pair[m]: int2{x = A_min[i], y = A_max[i]}
    __host__ __device__ int2* A_pair(char* base, int idx) const {
        return (int2*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }
    __host__ __device__ const int2* A_pair(const char* base, int idx) const {
        return (const int2*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }

    // F_count: int32 just after A_pair.
    __host__ __device__ int32_t* F_count(char* base, int idx) const {
        return (int32_t*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2));
    }
    __host__ __device__ const int32_t* F_count(const char* base, int idx) const {
        return (const int32_t*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2));
    }

    // F_entries: FEntryV5 array of length F_MAX_PER_STATE; only the first
    // *F_count entries are valid.
    __host__ __device__ FEntryV5* F_entries(char* base, int idx) const {
        return (FEntryV5*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2)
               + sizeof(int32_t));
    }
    __host__ __device__ const FEntryV5* F_entries(const char* base, int idx) const {
        return (const FEntryV5*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2)
               + sizeof(int32_t));
    }

    __host__ __device__ int32_t* ovf(char* base, int idx) const {
        return (int32_t*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2)
               + sizeof(int32_t)
               + F_MAX_PER_STATE * (int)sizeof(FEntryV5));
    }
    __host__ __device__ const int32_t* ovf(const char* base, int idx) const {
        return (const int32_t*)(base + (long long)idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t)
               + m * sizeof(int2)
               + sizeof(int32_t)
               + F_MAX_PER_STATE * (int)sizeof(FEntryV5));
    }
};

// ---------------------------------------------------------------------------
// Sparse F-entry lookup: given a state's F_mask|X bitmask and its
// F_entries / F_count, find the entry for job j.
//
// Entries are sorted ascending by job_idx (matching the order bits appear in
// the F_avail bitmask when walked low-to-high). Lookup is a linear scan
// (entries are <= F_MAX_PER_STATE which is moderate).
//
// Returns true and writes (*out_fmin, *out_fmax) iff j is in F_avail (i.e.
// has an entry). Returns false otherwise; in this case caller should treat
// (f_min, f_max) = (0, 0) -- consistent with the dense layout's initial
// zeros in legacy expand_test_main.cu:2310.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ bool sparse_F_lookup(
    const FEntryV5* entries, int F_count, int j,
    int32_t* out_fmin, int32_t* out_fmax)
{
    for (int e = 0; e < F_count; e++) {
        if (entries[e].job() == j) {
            *out_fmin = entries[e].f_min;
            *out_fmax = entries[e].f_max;
            return true;
        }
        if (entries[e].job() > j) break;  // sorted; not present
    }
    *out_fmin = 0;
    *out_fmax = 0;
    return false;
}

// f_min-only variant (K2 step 5 hot path).
__host__ __device__ __forceinline__ int32_t sparse_F_min(
    const FEntryV5* entries, int F_count, int j)
{
    for (int e = 0; e < F_count; e++) {
        if (entries[e].job() == j) return entries[e].f_min;
        if (entries[e].job() > j) return 0;
    }
    return 0;
}

// f_max-only variant (K1 hot path).
__host__ __device__ __forceinline__ int32_t sparse_F_max(
    const FEntryV5* entries, int F_count, int j)
{
    for (int e = 0; e < F_count; e++) {
        if (entries[e].job() == j) return entries[e].f_max;
        if (entries[e].job() > j) return 0;
    }
    return 0;
}

} // namespace v5
} // namespace sag
