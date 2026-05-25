// fused_eligibility.cu -- Fused eligibility kernel (Phase 1 + Phase 2)
//
// Grid:  ceil(num_states / STATES_PER_BLOCK) blocks
// Block: dim3(32, STATES_PER_BLOCK) = 128 threads
// Each warp (32 threads) processes ONE state.
// Within each warp, threads iterate over candidates with stride 32.
//
// Multi-word bitmask version: supports arbitrary job counts via W-word bitsets.

#include "sag_config.h"
#include "sag_types.h"

using namespace sag;
using namespace sag::config;

// ---------------------------------------------------------------------------
// Multi-word bitmask helpers
// ---------------------------------------------------------------------------

// Count trailing zeros of a 64-bit value; returns -1 if v == 0
__device__ __forceinline__ int ctz64(uint64_t v) {
    if (v == 0) return -1;
    return __ffsll(v) - 1;
}

// Test if bit b is set in a W-word bitset
__device__ __forceinline__ bool bit_test(const uint64_t* bitset, int b) {
    return (bitset[b / 64] >> (b % 64)) & 1ULL;
}

// Test if mask is a subset of set (i.e., (mask & ~set) == 0) over W words
__device__ __forceinline__ bool is_subset(const uint64_t* mask,
                                          const uint64_t* set, int W) {
    for (int w = 0; w < W; w++) {
        if (mask[w] & ~set[w]) return false;
    }
    return true;
}

__device__ __forceinline__ int32_t warp_min(int32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = min(val, other);
    }
    return val;
}

__global__ void FusedEligibilityKernel(
    const char*     d_input,        // flat state memory
    SAGStateLayout  layout,         // layout descriptor
    const int*      d_candidates,
    int num_states,
    int num_candidates,
    const uint64_t* d_TC,           // n*W uint64_t values (row j at d_TC[j*W])
    const uint64_t* d_PO,           // same
    const uint64_t* d_Pred,         // same
    const int32_t*  d_r_min,
    const int32_t*  d_r_max,
    const int32_t*  d_priority,
    const int32_t*  d_sus_min,
    const int32_t*  d_sus_max,
    int n, int W,
    ValidPair*      d_valid_pairs,
    int*            d_valid_count,
    int*            d_unschedulable_flag,
    int max_valid_pairs,
    int*            d_trunc_flag)
{
    int lane     = threadIdx.x;                          // 0..31
    int warpId   = threadIdx.y;                          // 0..STATES_PER_BLOCK-1
    int stateIdx = blockIdx.x * STATES_PER_BLOCK + warpId;

    if (stateIdx >= num_states) return;

    // Access state fields via layout
    const uint64_t* st_D      = layout.D(d_input, stateIdx);
    const uint64_t* st_F_mask = layout.F_mask(d_input, stateIdx);
    const int32_t*  st_A_min  = layout.A_min(d_input, stateIdx);
    const int32_t*  st_A_max  = layout.A_max(d_input, stateIdx);
    const int32_t*  st_F_min  = layout.F_min(d_input, stateIdx);
    const int32_t*  st_F_max  = layout.F_max(d_input, stateIdx);

    // ---- Dynamic shared memory for ready-set rho_max values ----
    // Layout: s_rho_max[STATES_PER_BLOCK * n] then s_in_ready[STATES_PER_BLOCK * n]
    extern __shared__ char smem[];
    int32_t* s_rho_max  = (int32_t*)smem;
    bool*    s_in_ready = (bool*)(smem + STATES_PER_BLOCK * n * sizeof(int32_t));

    for (int j = lane; j < n; j += WARP_SIZE) {
        s_rho_max[warpId * n + j]  = INF_TIME;
        s_in_ready[warpId * n + j] = false;
    }
    __syncwarp(0xFFFFFFFF);

    // --- Sub-Phase 1: Build ready set R and compute rho_max for each member ---
    // R = { c not in D | Pred[c] subset of D }
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int c = d_candidates[ci];

        if (bit_test(st_D, c)) continue;          // already dispatched

        const uint64_t* preds = &d_Pred[c * W];
        if (!is_subset(preds, st_D, W)) continue;  // not all preds dispatched

        // rho_max(c) = max(r_max_c, max_{i in Pred(c)} {F_max[i] + sus_max[c*n+i]})
        int32_t rho_max_c = d_r_max[c];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds[w];
            while (pbits) {
                int bit = ctz64(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                if (bit_test(st_F_mask, i)) {
                    rho_max_c = max(rho_max_c, st_F_max[i] + d_sus_max[c * n + i]);
                }
            }
        }
        s_rho_max[warpId * n + c]  = rho_max_c;
        s_in_ready[warpId * n + c] = true;
    }
    __syncwarp(0xFFFFFFFF);

    // rho_any = min rho_max over all jobs in R
    int32_t local_rho_any = INF_TIME;
    for (int j = lane; j < n; j += WARP_SIZE) {
        if (s_in_ready[warpId * n + j])
            local_rho_any = min(local_rho_any, s_rho_max[warpId * n + j]);
    }
    int32_t rho_any = warp_min(local_rho_any);

    // --- Sub-Phase 2: Evaluate eligibility for each candidate ---
    for (int ci = lane; ci < num_candidates; ci += WARP_SIZE) {
        int j = d_candidates[ci];

        if (bit_test(st_D, j)) continue;                          // dispatched

        const uint64_t* tc_j = &d_TC[j * W];
        if (!is_subset(tc_j, st_D, W)) continue;                  // TC guard

        const uint64_t* po_j = &d_PO[j * W];
        if (!is_subset(po_j, st_D, W)) continue;                  // PO guard

        if (!s_in_ready[warpId * n + j]) continue;                // not ready

        // rho_min(j) = max(r_min_j, max_{i in Pred(j)} {F_min[i] + sus_min[j*n+i]})
        int32_t rho_min_j = d_r_min[j];
        const uint64_t* preds_j = &d_Pred[j * W];
        for (int w = 0; w < W; w++) {
            uint64_t pbits = preds_j[w];
            while (pbits) {
                int bit = ctz64(pbits);
                pbits &= pbits - 1;
                int i = w * 64 + bit;
                if (bit_test(st_F_mask, i))
                    rho_min_j = max(rho_min_j, st_F_min[i] + d_sus_min[j * n + i]);
            }
        }

        // rho_hp(j) = min{ rho_max_h | h in R, h != j, priority[h] < priority[j] }
        int32_t prio_j = d_priority[j];
        int32_t rho_hp_j = INF_TIME;
        for (int h = 0; h < n; h++) {
            if (!s_in_ready[warpId * n + h]) continue;
            if (h == j) continue;
            if (d_priority[h] < prio_j)
                rho_hp_j = min(rho_hp_j, s_rho_max[warpId * n + h]);
        }

        // s_min(j) = max(rho_min_j, A_min[0])                        (Eq. 9)
        int32_t s_min_j = max(rho_min_j, st_A_min[0]);

        // s_max(j) = min(rho_hp_j - 1, max(A_max[0], rho_any))       (Eq. 10)
        int32_t s_max_j = min(rho_hp_j - 1, max(st_A_max[0], rho_any));

        if (s_min_j <= s_max_j) {
            int idx = atomicAdd(d_valid_count, 1);
            if (idx < max_valid_pairs) {
                d_valid_pairs[idx].state_idx = stateIdx;
                d_valid_pairs[idx].job_j     = j;
                d_valid_pairs[idx].s_min     = s_min_j;
                d_valid_pairs[idx].s_max     = s_max_j;
            } else {
                // Overflow: record truncation so the host can abort
                atomicExch(d_trunc_flag, 1);
            }
        }
    }
}
