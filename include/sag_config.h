#pragma once
#include <cstdint>
#include <climits>

namespace sag {
namespace config {

constexpr int BITS_PER_WORD    = 64;
constexpr int MAX_BITSET_WORDS = 8;   // Supports up to 512 jobs
constexpr int MAX_CORES  = 64;        // compile-time max (for stack arrays)
// Runtime m passed via SAGStateLayout.m and kernel parameters
constexpr int32_t INF_TIME = INT32_MAX / 2;

// Fused eligibility kernel
constexpr int STATES_PER_BLOCK   = 4;
constexpr int WARP_SIZE          = 32;
constexpr int FUSED_BLOCK_SIZE   = STATES_PER_BLOCK * WARP_SIZE;  // 128

// Successor creation kernel (legacy thread-per-pair)
constexpr int SUCCESSOR_BLOCK_SIZE = 256;

// Warp-cooperative successor kernel
constexpr int K2_WARPS_PER_BLOCK = 12;   // Iter6: 8 -> 12 warps (384 threads) to raise K2 SM occupancy
constexpr int K2_BLOCK_SIZE = K2_WARPS_PER_BLOCK * WARP_SIZE;  // 384

// Merge pipeline
constexpr int MAX_SLOTS_PER_GROUP = 64;   // max merge slots per D-group (increased for lmax-like merge)
constexpr int CPU_MERGE_THRESHOLD = 4;    // groups with size <= this go to CPU
constexpr int MERGE_BLOCK_SIZE = 256;     // threads per block for merge kernels
constexpr int MERGE_WARPS_PER_BLOCK = 4;  // warps per block for warp-cooperative merge

} // namespace config
} // namespace sag
