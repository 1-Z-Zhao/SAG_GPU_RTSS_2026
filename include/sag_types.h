#pragma once
#include "sag_config.h"
#include <cstdint>

namespace sag {

// Compute the number of uint64_t words needed for n bits
inline __host__ __device__ int bitset_words(int n) {
    return (n + 63) / 64;
}

// Layout descriptor for dynamically-sized SAG states in flat memory.
//
// Each state occupies a contiguous block of bytes:
//   D[W]  | X[W]  | F_mask[W]  | A_min[m] | A_max[m] | F_min[n] | F_max[n] | ovf (int32)
//
// where W = bitset_words(n) = ceil(n/64).
struct SAGStateLayout {
    int W;  // bitset words = ceil(n/64)
    int n;  // number of jobs
    int m;  // number of cores

    __host__ __device__ int bytes_per_state() const {
        int raw = 3 * W * (int)sizeof(uint64_t)  // D, X, F_mask
                + 2 * m * (int)sizeof(int32_t)    // A_min, A_max
                + 2 * n * (int)sizeof(int32_t)    // F_min, F_max
                + (int)sizeof(int32_t);           // ovf
        // Round up to 8-byte alignment so uint64_t D[] of the next state is aligned
        return (raw + 7) & ~7;
    }

    __host__ __device__ uint64_t* D(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state());
    }
    __host__ __device__ const uint64_t* D(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state());
    }

    __host__ __device__ uint64_t* X(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* X(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + W * sizeof(uint64_t));
    }

    __host__ __device__ uint64_t* F_mask(char* base, int state_idx) const {
        return (uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }
    __host__ __device__ const uint64_t* F_mask(const char* base, int state_idx) const {
        return (const uint64_t*)(base + (long long)state_idx * bytes_per_state()
               + 2 * W * sizeof(uint64_t));
    }

    __host__ __device__ int32_t* A_min(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }
    __host__ __device__ const int32_t* A_min(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t));
    }

    __host__ __device__ int32_t* A_max(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + m * sizeof(int32_t));
    }
    __host__ __device__ const int32_t* A_max(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + m * sizeof(int32_t));
    }

    __host__ __device__ int32_t* F_min(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t));
    }
    __host__ __device__ const int32_t* F_min(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t));
    }

    __host__ __device__ int32_t* F_max(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t) + n * sizeof(int32_t));
    }
    __host__ __device__ const int32_t* F_max(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t) + n * sizeof(int32_t));
    }

    __host__ __device__ int32_t* ovf(char* base, int state_idx) const {
        return (int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t) + 2 * n * sizeof(int32_t));
    }
    __host__ __device__ const int32_t* ovf(const char* base, int state_idx) const {
        return (const int32_t*)(base + (long long)state_idx * bytes_per_state()
               + 3 * W * sizeof(uint64_t) + 2 * m * sizeof(int32_t) + 2 * n * sizeof(int32_t));
    }
};

struct ValidPair {
    int     state_idx;
    int     job_j;
    int32_t s_min;
    int32_t s_max;
};

} // namespace sag
