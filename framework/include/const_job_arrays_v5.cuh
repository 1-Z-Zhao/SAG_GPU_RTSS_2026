// const_job_arrays_v5.cuh -- forward declarations of V5's constant-memory
// per-job arrays.
//
// The actual __constant__ storage is defined exactly once in controller.cu;
// other TUs/headers see only these extern declarations. K2 and IJP detect
// read these inside __device__ functions; constant memory's broadcast cache
// beats L1 when all warp lanes share the same job index j.
//
// Read-only arrays moved from global memory:
//   c_C_min   [V5_MAX_CONST_N]
//   c_C_max   [V5_MAX_CONST_N]
//   c_deadline[V5_MAX_CONST_N]
//   c_r_min   [V5_MAX_CONST_N]
//   c_priority[V5_MAX_CONST_N]
//
// Caller (host orchestration in framework_main.cu) must invoke
// V5_upload_const_job_arrays(...) once after parsing the job set; it returns
// false if N > V5_MAX_CONST_N (in which case the host should fail loud).
//
// Generic across A/H/B-series NV GPUs (constant memory available on SM_70+).

#pragma once
#include <cstdint>

#ifndef V5_MAX_CONST_N
#define V5_MAX_CONST_N 256
#endif

extern __constant__ int32_t c_C_min   [V5_MAX_CONST_N];
extern __constant__ int32_t c_C_max   [V5_MAX_CONST_N];
extern __constant__ int32_t c_deadline[V5_MAX_CONST_N];
extern __constant__ int32_t c_r_min   [V5_MAX_CONST_N];
extern __constant__ int32_t c_priority[V5_MAX_CONST_N];

extern "C" bool V5_upload_const_job_arrays(
    const int32_t* h_C_min, const int32_t* h_C_max,
    const int32_t* h_deadline, const int32_t* h_r_min,
    const int32_t* h_priority, int N);

// Phase 3 (ECRTS 2022) per-job gang parallelism is plumbed through the kernel
// signature as a global-memory `d_p_max` argument (same pattern as d_C_min /
// d_C_max). Non-gang variants pass an all-1 array (or skip altogether by not
// using the ECRTS22 kernel). See V5K1K2KernelECRTS22 in controller.cu.
