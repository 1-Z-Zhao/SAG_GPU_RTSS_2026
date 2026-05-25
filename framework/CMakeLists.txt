# framework/CMakeLists.txt -- bootstrap build for the GPU-resident SAG
# solver introduced in iter300.
#
# Side-by-side with the legacy build:
#   - Legacy: cmake --build build -j16 produces build/expand_test
#   - V5:     cmake -S framework -B framework/build ...
#             cmake --build framework/build -j16
#             produces framework/build/expand_test_v5
#
# Generic across A/H/B-series NV GPUs: emit native SASS for sm_80 (Ampere/A100),
# sm_90 (Hopper/H100), and sm_100 (Blackwell/B200) when the toolchain supports
# them. Falls back gracefully to whatever subset the current CUDA version can
# build, with PTX for the highest emitted arch as a JIT-fallback for newer GPUs.
# The cooperative-launch + this_grid().sync() APIs used here are SM_70+.
# No Hopper-only features (TMA, wgmma, DSM/cluster-shared-mem, FP8) are used.

cmake_minimum_required(VERSION 3.18)
project(framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Multi-arch: probe what nvcc supports and assemble the strongest possible
# native SASS list. Industrial users get zero JIT cost on first run when the
# build host's CUDA toolkit can target their deployment GPU.
#
# CUDA 12.5 supports up to compute_90 (Hopper). compute_100 (Blackwell) is
# added in CUDA 12.8+. We probe each candidate via nvcc --list-gpu-arch and
# include only the supported ones. The final list is converted to CMake's
# CUDA_ARCHITECTURES form so that PTX for the highest arch is also emitted
# (CMake appends the virtual arch automatically when given e.g. "90", which
# is sufficient for forward-compat JIT to unsupported newer GPUs).
execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
    OUTPUT_VARIABLE _v5_nvcc_arch_list
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(_v5_arch_candidates "80" "90" "100")
set(_v5_arch_emitted "")
foreach(_arch IN LISTS _v5_arch_candidates)
    if(_v5_nvcc_arch_list MATCHES "compute_${_arch}")
        list(APPEND _v5_arch_emitted "${_arch}")
    endif()
endforeach()

# Safety net: if the probe somehow returns nothing (e.g. older CMake/nvcc
# combination without --list-gpu-arch), fall back to sm_80 alone.
if(NOT _v5_arch_emitted)
    set(_v5_arch_emitted "80")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${_v5_arch_emitted})
message(STATUS "framework: emitting native SASS for sm_${_v5_arch_emitted}")

find_package(CUDAToolkit REQUIRED)

# Pull in the legacy include/ directory (sag_config.h, sag_types.h,
# k1_body.cuh) plus our own framework/include/ (k2_body_v5.cuh).
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(expand_test_v5
    src/controller.cu
    src/framework_main.cu
)

target_link_libraries(expand_test_v5 PRIVATE CUDA::cudart)

target_compile_options(expand_test_v5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo --expt-extended-lambda --default-stream=per-thread>
)

# Cooperative launch requires relocatable device code so cg::this_grid() can
# resolve across TUs (controller.cu invokes K1Body inline; relocatable lets
# the linker fold the cross-TU __device__ references safely).
set_target_properties(expand_test_v5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
