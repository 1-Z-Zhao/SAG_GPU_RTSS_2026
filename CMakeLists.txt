cmake_minimum_required(VERSION 3.18)
project(expand_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(expand_test
    src/expand_test_main.cu
    src/fused_eligibility.cu
    src/successor_creation.cu
    src/merge_kernels.cu
)

target_link_libraries(expand_test CUDA::cudart OpenMP::OpenMP_CXX)

target_compile_options(expand_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo --expt-extended-lambda --default-stream=per-thread -Xcompiler=-fopenmp>
)

set_target_properties(expand_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
