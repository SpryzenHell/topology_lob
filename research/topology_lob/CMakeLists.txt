cmake_minimum_required(VERSION 3.16)
project(topology_lob_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fractional_diff_cpu cpp/fractional_diff.cpp)
target_compile_options(fractional_diff_cpu PRIVATE -O3 -march=native -DNDEBUG)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(fractional_diff_cuda cuda/fractional_diff.cu)
    set_target_properties(
        fractional_diff_cuda
        PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    )
    target_compile_options(
        fractional_diff_cuda
        PRIVATE $<BUILD_INTERFACE:-O3>
    )
endif()