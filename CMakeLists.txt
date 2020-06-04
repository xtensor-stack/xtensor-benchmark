############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

project(xtensor-benchmark)

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/modules/")
include(CheckCXXCompilerFlag)
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

# Optional dependencies
# =====================

option(BENCHMARK_XTENSOR "benchmark against xtensor" ON)
option(BENCHMARK_EIGEN "benchmark against eigen" OFF)
option(BENCHMARK_BLITZ "benchmark against Blitz" OFF)
option(BENCHMARK_ARMADILLO "benchmark agains Armadillo" OFF)
option(BENCHMARK_PYTHONIC "benchmark agains numpy + pythran" OFF)
option(BENCHMARK_ALL "benchmark against all libraries" OFF)

if(BENCHMARK_ALL)
    set(BENCHMARK_XTENSOR ON)
    set(BENCHMARK_EIGEN ON)
    set(BENCHMARK_BLITZ ON)
    set(BENCHMARK_ARMADILLO ON)
    set(BENCHMARK_PYTHONIC ON)
endif()

# Compilation flags
# =================

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
    if (NOT HAS_CPP14_FLAG)
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++14 support!")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -march=native -pthread")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /arch:AVX2")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

    find_package(benchmark REQUIRED)
    set(GBENCHMARK_LIBRARIES benchmark::benchmark_main)


# Sources and target
# ==================

set(XTENSOR_BENCHMARK_TARGET xtensor_benchmark)
set(XTENSOR_BENCHMARK
    src/benchmark_add_1d.hpp
    src/benchmark_add_2d.hpp
    src/benchmark_broadcasting.hpp
    src/benchmark_views.hpp
    src/benchmark_fixed.hpp
    src/benchmark_constructor.hpp
    src/benchmark_scalar_assignment.hpp
    src/benchmark_iterators.hpp
    src/main.cpp
)

add_executable(${XTENSOR_BENCHMARK_TARGET} ${XTENSOR_BENCHMARK} ${XTENSOR_HEADERS})

# Mandatory dependencies
# ======================
set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include")


find_package(xtensor REQUIRED CONFIG)
find_package(xsimd REQUIRED)
find_package(xtl REQUIRED)
find_package(Threads)

target_include_directories(${XTENSOR_BENCHMARK_TARGET} PRIVATE ${xtensor_INCLUDE_DIRS} ${xtl_INCLUDE_DIRS})
target_include_directories(${XTENSOR_BENCHMARK_TARGET} PRIVATE ${GBENCHMARK_INCLUDE_DIRS})
target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE XTENSOR_USE_XSIMD=1 NDEBUG=1)
set_target_properties(${XTENSOR_BENCHMARK_TARGET} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS NO)
target_link_libraries(${XTENSOR_BENCHMARK_TARGET}
                      ${GBENCHMARK_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT})

if(BENCHMARK_XTENSOR)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE HAS_XTENSOR=1)
endif()

if(BENCHMARK_EIGEN)
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE HAS_EIGEN=1 EIGEN_FAST_MATH=1)
    target_link_libraries(${XTENSOR_BENCHMARK_TARGET} Eigen3::Eigen)
endif()

if(BENCHMARK_BLITZ)
    find_package(Blitz REQUIRED)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE HAS_BLITZ=1)
    target_include_directories(${XTENSOR_BENCHMARK_TARGET} PRIVATE ${BLITZ_INCLUDES})
    target_link_libraries(${XTENSOR_BENCHMARK_TARGET} ${BLITZ_LIBRARIES})
endif()

if(BENCHMARK_ARMADILLO)
    find_package(Armadillo REQUIRED)
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE HAS_ARMADILLO=1)
    target_include_directories(${XTENSOR_BENCHMARK_TARGET} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(${XTENSOR_BENCHMARK_TARGET} ${ARMADILLO_LIBRARIES})
endif()

if(BENCHMARK_PYTHONIC)
    find_package(PythonLibs)
    find_package(NumPy)
    find_package(Pythran )
    link_directories(${Pythran_INCLUDE_DIRS})
    target_compile_definitions(${XTENSOR_BENCHMARK_TARGET} PRIVATE HAS_PYTHONIC=1 ENABLE_PYTHON_MODULE=1 USE_BOOST_SIMD=1)
    target_include_directories(${XTENSOR_BENCHMARK_TARGET} PRIVATE
                               ${PYTHON_INCLUDE_DIRS}
                               ${NUMPY_INCLUDE_DIRS}
                               ${Pythran_INCLUDE_DIRS})
    target_link_libraries(${XTENSOR_BENCHMARK_TARGET} ${PYTHON_LIBRARIES})
endif()


    message("\n\n          COMPILING WITH\n======================================\n\n")
    message("COMPILER        : ${CMAKE_CXX_COMPILER}")
    message("FLAGS           : ${CMAKE_CXX_FLAGS}\n")
if(BENCHMARK_XTENSOR)
    message("Found xtensor   : ${xtensor_INCLUDE_DIRS}")
    message("Found xsimd     : ${xsimd_INCLUDE_DIRS}\n\n")
endif()
if(BENCHMARK_EIGEN)
    message("Found eigen     : ${EIGEN3_INCLUDE_DIR}")
endif()
if(BENCHMARK_BLITZ)
    message("Found Blitz     : ${BLITZ_INCLUDES} | ${BLITZ_LIBRARIES}")
endif()
if(BENCHMARK_ARMADILLO)
    message("Found Armadillo : ${ARMADILLO_INCLUDE_DIRS} | ${ARMADILLO_LIBRARIES}")
endif()
if(BENCHMARK_PYTHONIC)
    message("Found Pythran   : ${Pythran_INCLUDE_DIRS}")
endif()
    message("Using benchmark : ${GBENCHMARK_INCLUDE_DIRS} | ${GBENCHMARK_LIBRARIES}")


if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/profile_snip.cpp)
  add_executable(profile_snip
      src/profile_snip.cpp
      ${XTENSOR_HEADERS})
endif()

add_custom_target(xbenchmark
    COMMAND xtensor_benchmark --benchmark_out=bench.csv --benchmark_out_format=csv
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})

add_custom_target(xpowerbench
    COMMAND echo "sudo needed to set cpu power governor to performance"
    COMMAND sudo cpupower frequency-set --governor performance
    COMMAND xtensor_benchmark --benchmark_out=bench.csv --benchmark_out_format=csv
    COMMAND sudo cpupower frequency-set --governor powersave
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})
