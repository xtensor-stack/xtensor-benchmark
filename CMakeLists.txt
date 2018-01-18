############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

project(xtensor-benchmark)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/modules/")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -ffast-math -march=native -std=c++14 -pthread")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++14")

find_package(xtensor REQUIRED CONFIG)
find_package(xsimd REQUIRED)

find_package(xtl REQUIRED)
find_package(Threads)
find_package(Eigen3)
find_package(Blitz)
find_package(Armadillo)
find_package(Pythonic)

include_directories(${EIGEN3_INCLUDE_DIR} ${ARMADILLO_INCLUDE_DIRS} ${BLITZ_INCLUDES})
include_directories(${xtl_INCLUDE_DIRS} ${xsimd_INCLUDE_DIRS} ${xtensor_INCLUDE_DIRS})
include_directories(${Pythonic_INCLUDE_DIRS})


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
include(CheckCXXCompilerFlag)

message("\n\n          COMPILING WITH\n======================================\n\n")
message("COMPILER        : ${CMAKE_CXX_COMPILER}")
message("FLAGS           : ${CMAKE_CXX_FLAGS}\n")
message("Found eigen     : ${EIGEN3_INCLUDE_DIR}")
message("Found Blitz     : ${BLITZ_INCLUDES} | ${BLITZ_LIBRARIES}")
message("Found Armadillo : ${ARMADILLO_INCLUDE_DIRS} | ${ARMADILLO_LIBRARIES}")

message("Found xtensor   : ${xtensor_INCLUDE_DIRS}")
message("Found xsimd     : ${xsimd_INCLUDE_DIRS}\n\n")


message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

configure_file(modules/downloadGBenchmark.cmake.in googlebenchmark-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-download )

# Add googlebenchmark directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark-build)

set(GBENCHMARK_INCLUDE_DIRS "${googlebenchmark_SOURCE_DIR}/include")
set(GBENCHMARK_LIBRARIES benchmark)


add_definitions(-DXTENSOR_USE_XSIMD)
add_definitions(-DEIGEN_FAST_MATH)
add_definitions(-DNDEBUG)
add_definitions(-DUSE_BOOST_SIMD)

include_directories(${XTENSOR_INCLUDE_DIR} ${GBENCHMARK_INCLUDE_DIRS})

set(XTENSOR_BENCHMARK
    src/benchmark_1D.hpp
    src/benchmark_2D.hpp
    src/benchmark_views.hpp
    src/benchmark_allocators.hpp
    src/main.cpp
)

set(XTENSOR_BENCHMARK_TARGET xtensor_benchmark)
add_executable(${XTENSOR_BENCHMARK_TARGET} ${XTENSOR_BENCHMARK} ${XTENSOR_HEADERS})
target_link_libraries(${XTENSOR_BENCHMARK_TARGET}
                      ${GBENCHMARK_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT}
                      ${BLITZ_LIBRARIES}
                      ${ARMADILLO_LIBRARIES})

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/profile_snip.cpp)
  add_executable(profile_snip
      src/profile_snip.cpp
      ${XTENSOR_HEADERS})
endif()

add_custom_target(xbenchmark
    COMMAND xtensor_benchmark
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})

add_custom_target(xpowerbench
    COMMAND echo "sudo needed to set cpu power governor to performance"
    COMMAND sudo cpupower frequency-set --governor performance
    COMMAND xtensor_benchmark
    COMMAND sudo cpupower frequency-set --governor powersave
    DEPENDS ${XTENSOR_BENCHMARK_TARGET})