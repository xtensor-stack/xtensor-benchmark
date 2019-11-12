/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <iostream>

#include <benchmark/benchmark.h>

#include "benchmark_add_1d.hpp"
#include "benchmark_add_2d.hpp"
#include "benchmark_views.hpp"
#include "benchmark_broadcasting.hpp"
#include "benchmark_fixed.hpp"
#include "benchmark_constructor.hpp"
#include "benchmark_scalar_assignment.hpp"
#include "benchmark_iterators.hpp"
#include "benchmark_lazy_evaluation.hpp"
#include "benchmark_padding.hpp"


#ifdef XTENSOR_USE_XSIMD
#ifdef __GNUC__
template <class T>
void print_type(T&& /*t*/)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}
#endif
void print_stats()
{
    std::cout << "USING XSIMD\nSIMD SIZE: " << xsimd::simd_traits<double>::size << "\n\n";
#ifdef __GNUC__
    print_type(xt::xarray<double>());
    print_type(xt::xtensor<double, 2>());
#endif
}
#else
void print_stats()
{
    std::cout << "NOT USING XSIMD\n\n";
};
#endif

// Custom main function to print SIMD config
int main(int argc, char** argv)
{
    print_stats();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    
    return 0;
}

