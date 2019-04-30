/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#ifdef HAS_XTENSOR
#include "xtensor/xnoalias.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#endif

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

#ifdef HAS_XTENSOR
template <std::size_t N, std::size_t M>
void Add2dFixed_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor_fixed<double, xshape<N, M>> a = random::rand<double>({N, M});
    xtensor_fixed<double, xshape<N, M>> b = random::rand<double>({N, M});

    for (auto _ : state)
    {
        xtensor_fixed<double, xshape<N, M>> res;
        xt::noalias(res) = a + b;
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK_TEMPLATE(Add2dFixed_XTensor, 3, 3);
BENCHMARK_TEMPLATE(Add2dFixed_XTensor, 8, 8);
BENCHMARK_TEMPLATE(Add2dFixed_XTensor, 64, 64);
BENCHMARK_TEMPLATE(Add2dFixed_XTensor, 512, 512);
#endif

#ifdef HAS_EIGEN
template <std::size_t N, std::size_t M>
void Add2dFixed_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    Matrix<double, N, M> a = Matrix<double, N, N>::Random(N, M);
    Matrix<double, N, M> b = Matrix<double, N, N>::Random(N, M);

    for (auto _ : state)
    {
        Matrix<double, N, M> res;
        res.noalias() = a + b;
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK_TEMPLATE(Add2dFixed_Eigen, 3, 3);
BENCHMARK_TEMPLATE(Add2dFixed_Eigen, 8, 8);
BENCHMARK_TEMPLATE(Add2dFixed_Eigen, 64, 64);
#endif