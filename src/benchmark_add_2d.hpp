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
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#endif

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

#ifdef HAS_BLITZ
#include <blitz/array.h>
#endif

#ifdef HAS_ARMADILLO
#include <armadillo>
#endif

#ifdef HAS_PYTHONIC
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/numpy/random/rand.hpp>
#endif

#ifdef HAS_VIGRA
#include "vigra/multi_array.hxx"
#include "vigra/multi_math.hxx"
#endif

#define RANGE 16, 1024
#define MULTIPLIER 8


#ifdef HAS_XTENSOR
void Add2D_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> a = random::rand<double>({state.range(0), state.range(0)});
    xtensor<double, 2> b = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 2> res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add2D_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_EIGEN
void Add2D_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    MatrixXd a = MatrixXd::Random(state.range(0), state.range(0));
    MatrixXd b = MatrixXd::Random(state.range(0), state.range(0));
    for (auto _ : state)
    {
        MatrixXd res(state.range(0), state.range(0));
        res.noalias() = a + b;
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add2D_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_BLITZ
void Add2D_Blitz(benchmark::State& state)
{
    using namespace blitz;
    Array<double, 2> a(state.range(0), state.range(0));
    Array<double, 2> b(state.range(0), state.range(0));
    for (auto _ : state)
    {
        Array<double, 2> res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add2D_Blitz)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_ARMADILLO
void Add2D_Arma(benchmark::State& state)
{
    using namespace arma;
    mat a = randu<mat>(state.range(0), state.range(0));
    mat b = randu<mat>(state.range(0), state.range(0));
    for (auto _ : state)
    {
        mat res = a + b;
        benchmark::DoNotOptimize(res.memptr());
    }
}
BENCHMARK(Add2D_Arma)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_PYTHONIC
void Add2D_Pythonic(benchmark::State& state)
{
    auto x = pythonic::numpy::random::rand(state.range(0), state.range(0));
    auto y = pythonic::numpy::random::rand(state.range(0), state.range(0));

    for (auto _ : state)
    {
        pythonic::types::ndarray<double, 2> z = x + y;
        benchmark::DoNotOptimize(z.fbegin());
    }
}
BENCHMARK(Add2D_Pythonic)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_VIGRA
void Add2D_Vigra(benchmark::State& state)
{
    using namespace vigra::multi_math;
    vigra::MultiArray<2, double> vA(state.range(0), state.range(0));
    vigra::MultiArray<2, double> vB(state.range(0), state.range(0));
    for (auto _ : state)
    {
        vigra::MultiArray<2, double> vRes(state.range(0), state.range(0));
        vRes = vA + vB;
        benchmark::DoNotOptimize(vRes.data());
    }
}
BENCHMARK(Add2D_Vigra)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER

