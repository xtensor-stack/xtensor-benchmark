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
#ifndef EIGEN_VECTORIZE
static_assert(false, "NOT VECTORIZING");
#endif
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

#define RANGE 16, 128*128
#define MULTIPLIER 8


#ifdef HAS_XTENSOR
void Add1D_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 1> a = random::rand<double>({state.range(0)});
    xtensor<double, 1> b = random::rand<double>({state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 1> res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add1D_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_EIGEN
void Add1D_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    VectorXd a = VectorXd::Random(state.range(0));
    VectorXd b = VectorXd::Random(state.range(0));
    for (auto _ : state)
    {
        VectorXd res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add1D_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_BLITZ
void Add1D_Blitz(benchmark::State& state)
{
    using namespace blitz;
    Array<double, 1> a(state.range(0));
    Array<double, 1> b(state.range(0));
    for (auto _ : state)
    {
        Array<double, 1> res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add1D_Blitz)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_VIGRA
void Add1D_Vigra(benchmark::State& state)
{
    using namespace vigra::multi_math;
    vigra::MultiArray<1, double> vA(state.range(0));
    vigra::MultiArray<1, double> vB(state.range(0));
    for (auto _ : state)
    {
        vigra::MultiArray<1, double> vRes(state.range(0));
        vRes = vA + vB;
        benchmark::DoNotOptimize(vRes.data());
    }
}
BENCHMARK(Add1D_Vigra)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_ARMADILLO
void Add1D_Arma(benchmark::State& state)
{
    using namespace arma;
    vec a = randu<vec>(state.range(0));
    vec b = randu<vec>(state.range(0));
    for (auto _ : state)
    {
        vec res(a + b);
        benchmark::DoNotOptimize(res.memptr());
    }
}
BENCHMARK(Add1D_Arma)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_PYTHONIC
void Add1D_Pythonic(benchmark::State &state)
{
    auto x = pythonic::numpy::random::rand(state.range(0));
    auto y = pythonic::numpy::random::rand(state.range(0));

    for (auto _ : state)
    {
        pythonic::types::ndarray<double, 1> z(x + y);
        benchmark::DoNotOptimize(z.fbegin());
    }
}
BENCHMARK(Add1D_Pythonic)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER

