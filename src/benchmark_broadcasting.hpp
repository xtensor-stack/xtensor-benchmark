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

#ifdef HAS_PYTHONIC
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/numpy/random/rand.hpp>
#endif

#define SZ 100
#define RANGE 3, 1000
#define MULTIPLIER 8

#ifdef HAS_XTENSOR
void Add3d2dBroadcasting_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 3> a = random::rand<double>({state.range(0), state.range(0), state.range(0)});
    xtensor<double, 2> b = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 3> res(a + b);
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(Add3d2dBroadcasting_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_PYTHONIC
void pythonic_broadcasting(benchmark::State& state)
{
    auto x = pythonic::numpy::random::rand(state.range(0), state.range(0), state.range(0));
    auto y = pythonic::numpy::random::rand(state.range(0), state.range(0));

    for (auto _ : state)
    {
        pythonic::types::ndarray<double, 3> z = x + y;
        benchmark::DoNotOptimize(z.fbegin());
    }
}
BENCHMARK(pythonic_broadcasting)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef SZ
#undef RANGE
#undef MULTIPLIER
