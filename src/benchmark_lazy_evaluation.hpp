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

#define RANGE 1, 10
#define MULTIPLIER 2
#define ARRAY_SIZE 1000

#ifdef HAS_XTENSOR
void Add2DLazy_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> a = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> b = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});

    double percentage = static_cast<double>(state.range(0)) / 10;
    size_t reducedSize = static_cast<size_t>(percentage * a.size());
    double tmp = 0.0;

    for (auto _ : state)
    {
        auto&& res = (a + b);
        for (int i = 0; i < reducedSize; ++i){
            tmp += res[i];
        }
        benchmark::DoNotOptimize(tmp);
    }
}
BENCHMARK(Add2DLazy_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void Add2DEval_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> a = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> b = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});

    double percentage = static_cast<double>(state.range(0)) / 10;
    size_t reducedSize = static_cast<size_t>(percentage * a.size());
    double tmp = 0.0;

    for (auto _ : state)
    {
        auto&& res = xt::eval(a + b);
        for (int i = 0; i < reducedSize; ++i){
            tmp += res[i];
        }
        benchmark::DoNotOptimize(tmp);
    }
}
BENCHMARK(Add2DEval_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void AddDivide2DLazy_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> a = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> b = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> c = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> d = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});

    double percentage = static_cast<double>(state.range(0)) / 10;
    size_t reducedSize = static_cast<size_t>(percentage * a.size());
    double tmp = 0.0;

    for (auto _ : state)
    {
        auto&& res = (a + b + c) / d;
        for (int i = 0; i < reducedSize; ++i){
            tmp += res[i];
        }
        benchmark::DoNotOptimize(a.data());
    }
}
BENCHMARK(AddDivide2DLazy_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void AddDivide2DEval_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> a = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> b = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> c = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});
    xtensor<double, 2> d = random::rand<double>({ARRAY_SIZE, ARRAY_SIZE});

    double percentage = static_cast<double>(state.range(0)) / 10;
    size_t reducedSize = static_cast<size_t>(percentage * a.size());
    double tmp = 0.0;

    for (auto _ : state)
    {
        auto&& res = xt::eval((a + b + c) / d);
        for (int i = 0; i < reducedSize; ++i){
            tmp += res[i];
        }
        benchmark::DoNotOptimize(res.data());
    }
}
BENCHMARK(AddDivide2DEval_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER

