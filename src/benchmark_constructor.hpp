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

#define RANGE 3, 1000
#define MULTIPLIER 8

#ifdef HAS_XTENSOR
void Construct2D_XTensor(benchmark::State& state)
{
    for (auto _ : state)
    {
        xt::xtensor<double, 2> vTensor({state.range(0), state.range(0)});
        benchmark::DoNotOptimize(vTensor);
    }
}
BENCHMARK(Construct2D_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_EIGEN
void Construct2D_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    for (auto _ : state)
    {
        MatrixXd vMatrix(state.range(0), state.range(0));
        benchmark::DoNotOptimize(vMatrix);
    }
}
BENCHMARK(Construct2D_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_BLITZ
void Construct2D_Blitz(benchmark::State& state)
{
    using namespace blitz;
    for (auto _ : state)
    {
        Array<double, 2> vArray(state.range(0), state.range(0));
        benchmark::DoNotOptimize(vArray);
    }
}
BENCHMARK(Construct2D_Blitz)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_XTENSOR
void ConstructRandom2D_XTensor(benchmark::State& state)
{
    for (auto _ : state)
    {
        xt::xtensor<double, 2> vTensor = xt::random::rand<double>({state.range(0), state.range(0)});
        benchmark::DoNotOptimize(vTensor);
    }
}
BENCHMARK(ConstructRandom2D_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_EIGEN
void ConstructRandom2D_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    for (auto _ : state)
    {
        MatrixXd vMatrix = MatrixXd::Random(state.range(0), state.range(0));
        benchmark::DoNotOptimize(vMatrix);
    }
}
BENCHMARK(ConstructRandom2D_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif


#ifdef HAS_XTENSOR
void ConstructView2d_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double,2> vA = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        auto vAView = xt::view(vA, all(), all());
        benchmark::DoNotOptimize(vAView);
    }
}
BENCHMARK(ConstructView2d_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_EIGEN
void ConstructView2d_Eigen(benchmark::State& state)
{
    using namespace Eigen;
    MatrixXd vA = MatrixXd::Random(state.range(0), state.range(0));

    for (auto _ : state)
    {
        auto vAView = vA.topLeftCorner(state.range(0), state.range(0));
        benchmark::DoNotOptimize(vAView);
    }
}
BENCHMARK(ConstructView2d_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER

