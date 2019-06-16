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

#ifdef HAS_CLIKE
#include <vector>
#endif

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

#ifdef HAS_BLITZ
#include <blitz/array.h>
#endif

#ifdef HAS_VIGRA
#include "vigra/multi_array.hxx"
#include "vigra/multi_math.hxx"
#endif

#define RANGE 16, 1024
#define MULTIPLIER 8

#ifdef HAS_XTENSOR
void IterateWhole2D_XTensor(benchmark::State& state)
{
    xt::xtensor<double, 2> vTensor({state.range(0), state.range(0)});
    for (auto _ : state)
    {
        double vTmp = 0.0;
        for (auto it = vTensor.begin(); it != vTensor.end(); ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2D_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void IterateWhole2DView_XTensor(benchmark::State& state)
{
    xt::xtensor<double, 2> vTensor({state.range(0), state.range(0)});
    auto vView = xt::view(vTensor, xt::all());
    for (auto _ : state)
    {
        double vTmp = 0.0;
        for (auto it = vView.begin(); it != vView.end(); ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2DView_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void IterateWhole2DStridedView_XTensor(benchmark::State& state)
{
    xt::xtensor<double, 2> vTensor({state.range(0), state.range(0)});
    auto vView = xt::strided_view(vTensor, {xt::all()});
    for (auto _ : state)
    {
        double vTmp = 0.0;
        for (auto it = vView.begin(); it != vView.end(); ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2DStridedView_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_CLIKE
void IterateWhole2D_CLike(benchmark::State& state)
{
    std::vector<double> vTensor(state.range(0) * state.range(0));
    for (auto _ : state)
    {
        double vTmp = 0.0;
        double* it = vTensor.data();
        double* end = vTensor.data() + vTensor.size();
        for (; it != end; ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2D_CLike)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

//#ifdef HAS_EIGEN
//void IterateWhole2D_Eigen(benchmark::State& state)
//{
//    using namespace Eigen;
//    MatrixXd vMatrix(state.range(0), state.range(0));
//    for (auto _ : state)
//    {
//        double vTmp = 0.0;
//        for (auto it = vMatrix.begin(); it != vMatrix.end(); ++it) {
//            vTmp += *it;
//        }
//        benchmark::DoNotOptimize(vTmp);
//    }
//}
//BENCHMARK(IterateWhole2D_Eigen)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
//#endif

#ifdef HAS_BLITZ
void IterateWhole2D_Blitz(benchmark::State& state)
{
    using namespace blitz;
    Array<double, 2> vArray(state.range(0), state.range(0));
    for (auto _ : state)
    {
        double vTmp = 0.0;
        for (auto it = vArray.begin(); it != vArray.end(); ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2D_Blitz)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_VIGRA
void IterateWhole2D_Vigra(benchmark::State& state)
{
    using namespace vigra::multi_math;
    vigra::MultiArray<2, double> vArray(state.range(0), state.range(0));
    for (auto _ : state)
    {
        double vTmp = 0.0;
        for (auto it = vArray.begin(); it != vArray.end(); ++it) {
            vTmp += *it;
        }
        benchmark::DoNotOptimize(vTmp);
    }
}
BENCHMARK(IterateWhole2D_Vigra)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER

