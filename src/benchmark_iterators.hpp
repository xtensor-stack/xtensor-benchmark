/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
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

#undef RANGE
#undef MULTIPLIER

