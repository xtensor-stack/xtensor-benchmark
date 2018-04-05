/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif

namespace xt
{

    template <std::size_t N, std::size_t M>
    void xtensor_fixed(benchmark::State& state)
    {
        using namespace xt;
        using tensor = xtensorf<double, xshape<N, M>>;

        tensor a = random::rand<double>({N, M});
        tensor b = random::rand<double>({N, M});

        for (auto _ : state)
        {
            tensor res;
            xt::noalias(res) = a + b;
            benchmark::DoNotOptimize(res.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(xtensor_fixed, 3, 3);
    BENCHMARK_TEMPLATE(xtensor_fixed, 5, 5);
    BENCHMARK_TEMPLATE(xtensor_fixed, 20, 20);


#ifdef HAS_EIGEN
    template <std::size_t N, std::size_t M>
    void eigen_fixed(benchmark::State& state)
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
    BENCHMARK_TEMPLATE(eigen_fixed, 3, 3);
    BENCHMARK_TEMPLATE(eigen_fixed, 5, 5);
    BENCHMARK_TEMPLATE(eigen_fixed, 20, 20);
#endif

    template <std::size_t N, std::size_t M>
    void xtensor_nonfixed(benchmark::State& state)
    {
        using namespace xt;
        using tensor = xtensor<double, 2>;

        tensor a = random::rand<double>({N, M});
        tensor b = random::rand<double>({N, M});

        for (auto _ : state)
        {
            tensor res;
            xt::noalias(res) = a + b;
            benchmark::DoNotOptimize(res.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(xtensor_nonfixed, 3, 3);
    BENCHMARK_TEMPLATE(xtensor_nonfixed, 5, 5);
    BENCHMARK_TEMPLATE(xtensor_nonfixed, 20, 20);


}

#undef SZ
#undef RANGE
#undef MULTIPLIER