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
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"

#ifdef HAS_PYTHONIC
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/numpy/random/rand.hpp>
#endif

#define SZ 100
#define RANGE 3, 64
#define MULTIPLIER 8

namespace xt
{
    void xtensor_broadcasting(benchmark::State& state)
    {
        using namespace xt;
        using allocator = xsimd::aligned_allocator<double, 32>;
        using tensor3 = xtensor_container<xt::uvector<double, allocator>, 3, layout_type::row_major>;
        using tensor2 = xtensor_container<xt::uvector<double, allocator>, 2, layout_type::row_major>;

        tensor3 a = random::rand<double>({state.range(0), state.range(0), state.range(0)});
        tensor2 b = random::rand<double>({state.range(0), state.range(0)});

        for (auto _ : state)
        {
            tensor3 res(a + b);
            benchmark::DoNotOptimize(res.raw_data());
        }
    }
    BENCHMARK(xtensor_broadcasting)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

    void xarray_broadcasting(benchmark::State& state)
    {
        using namespace xt;
        using allocator = xsimd::aligned_allocator<double, 32>;
        using tensor3 = xarray_container<xt::uvector<double, allocator>, layout_type::row_major>;
        using tensor2 = xarray_container<xt::uvector<double, allocator>, layout_type::row_major>;

        tensor3 a = random::rand<double>({state.range(0), state.range(0), state.range(0)});
        tensor2 b = random::rand<double>({state.range(0), state.range(0)});

        for (auto _ : state)
        {
            tensor3 res(a + b);
            benchmark::DoNotOptimize(res.raw_data());
        }
    }
    BENCHMARK(xarray_broadcasting)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

    template <std::size_t N>
    void manual_broadcast_xtensorf(benchmark::State& state)
    {
        auto a = xt::xtensorf<double, xt::xshape<N, N, N>>();
        auto b = xt::xtensorf<double, xt::xshape<N, N>>();
        for (auto _ : state)
        {
            auto c = xt::xtensorf<double, xt::xshape<N, N, N>>();
            for (std::size_t i = 0; i < a.shape()[0]; ++i)
                for (std::size_t j = 0; j < a.shape()[1]; ++j)
                    for (std::size_t k = 0; k < a.shape()[2]; ++k)
                        c(i, j, k) = a(i, j, k) + b(i, j, k);
            benchmark::DoNotOptimize(c.raw_data());
        }
    }
    BENCHMARK_TEMPLATE(manual_broadcast_xtensorf, 3);
    BENCHMARK_TEMPLATE(manual_broadcast_xtensorf, 8);
    BENCHMARK_TEMPLATE(manual_broadcast_xtensorf, 64);

    void manual_broadcast_xtensor(benchmark::State& state)
    {
        auto a = xt::xtensor<double, 3>::from_shape({state.range(0), state.range(0), state.range(0)});
        auto b = xt::xtensor<double, 2>::from_shape({state.range(0), state.range(0)});
        for (auto _ : state)
        {
            xt::xtensor<double, 3> c = xt::xtensor<double, 3>::from_shape({state.range(0), state.range(0), state.range(0)});
            for (std::size_t i = 0; i < a.shape()[0]; ++i)
                for (std::size_t j = 0; j < a.shape()[1]; ++j)
                    for (std::size_t k = 0; k < a.shape()[2]; ++k)
                        c(i, j, k) = a(i, j, k) + b(i, j, k);
            benchmark::DoNotOptimize(c.raw_data());
        }
    }
    BENCHMARK(manual_broadcast_xtensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

    void manual_broadcast_xarray(benchmark::State& state)
    {
        auto a = xt::xarray<double>::from_shape({state.range(0), state.range(0), state.range(0)});
        auto b = xt::xarray<double>::from_shape({state.range(0), state.range(0)});
        for (auto _ : state)
        {
            xt::xarray<double> c = xt::xarray<double>::from_shape({state.range(0), state.range(0), state.range(0)});
            for (std::size_t i = 0; i < a.shape()[0]; ++i)
                for (std::size_t j = 0; j < a.shape()[1]; ++j)
                    for (std::size_t k = 0; k < a.shape()[2]; ++k)
                        c(i, j, k) = a(i, j, k) + b(i, j, k);
            benchmark::DoNotOptimize(c.raw_data());
        }
    }
    BENCHMARK(manual_broadcast_xarray)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

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

}

#undef SZ
#undef RANGE
#undef MULTIPLIER
