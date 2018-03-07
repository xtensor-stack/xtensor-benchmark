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

#undef SZ 100
#undef RANGE 3, 64
#undef MULTIPLIER 8
