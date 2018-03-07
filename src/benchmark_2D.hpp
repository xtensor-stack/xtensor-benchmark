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

#define SZ 100
#define RANGE 3, 1000
#define MULTIPLIER 8

namespace xt
{

	void xtensor_2D(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 2, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0), state.range(0)});
		tensor b = random::rand<double>({state.range(0), state.range(0)});

		for (auto _ : state)
		{
			tensor res(a + b);
			benchmark::DoNotOptimize(res.raw_data());
		}
	}
	BENCHMARK(xtensor_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

	void xsimd_2D(benchmark::State& state)
	{
		using allocator = xsimd::aligned_allocator<double, 32>;
		using bench_vector = xt::uvector<double, xsimd::aligned_allocator<double, 32>>;
		using batch = xsimd::simd_type<double>;
		using namespace xt;
		using namespace xsimd;

		bench_vector a(state.range(0) * state.range(0));
		bench_vector b(state.range(0) * state.range(0));
        std::size_t s = a.size();
        bool is_aligned = false;

        std::size_t sz = state.range(0) * state.range(0);

		for (auto _ : state)
		{
	        std::size_t align_begin = is_aligned ? 0 : xsimd::get_alignment_offset(a.data(), s, batch::size);
	        std::size_t align_end = align_begin + ((s - align_begin) & ~(batch::size - 1));
			bench_vector res(sz);
			std::size_t i = 0;
			for (; i < align_begin; ++i)
			{
            	res[i] = a[i] + b[i];
			}
            for (; i < align_end; i += batch::size)
            {
                batch blhs(&a[i], aligned_mode());
                batch brhs(&b[i], aligned_mode());
                batch bres = blhs + brhs;
                bres.store_aligned(&res[i]);
            }
			for (; i < s; ++i)
			{
            	res[i] = a[i] + b[i];
			}

            benchmark::DoNotOptimize(res.data());
		}
	}
	BENCHMARK(xsimd_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

#ifdef HAS_EIGEN
	void eigen_2D(benchmark::State& state)
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
	BENCHMARK(eigen_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_BLITZ
	void blitz_2D(benchmark::State& state)
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
	BENCHMARK(blitz_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_ARMADILLO
	void arma_2D(benchmark::State& state)
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
	BENCHMARK(arma_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_PYTHONIC
	void pythonic_2D(benchmark::State& state)
	{
		auto x = pythonic::numpy::random::rand(state.range(0), state.range(0));
		auto y = pythonic::numpy::random::rand(state.range(0), state.range(0));

		for (auto _ : state)
		{
			pythonic::types::ndarray<double, 2> z = x + y;
			benchmark::DoNotOptimize(z.fbegin());
		}
	}
	BENCHMARK(pythonic_2D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

}

#undef SZ
#undef RANGE
#undef MULTIPLIER

