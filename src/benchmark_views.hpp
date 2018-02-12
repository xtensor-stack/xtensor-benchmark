/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/Core>
// #include <unsupported/Eigen/MatrixFunctions>

#ifdef HAS_BLITZ
#include <blitz/array.h>
#endif

#include <armadillo>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

    // #include <pythonic/core.hpp>
    // #include <pythonic/types/ndarray.hpp>
    // #include <pythonic/numpy/random/rand.hpp>

#ifndef EIGEN_VECTORIZE
static_assert(false, "NOT VECTORIZING");
#endif

#define RANGE 128, 128

namespace xt
{
	void xtensor_dynamicview(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 2, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0), state.range(0)});
		tensor b = random::rand<double>({state.range(0), state.range(0)});

        auto sv = xt::slice_vector(a, range(0, 5), range(0, 5));

        auto av = xt::dynamic_view(a, sv);
        auto bv = xt::dynamic_view(b, sv);

		while (state.KeepRunning())
		{
			tensor res(av + bv);
			benchmark::DoNotOptimize(res.raw_data());
		}
	}

	void xtensor_view(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 2, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0), state.range(0)});
		tensor b = random::rand<double>({state.range(0), state.range(0)});

        auto av = xt::view(a, range(0, 5), range(0, 5));
        auto bv = xt::view(b, range(0, 5), range(0, 5));

		while (state.KeepRunning())
		{
			tensor res(av + bv);
			benchmark::DoNotOptimize(res.raw_data());
		}
	}

	void eigen_view(benchmark::State& state)
	{
		using namespace Eigen;
		MatrixXd a = MatrixXd::Random(state.range(0), state.range(0));
		MatrixXd b = MatrixXd::Random(state.range(0), state.range(0));

        auto av = a.topLeftCorner(5, 5);
        auto bv = b.topLeftCorner(5, 5);

		while (state.KeepRunning())
		{
			MatrixXd res(5, 5);
			res.noalias() = av + bv;
			benchmark::DoNotOptimize(res.data());
		}
	}

	void eigen_map(benchmark::State& state)
	{
		using namespace Eigen;
		MatrixXd a = VectorXd::Random(state.range(0));
		MatrixXd b = VectorXd::Random(state.range(0));

        auto av = Map<VectorXd, 0, InnerStride<2>>(a.data(), a.size() / 2);
        auto bv = Map<VectorXd, 0, InnerStride<2>>(b.data(), b.size() / 2);

		while (state.KeepRunning())
		{
			VectorXd res(av + bv);
			benchmark::DoNotOptimize(res.data());
		}
	}

	void xtensor_stride_2(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 1, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0)});
		tensor b = random::rand<double>({state.range(0)});

        auto sv = xt::slice_vector(a, range(0, state.range(0), 2));

        auto av = xt::dynamic_view(a, sv);
        auto bv = xt::dynamic_view(b, sv);

		while (state.KeepRunning())
		{
			tensor res(av + bv);
			benchmark::DoNotOptimize(res.data());
		}
	}

	void xtensor_max_speed(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 1, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0) / 2});
		tensor b = random::rand<double>({state.range(0) / 2});

		while (state.KeepRunning())
		{
			tensor res(a + b);
			benchmark::DoNotOptimize(res.data());
		}
	}

	void xtensor_adapt_view(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 1, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0)});
		tensor b = random::rand<double>({state.range(0)});
		std::size_t range_arg = static_cast<std::size_t>(state.range(0));
		std::array<std::size_t, 1> shape = {range_arg / 2};
		std::array<std::size_t, 1> stride = {2};
		auto av = xt::adapt(std::move(a.data()), shape, stride);
		auto bv = xt::adapt(std::move(b.data()), shape, stride);

		while (state.KeepRunning())
		{
			tensor res(av + bv);
			benchmark::DoNotOptimize(res.data());
		}
	}

	void xtensor_hand_loop(benchmark::State& state)
	{
		using namespace xt;
		using allocator = xsimd::aligned_allocator<double, 32>;
		using tensor = xtensor_container<xt::uvector<double, allocator>, 1, layout_type::row_major>;

		tensor a = random::rand<double>({state.range(0)});
		tensor b = random::rand<double>({state.range(0)});
		std::array<std::size_t, 1> shape = {static_cast<std::size_t>(state.range(0)) / 2};
		while (state.KeepRunning())
		{
			tensor res(shape);
			std::size_t j = 0;
			for (std::size_t i = 0; i < state.range(0); i += 2)
			{
				res(j) = a(i) + b(i);
				++j;
			}
			benchmark::DoNotOptimize(res.data());
		}
	}

	BENCHMARK(xtensor_view)->Range(RANGE);
	BENCHMARK(xtensor_dynamicview)->Range(RANGE);
	BENCHMARK(eigen_view)->Range(RANGE);
	BENCHMARK(eigen_map)->Range(RANGE);
	BENCHMARK(xtensor_stride_2)->Range(RANGE);
	BENCHMARK(xtensor_max_speed)->Range(RANGE);
	BENCHMARK(xtensor_adapt_view)->Range(RANGE);
	BENCHMARK(xtensor_hand_loop)->Range(RANGE);
}