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
#ifndef EIGEN_VECTORIZE
static_assert(false, "NOT VECTORIZING");
#endif
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
    namespace oned
    {

        void xtensor_test_1D(benchmark::State& state)
        {
            using namespace xt;
            using allocator = xsimd::aligned_allocator<double, 32>;
            using tensor = xtensor_container<xt::uvector<double, allocator>, 1, layout_type::row_major>;

            tensor a = random::rand<double>({state.range(0)});
            tensor b = random::rand<double>({state.range(0)});

            while (state.KeepRunning())
            {
                tensor res(a + b);
                benchmark::DoNotOptimize(res.raw_data());
            }
        }
        BENCHMARK(xtensor_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

        void xsimd_test_1D(benchmark::State& state)
        {
            using allocator = xsimd::aligned_allocator<double, 32>;
            using bench_vector = xt::uvector<double, xsimd::aligned_allocator<double, 32>>;
            using batch = xsimd::simd_type<double>;
            using namespace xt;
            using namespace xsimd;

            bench_vector a(state.range(0));
            bench_vector b(state.range(0));
            bool is_aligned = true;

            std::size_t sz = state.range(0);

            while (state.KeepRunning())
            {
                // std::cout << align_begin << ", " << align_end << std::endl;
                // std::cout << sz << ", " << batch::size << ", " << align_begin << std::endl;
                std::size_t align_begin = is_aligned ? 0 : xsimd::get_alignment_offset(a.data(), sz, batch::size);
                std::size_t align_end = align_begin + ((sz - align_begin) & ~(batch::size - 1));

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
                for (; i < sz; ++i)
                {
                    res[i] = a[i] + b[i];
                }

                benchmark::DoNotOptimize(res.data());
            }
        }
        BENCHMARK(xsimd_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

#ifdef HAS_EIGEN
        void eigen_test_1D(benchmark::State& state)
        {
            using namespace Eigen;
            VectorXd a = VectorXd::Random(state.range(0));
            VectorXd b = VectorXd::Random(state.range(0));
            while (state.KeepRunning())
            {
                VectorXd res(a + b);
                benchmark::DoNotOptimize(res.data());
            }
        }
        BENCHMARK(eigen_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_BLITZ
        void blitz_test_1D(benchmark::State& state)
        {
            using namespace blitz;
            Array<double, 1> a(state.range(0));
            Array<double, 1> b(state.range(0));
            while (state.KeepRunning())
            {
                Array<double, 1> res(a + b);
                benchmark::DoNotOptimize(res.data());
            }
        }
        BENCHMARK(blitz_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_ARMADILLO
        void arma_test_1D(benchmark::State& state)
        {
            using namespace arma;
            vec a = randu<vec>(state.range(0));
            vec b = randu<vec>(state.range(0));
            while (state.KeepRunning())
            {
                vec res(a + b);
                benchmark::DoNotOptimize(res.memptr());
            }
        }
        BENCHMARK(arma_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#ifdef HAS_PYTHONIC
        void pythonic_test_1D(benchmark::State &state)
        {
            auto x = pythonic::numpy::random::rand(state.range(0));
            auto y = pythonic::numpy::random::rand(state.range(0));

            while (state.KeepRunning())
            {
                pythonic::types::ndarray<double, 1> z(x + y);
                benchmark::DoNotOptimize(z.fbegin());
            }
        }
        BENCHMARK(pythonic_test_1D)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

    }
}

#undef SZ
#undef RANGE
#undef MULTIPLIER

