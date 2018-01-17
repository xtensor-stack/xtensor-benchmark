/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <armadillo>

#include "xsimd/xsimd.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

#define SZ 100
#define RANGE 4, 1000

namespace xt
{
    void eigen_alloc(benchmark::State& state)
    {
        using namespace Eigen;
        while (state.KeepRunning())
        {
            VectorXd res(state.range(0));
        }
    }

    template <std::size_t ALIGN>
    void xsimd_alloc(benchmark::State& state)
    {
        using namespace xsimd;
        using bench_vector = xt::uvector<double,
                                         xsimd::aligned_allocator<double, ALIGN>>;

        while (state.KeepRunning())
        {
            bench_vector res(state.range(0));
        }
    }

    template <std::size_t ALIGN>
    void mm_alloc(benchmark::State& state)
    {
        while (state.KeepRunning())
        {
            auto ptr = _mm_malloc(state.range(0) * sizeof(double), ALIGN);
            _mm_free(ptr);
        }
    }

    template <std::size_t ALIGN>
    void posix_alloc(benchmark::State& state)
    {
        while (state.KeepRunning())
        {
            double* res;

            int status = posix_memalign((void**) &res,
                ((ALIGN >= sizeof(void*)) ? ALIGN : sizeof(void*)),
                sizeof(double) * state.range(0));

            if (status)
            {
                res = 0;
            }
            else
            {
                benchmark::DoNotOptimize(res);
                free(res);
            }
        }
    }

    template <std::size_t ALIGN>
    void posix_chunk(benchmark::State& state)
    {
        while (state.KeepRunning())
        {
            void* res;
            auto sz = arma::memory::enlarge_to_mult_of_chunksize(state.range(0));
            const int failed = posix_memalign(&res, sz, ALIGN);
            if (failed)
            {
                res = 0;
            }
            else
            {
                benchmark::DoNotOptimize(res);
                free(res);
            }
        }
    }

    void std_alloc(benchmark::State& state)
    {
        using bench_vector = xt::uvector<double, std::allocator<double>>;

        while (state.KeepRunning())
        {
            bench_vector res(state.range(0));
        }
    }

    template <std::size_t ALIGN>
    void eigen_aligned_alloc(benchmark::State& state)
    {
        using bench_vector = xt::uvector<double, std::allocator<double>>;
        auto alloc = [](std::size_t size)
        {
            void *original = std::malloc(size * sizeof(double) + ALIGN);
            if (original == 0) return reinterpret_cast<void*>(0);
            void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(ALIGN - 1))) + ALIGN);
            *(reinterpret_cast<void**>(aligned) - 1) = original;
            return aligned;
        };

        while (state.KeepRunning())
        {
            auto ptr = alloc(state.range(0));
            if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
        }
    }

    template <std::size_t ALIGN>
    void super_alloc(benchmark::State& state)
    {
        using bench_vector = xt::uvector<double, std::allocator<double>>;
        auto alloc = [](std::size_t size)
        {
            void *original = _mm_malloc(size * sizeof(double) + ALIGN, 16);
            if (original == 0) return reinterpret_cast<void*>(0);
            void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(ALIGN - 1))) + ALIGN);
            *(reinterpret_cast<void**>(aligned) - 1) = original;
            return aligned;
        };

        while (state.KeepRunning())
        {
            auto ptr = alloc(state.range(0));
            if (ptr) _mm_free(*(reinterpret_cast<void**>(ptr) - 1));
        }
    }

    void arma_alloc(benchmark::State& state)
    {
        using namespace arma;

        while (state.KeepRunning())
        {
            vec test(state.range(0));
            benchmark::DoNotOptimize(test.memptr());
        }
    }

    BENCHMARK(eigen_alloc)->Range(RANGE);

    // BENCHMARK_TEMPLATE(xsimd_alloc, 32)->Range(RANGE);
    // BENCHMARK_TEMPLATE(xsimd_alloc, 16)->Range(RANGE);

    // BENCHMARK(std_alloc)->Range(RANGE);

    // BENCHMARK_TEMPLATE(eigen_aligned_alloc, 16)->Range(RANGE);
    // BENCHMARK_TEMPLATE(eigen_aligned_alloc, 32)->Range(RANGE);
    // BENCHMARK_TEMPLATE(super_alloc, 16)->Range(RANGE);
    BENCHMARK_TEMPLATE(super_alloc, 16)->Range(RANGE);
    BENCHMARK_TEMPLATE(super_alloc, 32)->Range(RANGE);

    BENCHMARK_TEMPLATE(mm_alloc, 16)->Range(RANGE);
    BENCHMARK_TEMPLATE(mm_alloc, 32)->Range(RANGE);
    // BENCHMARK_TEMPLATE(mm_alloc, 64)->Range(RANGE);
    // BENCHMARK_TEMPLATE(mm_alloc, 128)->Range(RANGE);
    // BENCHMARK_TEMPLATE(mm_alloc, 256)->Range(RANGE);

    // BENCHMARK_TEMPLATE(posix_chunk, 16)->Range(RANGE);
    // BENCHMARK(arma_alloc)->Range(RANGE);

    // BENCHMARK_TEMPLATE(posix_alloc, 16)->Range(RANGE);
    // BENCHMARK_TEMPLATE(posix_alloc, 32)->Range(RANGE);
}