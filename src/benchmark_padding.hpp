#include <benchmark/benchmark.h>

#ifdef HAS_XTENSOR
#include <xtensor/xtensor.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>
#include "xtensor/xrandom.hpp"
#include <xtensor/xnoalias.hpp>
#include <xtensor/xadapt.hpp>
#endif

#define RANGE 16, 1000
#define MULTIPLIER 8

#ifdef HAS_XTENSOR
void Pad2dConstant_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> vInput = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 2> vOutput = pad(vInput, {{100, 100}, {100, 100}}, pad_mode::constant, 5);
        benchmark::DoNotOptimize(vOutput.data());
    }
}
BENCHMARK(Pad2dConstant_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void Pad2dMirror_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> vInput = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 2> vOutput = pad(vInput, {{100, 100}, {100, 100}}, pad_mode::symmetric, 5);
        benchmark::DoNotOptimize(vOutput.data());
    }
}
BENCHMARK(Pad2dMirror_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);

void Pad2dCyclic_XTensor(benchmark::State& state)
{
    using namespace xt;

    xtensor<double, 2> vInput = random::rand<double>({state.range(0), state.range(0)});

    for (auto _ : state)
    {
        xtensor<double, 2> vOutput = pad(vInput, {{100, 100}, {100, 100}}, pad_mode::periodic, 5);
        benchmark::DoNotOptimize(vOutput.data());
    }
}
BENCHMARK(Pad2dCyclic_XTensor)->RangeMultiplier(MULTIPLIER)->Range(RANGE);
#endif

#undef RANGE
#undef MULTIPLIER
