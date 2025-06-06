/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "impl/cpu/bicos.hpp"
#include "impl/cpu/agree.hpp"

#include <benchmark/benchmark.h>

using namespace BICOS;
using namespace impl;
using namespace cpu;

template<typename TDescriptor, int FLAGS>
static void bench_bicos_search(benchmark::State &state) {

    int size = 1025;
    benchmark::DoNotOptimize(size);

    std::vector<TDescriptor> data(size);
    std::generate(data.begin(), data.end(), [](){
        TDescriptor v;
        arc4random_buf(&v, sizeof(TDescriptor));
        return v;
    });

    for (auto _ : state) {
        int ret = BicosSearch<TDescriptor>::template invoke<FLAGS>(data[0], data.data() + 1, data.size() - 1);
        benchmark::DoNotOptimize(ret);
    }
}

template<typename TDescriptor, int FLAGS>
static void bench_bicos_search_vanilla(benchmark::State &state) {

    int size = 1025;
    benchmark::DoNotOptimize(size);

    std::vector<TDescriptor> data(size);
    std::generate(data.begin(), data.end(), [](){
        TDescriptor v;
        arc4random_buf(&v, sizeof(TDescriptor));
        return v;
    });

    for (auto _ : state) {
        int ret = bicos_search_<TDescriptor, FLAGS>(data[0], data.data() + 1, data.size() - 1);
        benchmark::DoNotOptimize(ret);
    }
}

static void bench_nxcorr(benchmark::State &state) {
    
    int size = 25;
    benchmark::DoNotOptimize(size);

    std::vector<uint8_t> a(2 * size);
    std::generate(a.begin(), a.end(), [](){
        return (uint8_t)arc4random();
    });

    for (auto _ : state) {
        int ret = nxcorr(a.data(), a.data() + size, size, std::nullopt);
        benchmark::DoNotOptimize(ret);
    }

}

BENCHMARK(bench_bicos_search<uint32_t, 0>);
BENCHMARK(bench_bicos_search<uint32_t, BICOSFLAGS_NODUPES>);
BENCHMARK(bench_bicos_search<uint32_t, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>);

BENCHMARK(bench_bicos_search_vanilla<uint32_t, 0>);
BENCHMARK(bench_bicos_search_vanilla<uint32_t, BICOSFLAGS_NODUPES>);
BENCHMARK(bench_bicos_search_vanilla<uint32_t, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>);

BENCHMARK(bench_bicos_search<uint64_t, 0>);
BENCHMARK(bench_bicos_search<uint64_t, BICOSFLAGS_NODUPES>);
BENCHMARK(bench_bicos_search<uint64_t, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>);
BENCHMARK(bench_bicos_search<uint128_t, 0>);
BENCHMARK(bench_bicos_search<uint128_t, BICOSFLAGS_NODUPES>);
BENCHMARK(bench_bicos_search<uint128_t, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>);
BENCHMARK(bench_bicos_search<std::bitset<256>, 0>);
BENCHMARK(bench_bicos_search<std::bitset<256>, BICOSFLAGS_NODUPES>);
BENCHMARK(bench_bicos_search<std::bitset<256>, BICOSFLAGS_NODUPES | BICOSFLAGS_CONSISTENCY>);

BENCHMARK(bench_nxcorr);

BENCHMARK_MAIN();
