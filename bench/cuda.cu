/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024  Robotics Group @ Julius-Maximilian University
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

#include <benchmark/benchmark.h>

#include <opencv2/core/cuda.hpp>
#include <random>

#include "common.hpp"
#include "cuda.hpp"
#include "fileutils.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"
#include "opencv2/core.hpp"
#include "opencv2/core/traits.hpp"
#include "stepbuf.hpp"

using namespace BICOS;
using namespace impl;

constexpr int seed = 0x600DF00D;

constexpr double thresh = 0.9;
constexpr double minvar = 10.0;
constexpr float step = 0.25;
static const cv::Size size(3300, 2200);

template<typename TPrecision, cuda::NXCVariant VARIANT>
__global__ void nxcorr_kernel(const uint8_t* a, const uint8_t* b, size_t n, TPrecision* out, TPrecision minvar) {
    if constexpr (std::is_same_v<TPrecision, float>)
        *out = cuda::nxcorrf<VARIANT>(a, b, n, minvar);
    else
        *out = cuda::nxcorrd<VARIANT>(a, b, n, minvar);
}

template<typename TPrecision, cuda::NXCVariant VARIANT>
void bench_nxcorr_subroutine(benchmark::State& state) {
    uint8_t _a[50], _b[50], *a, *b;
    TPrecision minvar = 100;

    for (size_t i = 0; i < sizeof(_a); ++i) {
        _a[i] = rand();
        _b[i] = rand();
    }

    cudaMalloc(&a, sizeof(_a));
    cudaMalloc(&b, sizeof(_b));

    cudaMemcpy(a, _a, sizeof(_a), cudaMemcpyHostToDevice);
    cudaMemcpy(b, _b, sizeof(_b), cudaMemcpyHostToDevice);

    TPrecision* out;
    cudaMalloc(&out, 1);

    for (auto _: state) {
        nxcorr_kernel<TPrecision, VARIANT><<<1, 1>>>(a, b, sizeof(_a), out, minvar);
        cudaDeviceSynchronize();
    }
}

template<typename TInput>
void bench_agree_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);
    const int n = 10;

    cv::Mat_<int16_t> randdisp(size);
    cv::randu(randdisp, -1, size.width);
    cv::cuda::GpuMat randdisp_dev(randdisp);

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<TInput>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<TInput> randmat(size);
        cv::randu(randmat, 0, std::numeric_limits<TInput>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    cuda::RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    const dim3 block = cuda::max_blocksize(cuda::agree_kernel<TInput, double, cuda::NXCVariant::MINVAR>);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::agree_kernel<TInput, double, cuda::NXCVariant::MINVAR>
            <<<grid, block>>>(randdisp_dev, devptr, n, thresh, 0.0, minvar, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput>
void bench_agree_subpixel_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);
    const int n = 10;

    cv::Mat_<int16_t> randdisp(size);
    cv::randu(randdisp, -1, size.width);
    cv::cuda::GpuMat randdisp_dev(randdisp);

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<TInput>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<TInput> randmat(size);
        cv::randu(randmat, 0, std::numeric_limits<TInput>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    cuda::RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    const dim3 block = cuda::max_blocksize(cuda::agree_subpixel_kernel<TInput, double, cuda::NXCVariant::MINVAR>);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::agree_subpixel_kernel<TInput, double, cuda::NXCVariant::MINVAR>
            <<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, minvar, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput>
void bench_agree_subpixel_kernel_smem(benchmark::State& state) {
    cv::setRNGSeed(seed);
    const int n = 10;

    cv::Mat_<int16_t> randdisp(size);
    cv::randu(randdisp, -1, size.width);
    cv::cuda::GpuMat randdisp_dev(randdisp);

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<TInput>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<TInput> randmat(size);
        cv::randu(randmat, 0, std::numeric_limits<TInput>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    cuda::RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    size_t smem_size = size.width * n * sizeof(TInput);

    bool smem_fits = cudaSuccess == cudaFuncSetAttribute(
        cuda::agree_subpixel_kernel_smem<TInput, double, cuda::NXCVariant::MINVAR>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    if (!smem_fits) {
        state.SkipWithMessage("smem too small");
        return;
    }

    const dim3 block = cuda::max_blocksize(cuda::agree_subpixel_kernel_smem<TInput, double, cuda::NXCVariant::MINVAR>, smem_size);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::agree_subpixel_kernel_smem<TInput, double, cuda::NXCVariant::MINVAR>
            <<<grid, block, smem_size>>>(randdisp_dev, devptr, n, thresh, step, minvar, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename T>
void randomize_seeded(cpu::StepBuf<T>& sb) {
    static thread_local std::independent_bits_engine<std::default_random_engine, CHAR_BIT, uint8_t>
        ibe((uint8_t)seed);

    T* p = sb.row(0);

    std::generate(p, p + sb.size().area(), ibe);
}

template<typename TDescriptor>
void bench_bicos_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);

    cpu::StepBuf<TDescriptor> ld(size), rd(size);

    randomize_seeded(ld);
    randomize_seeded(rd);

    cuda::StepBuf<TDescriptor> ld_dev(ld), rd_dev(rd);

    cuda::RegisteredPtr lptr(&ld_dev, 1, true), rptr(&rd_dev, 1, true);

    cv::cuda::GpuMat out(size, cv::DataType<int16_t>::type);

    const dim3 block = cuda::max_blocksize(cuda::bicos_kernel<TDescriptor>);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::bicos_kernel<TDescriptor><<<grid, block>>>(lptr, rptr, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TDescriptor>
void bench_bicos_kernel_smem(benchmark::State& state) {
    cv::setRNGSeed(seed);

    cpu::StepBuf<TDescriptor> ld(size), rd(size);

    randomize_seeded(ld);
    randomize_seeded(rd);

    cuda::StepBuf<TDescriptor> ld_dev(ld), rd_dev(rd);

    cuda::RegisteredPtr lptr(&ld_dev, 1, true), rptr(&rd_dev, 1, true);

    cv::cuda::GpuMat out(size, cv::DataType<int16_t>::type);

    size_t smem_size = size.width * sizeof(TDescriptor);

    bool smem_fits = cudaSuccess == cudaFuncSetAttribute(
        cuda::bicos_kernel_smem<TDescriptor>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    if (!smem_fits) {
        state.SkipWithMessage("smem too small");
        return;
    }

    const dim3 block = cuda::max_blocksize(cuda::bicos_kernel_smem<TDescriptor>, smem_size);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::bicos_kernel_smem<TDescriptor><<<grid, block, smem_size>>>(lptr, rptr, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput, typename TDescriptor, TransformMode mode>
void bench_descriptor_transform_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);

    int bits = sizeof(TDescriptor) * 8;
    int n = mode == TransformMode::FULL ? int((2 + std::sqrt(4 - 4 * ( 3 - bits ))) / 2.0) : (bits + 7) / 4;

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<TInput>> devinput;

    for (int i = 0; i < n; ++i) {
        cv::Mat_<TInput> randmat(size);
        cv::randu(randmat, 0, std::numeric_limits<TInput>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    cuda::RegisteredPtr inptr(devinput.data(), n, true);

    cuda::StepBuf<TDescriptor> out(size);
    cuda::RegisteredPtr outptr(&out);

    const dim3 block = cuda::max_blocksize(mode == TransformMode::FULL ? cuda::transform_full_kernel<TInput, TDescriptor> : cuda::transform_limited_kernel<TInput, TDescriptor>);
    const dim3 grid  = create_grid(block, size);

    if constexpr (mode == TransformMode::FULL)
        for (auto _: state)
            cuda::transform_full_kernel<TInput, TDescriptor>
                <<<grid, block>>>(inptr, n, size, outptr);
    else
        for (auto _: state)
            cuda::transform_limited_kernel<TInput, TDescriptor>
                <<<grid, block>>>(inptr, n, size, outptr);

    assertCudaSuccess(cudaGetLastError());
}

void bench_integration(benchmark::State& state) {
    std::vector<SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> ldev, rdev;

    read_sequence(SOURCE_ROOT "/data/left", SOURCE_ROOT "/data/right", lseq, rseq, true);
    sort_sequence_to_stack(lseq, rseq, lhost, rhost);
    matvec_to_gpu(lhost, rhost, ldev, rdev);

    int n = std::min(state.range(0), (int64_t)ldev.size());
    float step = 0.01f * state.range(1);

    ldev.resize(n);
    rdev.resize(n);

    Config c { .nxcorr_thresh = thresh,
               .subpixel_step = step == 0.0f ? std::nullopt : std::optional(step),
               .mode = TransformMode::LIMITED };

    cv::cuda::GpuMat out;
    out.create(ldev.front().size(), cv::DataType<disparity_t>::type);

    for (auto _: state) {
        cuda::match(ldev, rdev, out, c, cv::cuda::Stream::Null());
        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_nxcorr_subroutine<float, cuda::NXCVariant::MINVAR>)
    ->Repetitions(10)
    ->ReportAggregatesOnly(true);
BENCHMARK(bench_nxcorr_subroutine<float, cuda::NXCVariant::PLAIN>)
    ->Repetitions(10)
    ->ReportAggregatesOnly(true);
BENCHMARK(bench_nxcorr_subroutine<double, cuda::NXCVariant::MINVAR>)
    ->Repetitions(10)
    ->ReportAggregatesOnly(true);
BENCHMARK(bench_nxcorr_subroutine<double, cuda::NXCVariant::PLAIN>)
    ->Repetitions(10)
    ->ReportAggregatesOnly(true);


BENCHMARK(bench_agree_kernel<uint8_t>);
BENCHMARK(bench_agree_kernel<uint16_t>);
BENCHMARK(bench_agree_subpixel_kernel<uint8_t>);
BENCHMARK(bench_agree_subpixel_kernel<uint16_t>);
BENCHMARK(bench_agree_subpixel_kernel_smem<uint8_t>);
BENCHMARK(bench_agree_subpixel_kernel_smem<uint16_t>);

BENCHMARK(bench_bicos_kernel<uint32_t>);
BENCHMARK(bench_bicos_kernel<uint64_t>);
BENCHMARK(bench_bicos_kernel<uint128_t>);
BENCHMARK(bench_bicos_kernel_smem<uint32_t>);
BENCHMARK(bench_bicos_kernel_smem<uint64_t>);
BENCHMARK(bench_bicos_kernel_smem<uint128_t>);

BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint32_t, TransformMode::LIMITED>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint32_t, TransformMode::LIMITED>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint64_t, TransformMode::LIMITED>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint64_t, TransformMode::LIMITED>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint128_t, TransformMode::LIMITED>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint128_t, TransformMode::LIMITED>);

BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint32_t, TransformMode::FULL>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint32_t, TransformMode::FULL>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint64_t, TransformMode::FULL>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint64_t, TransformMode::FULL>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint128_t, TransformMode::FULL>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint128_t, TransformMode::FULL>);

BENCHMARK(bench_integration)
    ->ArgsProduct({
        { 2, 8, 14, 20 }, // n
        { 0, 25, 20, 15, 10 } // step * 100
    });

BENCHMARK_MAIN();
