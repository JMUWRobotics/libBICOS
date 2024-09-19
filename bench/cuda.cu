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

constexpr int n = 5;
constexpr double thresh = 0.9;
constexpr float step = 0.25;
static const cv::Size size(3300, 2200);

template <typename TPrecision, cuda::corrfun<uint8_t, TPrecision> FCorr>
__global__ void nxcorr_kernel(const uint8_t* a, const uint8_t* b, size_t n, TPrecision* out) {
    *out = FCorr(a, b, n);
}

template <typename TPrecision, cuda::corrfun<uint8_t, TPrecision> FCorr>
void bench_nxcorr_subroutine(benchmark::State& state) {
    uint8_t _a[50], _b[50], *a, *b;

    for (size_t i = 0; i < sizeof(_a); ++i) {
        _a[i] = rand();
        _b[i] = rand();
    }

    cudaMalloc(&a, sizeof(_a));
    cudaMalloc(&b, sizeof(_b));

    cudaMemcpy(a, _a, sizeof(_a), cudaMemcpyHostToDevice);
    cudaMemcpy(b, _b, sizeof(_b), cudaMemcpyHostToDevice);

    TPrecision *out;
    cudaMalloc(&out, 1);

    for (auto _ : state) {
        nxcorr_kernel<TPrecision, FCorr><<<1,1>>>(a, b, sizeof(_a), out);
        cudaDeviceSynchronize();
    }
}

template<typename TInput>
void bench_agree_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);

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

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::agree_kernel<TInput, double, cuda::nxcorrd><<<grid, block>>>(randdisp_dev, devptr, n, thresh, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput>
void bench_agree_subpixel_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);

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

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::agree_subpixel_kernel<TInput, double, cuda::nxcorrd>
            <<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput>
void bench_agree_subpixel_kernel_smem(benchmark::State& state) {
    cv::setRNGSeed(seed);

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

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::cuda::GpuMat out(size, cv::DataType<disparity_t>::type);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, size);

    size_t smem_size = size.width * n * sizeof(TInput);

    assertCudaSuccess(cudaFuncSetAttribute(
        cuda::agree_subpixel_kernel_smem<TInput, double, cuda::nxcorrd>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    for (auto _: state) {
        cuda::agree_subpixel_kernel_smem<TInput, double, cuda::nxcorrd>
            <<<grid, block, smem_size>>>(randdisp_dev, devptr, n, thresh, step, out);
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

    RegisteredPtr lptr(&ld_dev, 1, true), rptr(&rd_dev, 1, true);

    cv::cuda::GpuMat out(size, cv::DataType<int16_t>::type);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, size);

    size_t smem_size = size.width * sizeof(TDescriptor);

    assertCudaSuccess(cudaFuncSetAttribute(
        cuda::bicos_kernel<TDescriptor>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    for (auto _: state) {
        cuda::bicos_kernel<TDescriptor><<<grid, block, smem_size>>>(lptr, rptr, out);
        cudaDeviceSynchronize();
    }

    assertCudaSuccess(cudaGetLastError());
}

template<typename TInput, typename TDescriptor>
void bench_descriptor_transform_kernel(benchmark::State& state) {
    cv::setRNGSeed(seed);

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<TInput>> devinput;

    for (int i = 0; i < n; ++i) {
        cv::Mat_<TInput> randmat(size);
        cv::randu(randmat, 0, std::numeric_limits<TInput>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    RegisteredPtr inptr(devinput.data(), n, true);

    cuda::StepBuf<TDescriptor> out(size);
    RegisteredPtr outptr(&out);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, size);

    for (auto _: state) {
        cuda::descriptor_transform_kernel<TInput, TDescriptor>
            <<<grid, block>>>(inptr, n, size, outptr);
    }

    assertCudaSuccess(cudaGetLastError());
}

void bench_integration(benchmark::State& state) {
    std::vector<SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> ldev, rdev;

    read_sequence(SOURCE_ROOT "/data", std::nullopt, lseq, rseq, true);
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

BENCHMARK(bench_nxcorr_subroutine<float, cuda::nxcorrf>)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(bench_nxcorr_subroutine<double, cuda::nxcorrd>)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK(bench_agree_kernel<uint8_t>);
BENCHMARK(bench_agree_kernel<uint16_t>);
BENCHMARK(bench_agree_subpixel_kernel<uint8_t>);
BENCHMARK(bench_agree_subpixel_kernel<uint16_t>);
BENCHMARK(bench_agree_subpixel_kernel_smem<uint8_t>);
BENCHMARK(bench_agree_subpixel_kernel_smem<uint16_t>);
BENCHMARK(bench_bicos_kernel<uint32_t>);
BENCHMARK(bench_bicos_kernel<uint64_t>);
BENCHMARK(bench_bicos_kernel<uint128_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint32_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint32_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint64_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint64_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint8_t, uint128_t>);
BENCHMARK(bench_descriptor_transform_kernel<uint16_t, uint128_t>);

BENCHMARK(bench_integration)
    ->ArgsProduct({
        { 2, 8, 14, 20 }, // n
        { 0, 25, 20, 15, 10 } // step * 100
    });

BENCHMARK_MAIN();
