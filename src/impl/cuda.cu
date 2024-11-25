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

#include "common.hpp"
#include "compat.hpp"
#include "cuda.hpp"

#include "impl/common.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace BICOS::impl::cuda {

template<typename TInput, typename TDescriptor>
static void match_impl(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    size_t n_images,
    cv::Size sz,
    double nxcorr_threshold,
    Precision precision,
    TransformMode mode,
    std::optional<float> subpixel_step,
    std::optional<double> min_var,
    std::optional<int> lr_max_diff,
    cv::cuda::GpuMat& out,
    cv::cuda::Stream& _stream
) {
    std::vector<cv::cuda::PtrStepSz<TInput>> ptrs_host(2 * n_images);

    for (size_t i = 0; i < n_images; ++i) {
        ptrs_host[i] = _stack0[i];
        ptrs_host[i + n_images] = _stack1[i];
    }

    StepBuf<TDescriptor> descr0(sz), descr1(sz);

    cudaStream_t mainstream = cv::cuda::StreamAccessor::getStream(_stream);

    /* descriptor transform */

    cudaStream_t substream0, substream1;
    assertCudaSuccess(cudaStreamCreate(&substream0));
    assertCudaSuccess(cudaStreamCreate(&substream1));

    cudaEvent_t event0, event1;
    assertCudaSuccess(cudaEventCreate(&event0));
    assertCudaSuccess(cudaEventCreate(&event1));

    RegisteredPtr ptrs_dev(ptrs_host.data(), 2 * n_images, true);
    RegisteredPtr descr0_dev(&descr0), descr1_dev(&descr1);

    dim3 block, grid;

    if (mode == TransformMode::LIMITED) {
        block = max_blocksize(transform_limited_kernel<TInput, TDescriptor>);
        grid = create_grid(block, sz);

        transform_limited_kernel<TInput, TDescriptor>
            <<<grid, block, 0, substream0>>>(ptrs_dev, n_images, sz, descr0_dev);
    } else {
        block = max_blocksize(transform_full_kernel<TInput, TDescriptor>);
        grid = create_grid(block, sz);

        transform_full_kernel<TInput, TDescriptor>
            <<<grid, block, 0, substream0>>>(ptrs_dev, n_images, sz, descr0_dev);
    }

    assertCudaSuccess(cudaGetLastError());
    assertCudaSuccess(cudaEventRecord(event0, substream0));

    if (mode == TransformMode::LIMITED)
        transform_limited_kernel<TInput, TDescriptor>
            <<<grid, block, 0, substream1>>>(ptrs_dev + n_images, n_images, sz, descr1_dev);
    else
        transform_full_kernel<TInput, TDescriptor>
            <<<grid, block, 0, substream1>>>(ptrs_dev + n_images, n_images, sz, descr1_dev);

    assertCudaSuccess(cudaGetLastError());
    assertCudaSuccess(cudaEventRecord(event1, substream1));

    assertCudaSuccess(cudaStreamWaitEvent(mainstream, event0));
    assertCudaSuccess(cudaStreamWaitEvent(mainstream, event1));

    /* bicos disparity */

    cv::cuda::GpuMat bicos_disp(sz, cv::DataType<int16_t>::type);
    bicos_disp.setTo(INVALID_DISP_<int16_t>, _stream);

    auto kernel = lr_max_diff.has_value()
        ? bicos_kernel_smem<TDescriptor, BICOSVariant::WITH_REVERSE>
        : bicos_kernel_smem<TDescriptor, BICOSVariant::DEFAULT>;

    size_t smem_size = sz.width * sizeof(TDescriptor);
    bool bicos_smem_fits = cudaSuccess
        == cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaGetLastError(); // clear potential error from previous call to cudaFuncSetAttribute

    if (bicos_smem_fits) {
        block = max_blocksize(kernel, smem_size);
        grid = create_grid(block, sz);

        kernel<<<grid, block, smem_size, mainstream>>>(descr0_dev, descr1_dev, lr_max_diff.value_or(-1), bicos_disp);
    } else {
        kernel = lr_max_diff.has_value()
            ? bicos_kernel<TDescriptor, BICOSVariant::WITH_REVERSE>
            : bicos_kernel<TDescriptor, BICOSVariant::DEFAULT>;

        block = max_blocksize(kernel);
        grid = create_grid(block, sz);

        kernel<<<grid, block, 0, mainstream>>>(descr0_dev, descr1_dev, lr_max_diff.value_or(-1), bicos_disp);
    }
    assertCudaSuccess(cudaGetLastError());

    /* nxcorr */

    out.create(sz, cv::DataType<disparity_t>::type);
    out.setTo(INVALID_DISP, _stream);

    // clang-format off

    switch (precision) {
    case Precision::SINGLE: {

        static agree_kernel_t<TInput, float> lut[2][2] = {
            { agree_kernel<TInput, float, NXCVariant::PLAIN>, agree_kernel<TInput, float, NXCVariant::MINVAR> },
            { agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN>, agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR> }
        };

        auto kernel = lut[subpixel_step.has_value()][min_var.has_value()];

        block = max_blocksize(kernel);
        grid = create_grid(block, sz);

        kernel<<<grid, block, 0, mainstream>>>(
        bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value_or(0.0f), n_images * min_var.value_or(0.0f), out);

    } break;
    case Precision::DOUBLE: {

        static agree_kernel_t<TInput, double> lut[2][2] = {
            { agree_kernel<TInput, double, NXCVariant::PLAIN>, agree_kernel<TInput, double, NXCVariant::MINVAR> },
            { agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN>, agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR> }
        };

        auto kernel = lut[subpixel_step.has_value()][min_var.has_value()];

        block = max_blocksize(kernel);
        grid = create_grid(block, sz);

        kernel<<<grid, block, 0, mainstream>>>(
        bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value_or(0.0f), n_images * min_var.value_or(0.0), out);

    } break;
    }

    // clang-format on

    assertCudaSuccess(cudaGetLastError());
}

void match(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::Stream& stream
) {
    const size_t n = _stack0.size();
    const int depth = _stack0.front().depth();
    const cv::Size sz = _stack0.front().size();

    // clang-format off

    int required_bits = cfg.mode == TransformMode::FULL
        ? n * n - 2 * n + 3
        : 4 * n - 7;

    std::optional<int> lr_max_diff = std::nullopt;
    if (std::holds_alternative<Variant::WithReverse>(cfg.variant))
        lr_max_diff = std::get<Variant::WithReverse>(cfg.variant).max_lr_diff;

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_impl<uint8_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            else
                match_impl<uint16_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            else
                match_impl<uint16_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            break;
#ifdef BICOS_CUDA_HAS_UINT128
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            else
                match_impl<uint16_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, lr_max_diff, disparity, stream);
            break;
#endif
        default:
            throw std::invalid_argument(BICOS::format("input stacks too large, would require {} bits", required_bits));
    }

    // clang-format on
}

} // namespace BICOS::impl::cuda