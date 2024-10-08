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
#include "cuda.hpp"

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
    std::optional<double> _min_var,
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

    size_t smem_size = sz.width * sizeof(TDescriptor);

    assertCudaSuccess(cudaFuncSetAttribute(
        bicos_kernel<TDescriptor>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    block = max_blocksize(bicos_kernel<TDescriptor>, smem_size);
    grid  = create_grid(block, sz);

    bicos_kernel<TDescriptor>
        <<<grid, block, smem_size, mainstream>>>(descr0_dev, descr1_dev, bicos_disp);
    assertCudaSuccess(cudaGetLastError());

    /* nxcorr */

    out.create(sz, cv::DataType<disparity_t>::type);
    out.setTo(INVALID_DISP, _stream);

    // clang-format off

    // TODO: switch to flags and LUT for function calls

    double min_var = _min_var.value_or(1.0) * n_images;

    switch (precision) {
        case Precision::SINGLE:
            if (subpixel_step.has_value()) {
                if (_min_var.has_value()) {
                    block = max_blocksize(agree_subpixel_kernel<TInput, float, nxcorrf_minvar>);
                    grid = create_grid(block, sz);
                    agree_subpixel_kernel<TInput, float, nxcorrf_minvar>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value(), min_var, out);
                } else {
                    block = max_blocksize(agree_subpixel_kernel<TInput, float, nxcorrf>);
                    grid = create_grid(block, sz);
                    agree_subpixel_kernel<TInput, float, nxcorrf>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value(), min_var, out);
                }
            } else {
                if (_min_var.has_value()) {
                    block = max_blocksize(agree_kernel<TInput, float, nxcorrf_minvar>);
                    grid = create_grid(block, sz);
                    agree_kernel<TInput, float, nxcorrf_minvar>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, min_var, out);
                } else {
                    block = max_blocksize(agree_kernel<TInput, float, nxcorrf>);
                    grid = create_grid(block, sz);
                    agree_kernel<TInput, float, nxcorrf>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, min_var, out);
                }
            } break;
        case Precision::DOUBLE:
            if (subpixel_step.has_value()) {
                if (_min_var.has_value()) {
                    block = max_blocksize(agree_subpixel_kernel<TInput, double, nxcorrd_minvar>);
                    grid = create_grid(block, sz);
                    agree_subpixel_kernel<TInput, double, nxcorrd_minvar>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value(), min_var, out);
                } else {
                    block = max_blocksize(agree_subpixel_kernel<TInput, double, nxcorrd>);
                    grid = create_grid(block, sz);
                    agree_subpixel_kernel<TInput, double, nxcorrd>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, subpixel_step.value(), min_var, out);
                }
            } else {
                if (_min_var.has_value()) {
                    block = max_blocksize(agree_kernel<TInput, double, nxcorrd_minvar>);
                    grid = create_grid(block, sz);
                    agree_kernel<TInput, double, nxcorrd_minvar>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, min_var, out);
                } else {
                    block = max_blocksize(agree_kernel<TInput, double, nxcorrd>);
                    grid = create_grid(block, sz);
                    agree_kernel<TInput, double, nxcorrd>
                        <<<grid, block, 0, mainstream>>>(
                            bicos_disp, ptrs_dev, n_images, nxcorr_threshold, min_var, out);
                }
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

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_impl<uint8_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            else
                match_impl<uint16_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            else
                match_impl<uint16_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            break;
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            else
                match_impl<uint16_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_thresh, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, disparity, stream);
            break;
        default:
            throw std::invalid_argument("input stacks too large, exceeding 128 bits");
    }

    // clang-format on
}

} // namespace BICOS::impl::cuda