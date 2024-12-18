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

#include "impl/common.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace BICOS::impl::cuda {

template<typename TDescriptor>
static auto select_bicos_kernel(SearchVariant variant, bool smem) {
    if (std::holds_alternative<Variant::Consistency>(variant)) {
        auto consistency = std::get<Variant::Consistency>(variant);
        if (consistency.no_dupes)
            return smem
                ? bicos_kernel_smem<TDescriptor, BICOSFLAGS_CONSISTENCY | BICOSFLAGS_NODUPES>
                : bicos_kernel<TDescriptor, BICOSFLAGS_CONSISTENCY | BICOSFLAGS_NODUPES>;
        else
            return smem ? bicos_kernel_smem<TDescriptor, BICOSFLAGS_CONSISTENCY>
                        : bicos_kernel<TDescriptor, BICOSFLAGS_CONSISTENCY>;
    } else if (std::holds_alternative<Variant::NoDuplicates>(variant))
        return smem ? bicos_kernel_smem<TDescriptor, BICOSFLAGS_NODUPES>
                    : bicos_kernel<TDescriptor, BICOSFLAGS_NODUPES>;

    throw std::invalid_argument("unimplemented");
}

template<typename TInput, typename TDescriptor>
static void match_impl(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    size_t n_images,
    cv::Size sz,
    std::optional<float> min_nxc,
    Precision precision,
    TransformMode mode,
    std::optional<float> subpixel_step,
    std::optional<float> min_var,
    SearchVariant variant,
    cv::cuda::GpuMat& out,
    cv::cuda::GpuMat* corrmap,
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

    cv::cuda::GpuMat bicos_disp;
    if (out.type() == cv::DataType<int16_t>::type) {
        // input buffer is probably output from previous,
        // non-subpixel call to match()
        // we can reuse that
        bicos_disp = out;
    }

    init_disparity<int16_t>(bicos_disp, sz, _stream);

    auto kernel = select_bicos_kernel<TDescriptor>(variant, true);
    int lr_max_diff = std::holds_alternative<Variant::Consistency>(variant)
        ? std::get<Variant::Consistency>(variant).max_lr_diff
        : -1;

    size_t smem_size = sz.width * sizeof(TDescriptor);
    bool bicos_smem_fits = cudaSuccess
        == cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaGetLastError(); // clear potential error from previous call to cudaFuncSetAttribute

    if (bicos_smem_fits) {
        block = max_blocksize(kernel, smem_size);
        grid = create_grid(block, sz);

        kernel<<<grid, block, smem_size, mainstream>>>(
            descr0_dev,
            descr1_dev,
            lr_max_diff,
            bicos_disp
        );
    } else {
        kernel = select_bicos_kernel<TDescriptor>(variant, false);

        block = max_blocksize(kernel);
        grid = create_grid(block, sz);

        kernel<<<grid, block, 0, mainstream>>>(descr0_dev, descr1_dev, lr_max_diff, bicos_disp);
    }
    assertCudaSuccess(cudaGetLastError());

    // optimized for subpixel interpolation.
    // keep `out` as output for the interpolated
    // depth map.

    if (!min_nxc.has_value()) {
        out = bicos_disp;
        return;
    }

    /* nxcorr */

    if (corrmap)
        corrmap->create(sz, precision == Precision::SINGLE ? CV_32FC1 : CV_64FC1);

    // clang-format off
    // :(

    auto invoke = [&](auto kernel, auto&&... args) {
        auto block = max_blocksize(kernel);
        auto grid  = create_grid(block, sz);
        kernel<<<grid, block, 0, mainstream>>>(std::forward<decltype(args)>(args)...);
    };

    if (subpixel_step.has_value()) {
        init_disparity<float>(out, sz);
        if (corrmap) {
            if (min_var.has_value()) {
                if (precision == Precision::SINGLE)
                    invoke(agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, subpixel_step.value(), out, *corrmap);
                else
                    invoke(agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, subpixel_step.value(), out, *corrmap);
            } else {
                if (precision == Precision::SINGLE)
                    invoke(agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0f, subpixel_step.value(), out, *corrmap);
                else
                    invoke(agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0, subpixel_step.value(), out, *corrmap);
            }
        } else {
            if (min_var.has_value()) {
                if (precision == Precision::SINGLE)
                    invoke(agree_subpixel_kernel<TInput, float, NXCVariant::MINVAR, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, subpixel_step.value(), out, cv::cuda::PtrStepSz<float>());
                else
                    invoke(agree_subpixel_kernel<TInput, double, NXCVariant::MINVAR, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, subpixel_step.value(), out, cv::cuda::PtrStepSz<double>());
            } else {
                if (precision == Precision::SINGLE)
                    invoke(agree_subpixel_kernel<TInput, float, NXCVariant::PLAIN, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0f, subpixel_step.value(), out, cv::cuda::PtrStepSz<float>());
                else
                    invoke(agree_subpixel_kernel<TInput, double, NXCVariant::PLAIN, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0, subpixel_step.value(), out, cv::cuda::PtrStepSz<double>());
            }
        }
    } else {
        // working on bicos_disp
        if (corrmap) {
            if (min_var.has_value()) {
                if (precision == Precision::SINGLE)
                    invoke(agree_kernel<TInput, float, NXCVariant::MINVAR, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, *corrmap);
                else
                    invoke(agree_kernel<TInput, double, NXCVariant::MINVAR, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, *corrmap);
            } else {
                if (precision == Precision::SINGLE)
                    invoke(agree_kernel<TInput, float, NXCVariant::PLAIN, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0f, *corrmap);
                else
                    invoke(agree_kernel<TInput, double, NXCVariant::PLAIN, true>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0, *corrmap);
            }
        } else {
            if (min_var.has_value()) {
                if (precision == Precision::SINGLE)
                    invoke(agree_kernel<TInput, float, NXCVariant::MINVAR, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, cv::cuda::PtrStepSz<float>());
                else
                    invoke(agree_kernel<TInput, double, NXCVariant::MINVAR, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), min_var.value() * n_images, cv::cuda::PtrStepSz<double>());
            } else {
                if (precision == Precision::SINGLE)
                    invoke(agree_kernel<TInput, float, NXCVariant::PLAIN, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0f, cv::cuda::PtrStepSz<float>());
                else
                    invoke(agree_kernel<TInput, double, NXCVariant::PLAIN, false>, bicos_disp, ptrs_dev, n_images, min_nxc.value(), 0.0, cv::cuda::PtrStepSz<double>());
            }
        }
        out = bicos_disp;
    }

    // clang-format on

    assertCudaSuccess(cudaGetLastError());
}

void match(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::GpuMat* corrmap,
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
                match_impl<uint8_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint32_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint64_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
#ifdef BICOS_CUDA_HAS_UINT128
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, uint128_t>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
        case 129 ... 256:
#else
        case 65 ... 256:
#endif
            if (depth == CV_8U)
                match_impl<uint8_t, varuint_<256>>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            else
                match_impl<uint16_t, varuint_<256>>(_stack0, _stack1, n, sz, cfg.nxcorr_threshold, cfg.precision, cfg.mode, cfg.subpixel_step, cfg.min_variance, cfg.variant, disparity, corrmap, stream);
            break;
            break;
        default:
            throw std::invalid_argument(fmt::format("input stacks too large, would require {} bits", required_bits));
    }

    // clang-format on
}

} // namespace BICOS::impl::cuda