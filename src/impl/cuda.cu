#include "config.hpp"
#include "cuda.hpp"

#include "impl/cuda/agree.cuh"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace BICOS::impl::cuda {

dim3 create_grid(dim3 block, cv::Size sz) {
    return dim3(
        cv::cuda::device::divUp(sz.width, block.x),
        cv::cuda::device::divUp(sz.height, block.y)
    );
}

template<typename TInput, typename TDescriptor>
static void match_impl(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    size_t n_images,
    cv::Size sz,
    double nxcorr_threshold,
    std::optional<float> subpixel_step,
    cv::cuda::GpuMat& out,
    cv::cuda::Stream& _stream
) {
    std::vector<cv::cuda::PtrStepSz<TInput>> ptrs_host(2 * n_images);

    for (size_t i = 0; i < n_images; ++i) {
        ptrs_host[i] = _stack0[i];
        ptrs_host[i + n_images] = _stack1[i];
    }

    StepBuf<TDescriptor> descr0(sz), descr1(sz);

    size_t smem_size;
    dim3 block(1024);
    dim3 grid = create_grid(block, sz);

    cudaStream_t mainstream = cv::cuda::StreamAccessor::getStream(_stream);

    /* descriptor transform */

    cudaStream_t substream0, substream1;
    cudaStreamCreate(&substream0);
    cudaStreamCreate(&substream1);

    cudaEvent_t event0, event1;
    cudaEventCreate(&event0);
    cudaEventCreate(&event1);

    RegisteredPtr ptrs_dev(ptrs_host.data(), 2 * n_images, true);
    RegisteredPtr descr0_dev(&descr0), descr1_dev(&descr1);

    descriptor_transform_kernel<TInput, TDescriptor>
        <<<grid, block, 0, substream0>>>(ptrs_dev, n_images, sz, descr0_dev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaEventRecord(event0, substream0));

    descriptor_transform_kernel<TInput, TDescriptor>
        <<<grid, block, 0, substream1>>>(ptrs_dev + n_images, n_images, sz, descr1_dev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaEventRecord(event1, substream1));

    cudaSafeCall(cudaStreamWaitEvent(mainstream, event0));
    cudaSafeCall(cudaStreamWaitEvent(mainstream, event1));

    /* bicos disparity */

    cv::cuda::GpuMat bicos_disp(sz, cv::DataType<int16_t>::type);
    bicos_disp.setTo(INVALID_DISP_<int16_t>);

    smem_size = sz.width * sizeof(TDescriptor);

    cudaSafeCall(cudaFuncSetAttribute(
        bicos_kernel<TDescriptor>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    bicos_kernel<TDescriptor>
        <<<grid, block, smem_size, mainstream>>>(descr0_dev, descr1_dev, bicos_disp);
    cudaSafeCall(cudaGetLastError());

    /* nxcorr */

    out.create(sz, cv::DataType<disparity_t>::type);
    out.setTo(INVALID_DISP);

    // smem_size = sz.width * n_images * sizeof(TInput);

    block = dim3(768);
    grid = create_grid(block, sz);

    if (subpixel_step.has_value()) {
        cudaSafeCall(cudaDeviceSetLimit(
            cudaLimitStackSize,
            1024 + 3 * n_images * (sizeof(TInput) + sizeof(float))
        ));
        agree_subpixel_kernel<TInput><<<grid, block, 0, mainstream>>>(
            bicos_disp,
            ptrs_dev,
            n_images,
            nxcorr_threshold,
            subpixel_step.value(),
            out
        );
    } else {
        cudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, 1024 + 2 * n_images * sizeof(TInput)));
        agree_kernel<TInput>
            <<<grid, block, 0, mainstream>>>(bicos_disp, ptrs_dev, n_images, nxcorr_threshold, out);
    }

    cudaSafeCall(cudaGetLastError());
}

void match(
    const std::vector<cv::cuda::GpuMat>& _stack0,
    const std::vector<cv::cuda::GpuMat>& _stack1,
    cv::cuda::GpuMat& disparity,
    Config cfg,
    cv::cuda::Stream& stream
) {
    const size_t n_images = _stack0.size();
    const int depth = _stack0.front().depth();
    const cv::Size sz = _stack0.front().size();

    int required_bits = cfg.mode == TransformMode::FULL
        ? throw std::invalid_argument("unimplemented")
        : 4 * n_images - 7;

    switch (required_bits) {
        case 0 ... 32:
            if (depth == CV_8U)
                match_impl<uint8_t, uint32_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            else
                match_impl<uint16_t, uint32_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            break;
        case 33 ... 64:
            if (depth == CV_8U)
                match_impl<uint8_t, uint64_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            else
                match_impl<uint16_t, uint64_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            break;
        case 65 ... 128:
            if (depth == CV_8U)
                match_impl<uint8_t, uint128_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            else
                match_impl<uint16_t, uint128_t>(
                    _stack0,
                    _stack1,
                    n_images,
                    sz,
                    cfg.nxcorr_thresh,
                    cfg.subpixel_step,
                    disparity,
                    stream
                );
            break;
        default:
            throw std::invalid_argument("input stacks too large, exceeding 128 bits");
    }
}

} // namespace BICOS::impl::cuda