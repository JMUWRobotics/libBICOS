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

#include "common.cuh"
#include "common.hpp"
#include "fileutils.hpp"
#include "impl/cpu/bicos.hpp"
#include "impl/cpu/descriptor_transform.hpp"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <opencv2/core/cuda_types.hpp>

using namespace BICOS;
using namespace test;
using namespace impl;

int main(int argc, char const* const* argv) {
    std::vector<SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> _ldev, _rdev;
    std::vector<cv::cuda::PtrStepSz<uint8_t>> dev;

    read_sequence(argv[1], std::nullopt, lseq, rseq, true);
    sort_sequence_to_stack(lseq, rseq, lhost, rhost);
    matvec_to_gpu(lhost, rhost, _ldev, _rdev);

    const cv::Size sz = lhost.front().size();
    const size_t n = lhost.size();

    for (size_t i = 0; i < n; ++i) {
        dev.push_back(_ldev[i]);
    }
    for (size_t i = 0; i < n; ++i) {
        dev.push_back(_rdev[i]);
    }

    cv::Mat_<int16_t> raw_gpu_host, raw_host;

    cuda::RegisteredPtr dptr(dev.data(), 2 * n, true);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, sz);

    impl::cuda::StepBuf<uint128_t> lddev(sz), rddev(sz);

    cuda::RegisteredPtr ldptr(&lddev), rdptr(&rddev);

    cudaStream_t lstream, rstream, mainstream;
    cudaStreamCreate(&lstream);
    cudaStreamCreate(&rstream);
    cudaStreamCreate(&mainstream);

    cudaEvent_t ldescev, rdescev;
    cudaEventCreate(&ldescev);
    cudaEventCreate(&rdescev);
    impl::cuda::descriptor_transform_kernel<uint8_t, uint128_t>
        <<<grid, block, 0, lstream>>>(dptr, n, sz, ldptr);
    impl::cuda::descriptor_transform_kernel<uint8_t, uint128_t>
        <<<grid, block, 0, rstream>>>(dptr + n, n, sz, rdptr);

    assertCudaSuccess(cudaGetLastError());

    cudaEventRecord(ldescev, lstream);
    cudaEventRecord(rdescev, rstream);

    cudaStreamWaitEvent(mainstream, ldescev);
    cudaStreamWaitEvent(mainstream, rdescev);

    cv::cuda::GpuMat raw_gpu(sz, cv::DataType<int16_t>::type);
    raw_gpu.setTo(INVALID_DISP_<int16_t>);

    size_t smem_size = sz.width * sizeof(uint128_t);

    assertCudaSuccess(cudaFuncSetAttribute(
        impl::cuda::bicos_kernel<uint128_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    impl::cuda::bicos_kernel<uint128_t>
        <<<grid, block, smem_size, mainstream>>>(ldptr, rdptr, raw_gpu);

    assertCudaSuccess(cudaGetLastError());

    cv::Mat lhin, rhin;

    cv::merge(lhost, lhin);
    cv::merge(rhost, rhin);

    auto ldhost = impl::cpu::descriptor_transform<uint8_t, uint128_t>(
             lhin,
             sz,
             n,
             TransformMode::LIMITED
         ),
         rdhost = impl::cpu::descriptor_transform<uint8_t, uint128_t>(
             rhin,
             sz,
             n,
             TransformMode::LIMITED
         );

    raw_host = impl::cpu::bicos(ldhost, rdhost, sz);

    cudaStreamSynchronize(mainstream);

    raw_gpu.download(raw_gpu_host);

    if (!equals(raw_host, raw_gpu_host))
        return 1;

    return 0;
}