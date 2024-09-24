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
#include "impl/cpu/bicos.hpp"
#include "impl/cuda/bicos.cuh"
#include "impl/cuda/cutil.cuh"

#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(void) {
    const cv::Size randsize(randint(256, 1028), randint(128, 512));

    auto ld = std::make_unique<cpu::StepBuf<DESCRIPTOR_TYPE>>(randsize),
         rd = std::make_unique<cpu::StepBuf<DESCRIPTOR_TYPE>>(randsize);

    randomize(*ld);
    randomize(*rd);

    auto ld_dev = std::make_unique<cuda::StepBuf<DESCRIPTOR_TYPE>>(*ld),
         rd_dev = std::make_unique<cuda::StepBuf<DESCRIPTOR_TYPE>>(*rd);

    cuda::RegisteredPtr lptr(ld_dev.get(), 1, true), rptr(rd_dev.get(), 1, true);

    cv::cuda::GpuMat disp_dev(randsize, cv::DataType<int16_t>::type);
    disp_dev.setTo(INVALID_DISP_<int16_t>);

    size_t smem_size = randsize.width * sizeof(DESCRIPTOR_TYPE);

    dim3 block = cuda::max_blocksize(cuda::bicos_kernel<DESCRIPTOR_TYPE>, smem_size);
    dim3 grid = create_grid(block, randsize);

    cuda::bicos_kernel<DESCRIPTOR_TYPE><<<grid, block, smem_size>>>(lptr, rptr, disp_dev);

    assertCudaSuccess(cudaGetLastError());

    cv::Mat1s disp = cpu::bicos(ld, rd, randsize);

    assertCudaSuccess(cudaDeviceSynchronize());

    cv::Mat1s devout;
    disp_dev.download(devout);

    if (!equals(disp, devout)) {
        return 1;
    }

    return 0;
}