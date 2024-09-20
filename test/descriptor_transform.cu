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
#include "impl/cpu/descriptor_transform.hpp"
#include "impl/cuda/cutil.cuh"
#include "impl/cuda/descriptor_transform.cuh"

#include <iostream>
#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

#define _STR(s) #s
#define STR(s) _STR(s)

int main(void) {
    cv::Mat hoststack;
    std::vector<cv::Mat_<INPUT_TYPE>> rand_host;
    std::vector<cv::cuda::GpuMat> _rand_dev;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> rand_dev;

    const cv::Size randsize(randint(256, 1028), randint(128, 512));

    std::cout << "descriptor transform on " << randsize << " " << STR(INPUT_TYPE) << " "
              << STR(DESCRIPTOR_TYPE) << std::endl;

    int max_bits = sizeof(DESCRIPTOR_TYPE) * 8;
    size_t n = (max_bits + 7) / 4;

    for (size_t i = 0; i < n; ++i) {
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());
        rand_host.push_back(randmat);

        cv::cuda::GpuMat randmat_dev(randmat);
        _rand_dev.push_back(randmat_dev);
        rand_dev.push_back(randmat_dev);
    }

    cuda::RegisteredPtr rand_devptr(rand_dev.data(), n, true);

    cuda::StepBuf<DESCRIPTOR_TYPE> gpuout(randsize);
    cuda::RegisteredPtr gpuout_devptr(&gpuout);

    const dim3 block(1024);
    const dim3 grid = create_grid(block, randsize);

    cuda::descriptor_transform_kernel<INPUT_TYPE, DESCRIPTOR_TYPE>
        <<<grid, block>>>(rand_devptr, n, randsize, gpuout_devptr);

    assertCudaSuccess(cudaGetLastError());

    cv::merge(rand_host, hoststack);

    auto cpuout = cpu::descriptor_transform<INPUT_TYPE, DESCRIPTOR_TYPE>(
        hoststack,
        randsize,
        n,
        TransformMode::LIMITED
    );

    assertCudaSuccess(cudaDeviceSynchronize());

    cpu::StepBuf<DESCRIPTOR_TYPE> gpuout_host(gpuout);

    if (!equals(*cpuout, gpuout_host, randsize))
        return 1;

    return 0;
}
