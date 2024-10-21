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
#include "impl/cuda/agree.cuh"
#include "impl/cuda/cutil.cuh"

#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(void) {
    int n = 15;

    const cv::Size randsize(randint(512, 2048), randint(256, 1024));

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> devinput;

    for (int i = 0; i < 2 * n; ++i) {
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());

        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }

    const cuda::RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::Mat_<int16_t> randdisp(randsize);
    cv::randu(randdisp, -1, randsize.width);

    cv::cuda::GpuMat randdisp_dev;
    randdisp_dev.upload(randdisp);

    dim3 block, grid;

    double thresh = randreal(-0.9, 0.9);
    double minvar = randreal(0.1, 10.0);

    cv::cuda::GpuMat devout_gmem(randsize, cv::DataType<disparity_t>::type),
        devout_smem(randsize, cv::DataType<disparity_t>::type);

    devout_gmem.setTo(INVALID_DISP);
    devout_smem.setTo(INVALID_DISP);

    size_t smem_size = randsize.width * n * sizeof(INPUT_TYPE);

    float step = 0.25f;

    auto kernel = cuda::agree_subpixel_kernel<INPUT_TYPE, double, cuda::NXCVariant::MINVAR>;

    block = cuda::max_blocksize(kernel);
    grid = create_grid(block, randsize);

    kernel<<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, minvar, devout_gmem);
    assertCudaSuccess(cudaGetLastError());

    kernel = cuda::agree_subpixel_kernel_smem<INPUT_TYPE, double, cuda::NXCVariant::MINVAR>;

    bool smem_fits = cudaSuccess == cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    if (!smem_fits)
        return 77; // skip, see https://mesonbuild.com/Unit-tests.html#skipped-tests-and-hard-errors

    block = cuda::max_blocksize(kernel);
    grid = create_grid(block, randsize);

    kernel<<<grid, block, smem_size>>>(randdisp_dev, devptr, n, thresh, step, minvar, devout_smem);
    assertCudaSuccess(cudaGetLastError());

    cv::Mat_<disparity_t> gmem, smem;
    devout_gmem.download(gmem);
    devout_smem.download(smem);

    double err = maxerr(gmem, smem);

    std::cout << "max-err: " << err << std::endl;
    if (err > 2.0) {
        return 1;
    }

    return 0;

    return 0;
}
