#include "common.hpp"
#include "impl/cpu/agree.hpp"
#include "impl/cuda/agree.cuh"
#include "impl/cuda/cutil.cuh"
#include "opencv2/core.hpp"
#include "opencv2/core/traits.hpp"
#include "common.cuh"
#include "opencv2/core/types.hpp"

#include <limits>
#include <opencv2/core/cuda.hpp>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(void) {
    int n = 10;

    const cv::Size randsize(randint(1024, 4096), randint(512, 2048));

    std::vector<cv::cuda::GpuMat> _devinput;
    std::vector<cv::Mat> _hostinput;
    std::vector<cv::cuda::PtrStepSz<INPUT_TYPE>> devinput;
    cv::Mat hinput_l, hinput_r;

    for (int i = 0; i < 2 * n; ++i) {
        if (i == n) {
            cv::merge(_hostinput, hinput_l);
            _hostinput.clear();
        }
        cv::Mat_<INPUT_TYPE> randmat(randsize);
        cv::randu(randmat, 0, std::numeric_limits<INPUT_TYPE>::max());

        _hostinput.push_back(randmat);
        cv::cuda::GpuMat randmat_dev(randmat);

        _devinput.push_back(randmat_dev);
        devinput.push_back(randmat_dev);
    }
    cv::merge(_hostinput, hinput_r);

    RegisteredPtr devptr(devinput.data(), 2 * n, true);

    cv::Mat_<int16_t> randdisp(randsize);
    cv::randu(randdisp, -1, randsize.width);

    {
        double min, max;
        cv::minMaxIdx(randdisp, &min, &max);
        assert(min == -1.0);
    }

    cv::cuda::GpuMat randdisp_dev;
    randdisp_dev.upload(randdisp);

    const dim3 block(768);
    const dim3 grid = create_grid(block, randsize);

    double thresh = randreal(-0.9, 0.9);

    cv::cuda::GpuMat devout(randsize, cv::DataType<disparity_t>::type);
    devout.setTo(INVALID_DISP);
    cv::Mat_<disparity_t> hostout(randsize), devout_host;

#if TEST_SUBPIXEL

    float step = 0.25f;

    cuda::agree_subpixel_kernel<INPUT_TYPE>
        <<<grid, block>>>(randdisp_dev, devptr, n, thresh, step, devout);

    cudaSafeCall(cudaGetLastError());

    cpu::agree_subpixel<INPUT_TYPE>(randdisp, hinput_l, hinput_r, n, thresh, step, hostout);

    cudaSafeCall(cudaDeviceSynchronize());

    devout.download(devout_host);

    const float magic = -10000.0f;
    cv::patchNaNs(hostout, magic);
    cv::patchNaNs(devout_host, magic);

    hostout.setTo(0.0f, devout_host == magic);
    devout_host.setTo(0.0f, hostout == magic);
    hostout.setTo(0.0f, hostout == magic);
    devout_host.setTo(0.0f, devout_host == magic);

    double maxerr;

    cv::Mat absd;
    cv::absdiff(hostout, devout_host, absd);
    cv::minMaxIdx(absd, NULL, &maxerr);

    // TODO investigate why agree_subpixel fails on random input data

    std::cout << "max-err: " << maxerr << std::endl;
    if (maxerr > 2.0) {
        return 1;
    }

    return 0;

#else

    cuda::agree_kernel<INPUT_TYPE><<<grid, block>>>(randdisp_dev, devptr, n, thresh, devout);

    cudaSafeCall(cudaGetLastError());

    cpu::agree<INPUT_TYPE>(randdisp, hinput_l, hinput_r, n, thresh, hostout);

    cudaSafeCall(cudaDeviceSynchronize());

    devout.download(devout_host);

    if (!equals(hostout, devout_host))
        return 1;

    return 0;

#endif
}