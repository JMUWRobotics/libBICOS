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
#include "cpu.hpp"
#include "cuda.hpp"
#include "fileutils.hpp"

#include <iostream>
#include <optional>

using namespace BICOS;
using namespace impl;
using namespace test;

int main(int argc, char const* const* argv) {
    std::vector<SequenceEntry> lseq, rseq;
    std::vector<cv::Mat> lhost, rhost;
    std::vector<cv::cuda::GpuMat> ldev, rdev;

    read_sequence(argv[1], argv[2], lseq, rseq, true);
    sort_sequence_to_stack(lseq, rseq, lhost, rhost);
    matvec_to_gpu(lhost, rhost, ldev, rdev);

    for (double thresh: { 0.5, 0.75, 0.9 }) {
        Config cfg { .nxcorr_thresh = thresh,
                     .subpixel_step = std::nullopt,
                     .mode = TransformMode::LIMITED,
                     .precision = Precision::DOUBLE };

        cv::Mat_<disparity_t> dhost, ddev_host;
        cv::cuda::GpuMat ddev;

        cv::cuda::Stream stream;
        impl::cuda::match(ldev, rdev, ddev, cfg, stream);
        ddev.download(ddev_host, stream);

        impl::cpu::match(lhost, rhost, dhost, cfg);
        stream.waitForCompletion();

        if (!equals(dhost, ddev_host)) {
            std::cerr << "thresh: " << thresh << std::endl;
            return 1;
        }

        for (float step: { 0.1f, 0.25f, 0.5f }) {
            cfg.subpixel_step = step;

            impl::cuda::match(ldev, rdev, ddev, cfg, stream);
            ddev.download(ddev_host, stream);

            impl::cpu::match(lhost, rhost, dhost, cfg);
            stream.waitForCompletion();

            if (!equals(dhost, ddev_host)) {
                std::cerr << "thresh: " << thresh << " step: " << step << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
