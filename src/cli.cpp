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
#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <stdexcept>

#ifdef BICOS_CUDA
    #include <opencv2/core/cuda.hpp>
#endif

#include "fileutils.hpp"
#include "match.hpp"

#define DELTA_MS(name) double delta_##name = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - tick).count() / 1000.0

using namespace BICOS;

#define LICENSE_HEADER "libBICOS  Copyright (C) 2024  Robotics Group @ JMU\n"\
                       "This program is free software, and you are welcome to redistribute\n"\
                       "it under the conditions of the GNU LGPL-3.0-or-later license.\n"\
                       "Refer to https://github.com/JMUWRobotics/libBICOS for details.\n"

int main(int argc, char const* const* argv) {
    cxxopts::Options opts(argv[0], "cli to process images with BICOS");

    // clang-format off

    opts.add_options()
        ("folder0", "First folder containing input images with numbered names", cxxopts::value<std::string>())
        ("folder1", "Optional second folder with input images. If specified, file names need to be 0.png, 1.png... Else, folder0 needs to contain 0_left.png, 0_right.png, 1_left.png...", cxxopts::value<std::string>())
        ("t,threshold", "Normalized cross corellation threshold", cxxopts::value<double>()->default_value("0.5"))
        ("s,step", "Subpixel step (optional)", cxxopts::value<float>())
        ("limited", "Limit transformation mode. Allows for more images to be used.")
        ("o,outfile", "Output file for disparity image", cxxopts::value<std::string>()->default_value("bicosdisp.png"))
        ("n,stacksize", "Number of images to process. Defaults to all.", cxxopts::value<uint>())
        ("q,qmatrix", "Path to cv::FileStorage with single matrix \"Q\" for computing pointcloud", cxxopts::value<std::string>())
#ifdef BICOS_CUDA
        ("single", "Set single instead of double precision")
#endif
        ("h,help", "Display this message");

    opts.parse_positional({"folder0", "folder1"});
    opts.positional_help("folder0 [folder1]");

    auto args = opts.parse(argc, argv);

    // clang-format on

    if (args.count("help")) {
        std::cout << opts.help() << std::endl;
        return 0;
    }

    std::cout << LICENSE_HEADER << std::endl;

    std::filesystem::path folder0 = args["folder0"].as<std::string>();
    std::filesystem::path outfile = args["outfile"].as<std::string>();
    std::optional<std::filesystem::path> folder1 = std::nullopt;
    std::optional<std::filesystem::path> q_store = std::nullopt;

    if (args.count("qmatrix")) {
        q_store = args["qmatrix"].as<std::string>();

        if (!std::filesystem::exists(q_store.value()))
            throw std::invalid_argument(std::format("'{}' does not exist", q_store.value().string()));
    }

    if (args.count("folder1"))
        folder1 = args["folder1"].as<std::string>();

    std::vector<cv::Mat> lstack, rstack;
    {
        std::vector<SequenceEntry> lseq, rseq;

        read_sequence(folder0, folder1, lseq, rseq, true);
        sort_sequence_to_stack(lseq, rseq, lstack, rstack);

        if (args.count("stacksize")) {
            uint n = args["stacksize"].as<uint>();

            if (n < lstack.size()) {
                lstack.resize(n);
                rstack.resize(n);
            }
        }

        if (lstack.size() != rstack.size())
            throw std::invalid_argument(std::format("Left stack: {}, right stack: {} images", lstack.size(), rstack.size()));

        std::cout << "Loaded " << lstack.size() + rstack.size() << " images total\n";
    }

    cv::Mat_<BICOS::disparity_t> disp;

    BICOS::Config c {
        .nxcorr_thresh = args["threshold"].as<double>(),
        .mode = TransformMode::FULL
    };

    if (args.count("step"))
        c.subpixel_step = args["step"].as<float>();
    if (args.count("limited"))
        c.mode = TransformMode::LIMITED;
#ifdef BICOS_CUDA
    if (args.count("single"))
        c.precision = Precision::SINGLE;
#endif

#ifdef BICOS_CUDA

    std::vector<cv::cuda::GpuMat> lstack_gpu, rstack_gpu;

    auto tick = std::chrono::high_resolution_clock::now();

    matvec_to_gpu(lstack, rstack, lstack_gpu, rstack_gpu);

    DELTA_MS(upload);

    std::cout << "Latency:\t" << delta_upload << "ms (upload)\t";
    std::cout.flush();

    cv::cuda::GpuMat disp_gpu;

    tick = std::chrono::high_resolution_clock::now();

    BICOS::match(lstack_gpu, rstack_gpu, disp_gpu, c);

    DELTA_MS(match);

    std::cout << delta_match << "ms (match)\t";
    std::cout.flush();

    tick = std::chrono::high_resolution_clock::now();

    disp_gpu.download(disp);

    DELTA_MS(download);

    std::cout << delta_download << "ms (download)" << std::endl;

#else

    auto tick = std::chrono::high_resolution_clock::now();

    BICOS::match(lstack, rstack, disp, c);

    DELTA_MS(match);

    std::cout << "Latency:\t" << delta_match << "ms" << std::endl;

#endif

    save_disparity(disp, outfile);

    if (q_store.has_value()) {
        cv::Mat Q;
        cv::Mat3f points;
        cv::FileStorage fs(q_store.value(), cv::FileStorage::READ);

        fs["Q"] >> Q;

        fs.release();

        cv::reprojectImageTo3D(disp, points, Q, false, CV_32F);

        save_pointcloud(points, disp, outfile);
    }

    return 0;
}