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

#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>

#ifdef BICOS_CUDA
    #include <opencv2/core/cuda.hpp>
#endif

#include "fileutils.hpp"
#include "match.hpp"

using namespace BICOS;

int main(int argc, char const* const* argv) {
    cxxopts::Options opts(argv[0], "cli to process images with BICOS");

    // clang-format off

    opts.add_options()
        ("folder0", "First folder containing input images with numbered names", cxxopts::value<std::string>())
        ("folder1", "Optional second folder with input images. If specified, file names need to be 0.png, 1.png... Else, folder0 needs to contain 0_left.png, 0_right.png, 1_left.png...", cxxopts::value<std::string>())
        ("t,threshold", "Normalized cross corellation threshold", cxxopts::value<double>()->default_value("0.5"))
        ("s,step", "Subpixel step (optional)", cxxopts::value<float>())
        ("m,mode", "Tranformation mode {'FULL', 'LIMITED'} (unused)", cxxopts::value<std::string>()->default_value("LIMITED"))
        ("o,outfile", "Output file for disparity image", cxxopts::value<std::string>()->default_value("bicosdisp.png"))
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

    std::filesystem::path folder0 = args["folder0"].as<std::string>();
    std::optional<std::filesystem::path> folder1 = std::nullopt;

    if (args.count("folder1"))
        folder1 = args["folder1"].as<std::string>();

    std::vector<cv::Mat> lstack, rstack;
    {
        std::vector<SequenceEntry> lseq, rseq;

        read_sequence(folder0, folder1, lseq, rseq, true);
        sort_sequence_to_stack(lseq, rseq, lstack, rstack);

        std::cout << "loaded " << lseq.size() + rseq.size() << " images\n";
    }

    cv::Mat_<BICOS::disparity_t> disp;

    BICOS::Config c {
        .nxcorr_thresh = args["threshold"].as<double>(),
    };

    if (args.count("step"))
        c.subpixel_step = args["step"].as<float>();
#ifdef BICOS_CUDA
    if (args.count("single"))
        c.precision = Precision::SINGLE;
#endif

#ifdef BICOS_CUDA

    std::vector<cv::cuda::GpuMat> lstack_gpu, rstack_gpu;
    matvec_to_gpu(lstack, rstack, lstack_gpu, rstack_gpu);

    cv::cuda::GpuMat disp_gpu;

    BICOS::match(lstack_gpu, rstack_gpu, disp_gpu, c);

    disp_gpu.download(disp);

#else

    BICOS::match(lstack, rstack, disp, c);

#endif

    save_disparity(disp, args["outfile"].as<std::string>());

    return 0;
}