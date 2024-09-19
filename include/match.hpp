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

#pragma once

#include "common.hpp"

#include <opencv2/core.hpp>

#if defined(BICOS_CUDA)
    #include <opencv2/core/cuda.hpp>
#endif

namespace BICOS {

void match(
    const std::vector<InputImage>& stack0,
    const std::vector<InputImage>& stack1,
    OutputImage& disparity,
    Config cfg = Config()
#if defined(BICOS_CUDA)
    ,
    cv::cuda::Stream& stream = cv::cuda::Stream::Null()
#endif
);

} // namespace BICOS
