/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
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

#include <fmt/core.h>

#include "impl/cpu/bicos.hpp"

using namespace BICOS;
using namespace impl;
using namespace cpu;

#if ! CV_SIMD

int main(void) { return 77; }

#else

#define loop(n) for (size_t __i = 0; __i < (n); ++__i)

int main(void) {

    loop(10) {
        for (size_t off = 0; off < cv::v_uint32::nlanes; ++off) {
            std::vector<uint32_t> vals(1024 + off + 1);
            std::generate(vals.begin(), vals.end(), [](){ return arc4random(); });

            int vanilla = bicos_search_<uint32_t, BICOSFLAGS_NODUPES>(vals[0], vals.data() + 1, vals.size() - 1);
            fmt::println(stderr, "vanilla: {}", vanilla);
            int simd    = BicosSearch<uint32_t>::template invoke<BICOSFLAGS_NODUPES>(vals[0], vals.data() + 1, vals.size() - 1);
            fmt::println(stderr, "simd: {}", simd);

            if (simd != vanilla) {
                fmt::println(stderr, "{} != {}", simd, vanilla);
                return 1;
            }
        }
    }

    return 0;
}

#endif