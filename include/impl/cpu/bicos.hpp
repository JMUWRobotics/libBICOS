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
#include "stepbuf.hpp"

#include <bit>

namespace BICOS::impl::cpu {

[[maybe_unused]] static int ham(uint32_t a, uint32_t b) {
    return std::popcount(a ^ b);
}

[[maybe_unused]] static int ham(uint64_t a, uint64_t b) {
    return std::popcount(a ^ b);
}

[[maybe_unused]] static int ham(uint128_t a, uint128_t b) {
    uint128_t diff = a ^ b;
    return std::popcount((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL))
        + std::popcount((uint64_t)(diff >> 64));
}

template<typename TDescriptor>
cv::Mat1s bicos(
    const std::unique_ptr<StepBuf<TDescriptor>>& desc0,
    const std::unique_ptr<StepBuf<TDescriptor>>& desc1,
    cv::Size sz
) {
    cv::Mat1s ret(sz);
    ret.setTo(INVALID_DISP_<int16_t>);

    cv::parallel_for_(cv::Range(0, ret.rows), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const TDescriptor *drow0 = desc0->row(row), *drow1 = desc1->row(row);

            for (int col0 = 0; col0 < ret.cols; ++col0) {
                const TDescriptor d0 = drow0[col0];

                int best_col1 = INT_MIN, min_cost = INT_MAX, num_duplicate_minima = 0;

                for (int col1 = 0; col1 < ret.cols; ++col1) {
                    const TDescriptor d1 = drow1[col1];

                    int cost = ham(d0, d1);

                    if (cost < min_cost) {
                        min_cost = cost;
                        best_col1 = col1;
                        num_duplicate_minima = 0;
                    } else if (cost == min_cost) {
                        num_duplicate_minima++;
                    }
                }

                if (INT_MIN == best_col1 || 0 < num_duplicate_minima)
                    continue;

                ret(row, col0) = col0 - best_col1;
            }
        }
    });

    return ret;
}

} // namespace BICOS::impl::cpu
