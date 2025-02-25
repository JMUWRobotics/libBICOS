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

#pragma once

#include "common.hpp"
#include "stepbuf.hpp"
#include "impl/common.hpp"

#include <bitset>

namespace BICOS::impl::cpu {

[[maybe_unused]] static int ham(uint32_t a, uint32_t b) {
    return __builtin_popcount(a ^ b);
}

[[maybe_unused]] static int ham(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

[[maybe_unused]] static int ham(uint128_t a, uint128_t b) {
    // clang-format off

    uint128_t diff = a ^ b;
    return __builtin_popcountll((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL))
         + __builtin_popcountll((uint64_t)(diff >> 64));
}

template<size_t N>
[[maybe_unused]] static int ham(std::bitset<N> a, std::bitset<N> b) {
    return (a ^ b).count();
}

template<typename TDescriptor, int FLAGS>
int bicos_search(TDescriptor d0, const TDescriptor* row1, size_t cols) {
    int best_col1 = INVALID_DISP<int>, min_cost = INT_MAX, num_duplicate_minima = 0;

    for (size_t col1 = 0; col1 < cols; ++col1) {
        const TDescriptor d1 = row1[col1];

        int cost = ham(d0, d1);

        if (cost < min_cost) {
            min_cost = cost;
            best_col1 = col1;

            if constexpr (FLAGS & BICOSFLAGS_NODUPES)
                num_duplicate_minima = 0;

        } else if constexpr (FLAGS & BICOSFLAGS_NODUPES)
            if (cost == min_cost)
                num_duplicate_minima++;
    }

    if constexpr (FLAGS & BICOSFLAGS_NODUPES)
        if (0 < num_duplicate_minima)
            return INVALID_DISP<int>;

    return best_col1;
}

template<typename TDescriptor, int FLAGS>
void bicos(
    const std::unique_ptr<StepBuf<TDescriptor>>& desc0,
    const std::unique_ptr<StepBuf<TDescriptor>>& desc1,
    int max_lr_diff,
    cv::Size sz,
    cv::Mat &out
) {
    out.create(sz, cv::DataType<int16_t>::type);
    out.setTo(INVALID_DISP<int16_t>);

    cv::parallel_for_(cv::Range(0, out.rows), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const TDescriptor *drow0 = desc0->row(row), *drow1 = desc1->row(row);

            for (int col0 = 0; col0 < out.cols; ++col0) {
                int best_col1 = bicos_search<TDescriptor, FLAGS>(drow0[col0], drow1, out.cols);

                if (is_invalid(best_col1))
                    continue;

                if constexpr (FLAGS & BICOSFLAGS_CONSISTENCY) {
                    int reverse_col0 =
                        bicos_search<TDescriptor, FLAGS>(drow1[best_col1], drow0, out.cols);

                    if (is_invalid(reverse_col0) || abs(col0 - reverse_col0) > max_lr_diff)
                        continue;

                    out.at<int16_t>(row, col0) = (col0 + reverse_col0) / 2 - best_col1;

                } else
                    out.at<int16_t>(row, col0) = col0 - best_col1;
            }
        }
    });
}

} // namespace BICOS::impl::cpu
