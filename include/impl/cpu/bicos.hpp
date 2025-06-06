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
#include "impl/common.hpp"
#include "stepbuf.hpp"

#include <bitset>
#include <numeric>

#include <opencv2/core/simd_intrinsics.hpp>

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
int bicos_search_(TDescriptor d0, const TDescriptor* row1, size_t cols) {
    int best_col1 = INVALID_DISP<int>, min_cost = INT_MAX, num_duplicate_minima = 0;

    for (size_t col1 = 0; col1 < cols; ++col1) {
        if (int cost = ham(d0, row1[col1]); cost < min_cost) {
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

template<typename TDescriptor>
struct BicosSearch {
    template<int FLAGS> static int invoke(TDescriptor d0, const TDescriptor* row1, size_t cols) {
        return bicos_search_<TDescriptor, FLAGS>(d0, row1, cols);
    }
};

#if defined(CV_SIMD) && CV_SIMD

template<typename T>
auto vx_idxs() {
    std::array<typename T::lane_type, T::nlanes> idxs;
    std::iota(idxs.begin(), idxs.end(), 0);
    return cv::vx_load(idxs.data());
}

template<>
struct BicosSearch<uint32_t> {
    template<int FLAGS> static int invoke(uint32_t d0, const uint32_t *row1, size_t cols) {
        const cv::v_uint32
            incr = cv::vx_setall_u32(cv::v_uint32::nlanes);
        const cv::v_uint32 vd0 = cv::vx_setall_u32(d0);
        cv::v_uint32 idxs_best,
            idxs_cur = vx_idxs<cv::v_uint32>(),
            cost_best = cv::vx_setall_u32(UINT32_MAX);
        
        for (size_t i = 0; i < cols; i += cv::v_uint32::nlanes) {
            auto cost_cur = cv::v_popcount(cv::v_xor(vd0, cv::vx_load(row1 + i)));
            auto lt = cv::v_lt(cost_cur, cost_best);

            /* for correctness, we should check if cost_cur == cost_best,
               since now, this SIMD implementation does not account for
               duplicates on same lanes */

            idxs_best = cv::v_select(lt, idxs_cur, idxs_best);
            cost_best = cv::v_select(lt, cost_cur, cost_best);
            idxs_cur = cv::v_add(idxs_cur, incr);
        }

        const size_t rest = cols % cv::v_uint32::nlanes,
                     n = cv::v_uint32::nlanes + rest;
        std::array<uint32_t, 2 * cv::v_uint32::nlanes - 1> idxs_rest, cost_rest, idxs;

        cv::v_store(idxs_rest.data(), idxs_best);
        cv::v_store(cost_rest.data(), cost_best);

        std::iota(idxs.begin(), idxs.begin() + n, 0);

        std::transform(row1 + cols - rest, row1 + cols, cost_rest.begin() + cv::v_uint32::nlanes, [&](const uint32_t &d1) { return ham(d0, d1); });
        std::iota(idxs_rest.begin() + cv::v_uint32::nlanes, idxs_rest.begin() + n, cols - rest);

        std::sort(idxs.begin(), idxs.begin() + n, [&](const uint32_t &i1, const uint32_t &i2) { return idxs_rest[i1] < idxs_rest[i2]; });

        const auto minidx = std::min_element(idxs.begin(), idxs.begin() + n, [&](const uint32_t &i1, const uint32_t &i2) { return cost_rest[i1] < cost_rest[i2]; });

        if constexpr (FLAGS & BICOSFLAGS_NODUPES) {
            const uint32_t &mincost = cost_rest[*minidx];
            if (std::count_if(cost_rest.begin(), cost_rest.begin() + n, [&](const uint32_t &c) { return c == mincost; }) > 1)
                return INVALID_DISP<int>;
        }

        return idxs_rest[*minidx];
    }
};

#endif

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

    const auto bicos_search = BicosSearch<TDescriptor>::template invoke<FLAGS>;

    cv::parallel_for_(cv::Range(0, out.rows), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const TDescriptor *drow0 = desc0->row(row), *drow1 = desc1->row(row);

            for (int col0 = 0; col0 < out.cols; ++col0) {
                int best_col1 = bicos_search(drow0[col0], drow1, out.cols);

                if (is_invalid(best_col1))
                    continue;

                if constexpr (FLAGS & BICOSFLAGS_CONSISTENCY) {
                    int reverse_col0 = bicos_search(drow1[best_col1], drow0, out.cols);

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
