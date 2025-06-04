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
#include "compat.hpp"

namespace BICOS::impl::cpu {

template<typename T>
static float nxcorr(const T* pix0, const T* pix1, size_t n, std::optional<float> minvar) {
    float mean0 = 0.f, mean1 = 0.f;
    for (size_t i = 0; i < n; ++i) {
        mean0 += pix0[i];
        mean1 += pix1[i];
    }
    mean0 /= n;
    mean1 /= n;

    float covar = 0.f, var0 = 0.f, var1 = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        covar = std::fmaf(diff0, diff1, covar);
        var0 = std::fmaf(diff0, diff0, var0);
        var1 = std::fmaf(diff1, diff1, var1);
    }

    if (minvar.has_value() && (var0 < *minvar || var1 < *minvar))
        return -1.f;

    return covar / std::sqrt(var0 * var1);
}

template<typename TInput>
static void agree(
    cv::Mat& raw_disp, // int16_t
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    size_t n_images,
    float nxcorr_threshold,
    std::optional<float> min_var,
    cv::Mat *corrmap
) {
    auto sz = raw_disp.size();

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            cv::Mat raw_row = raw_disp.row(row);

            for (int col = 0; col < sz.width; ++col) {
                int16_t& d = raw_row.at<int16_t>(col);

                if (is_invalid(d))
                    continue;

                const int idx1 = col - d;

                if UNLIKELY(idx1 < 0 || sz.width <= idx1) {
                    d = INVALID_DISP<int16_t>;
                    continue;
                }

                float nxc =
                    nxcorr(stack0.ptr<TInput>(row, col), stack1.ptr<TInput>(row, idx1), n_images, min_var);

                if (corrmap)
                    corrmap->at<float>(row, col) = nxc;

                if (nxc < nxcorr_threshold)
                    d = INVALID_DISP<int16_t>;
            }
        }
    });
}

template<typename TInput>
static void agree_subpixel(
    const cv::Mat1s& raw_disp,
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    size_t n,
    float min_nxc,
    float subpixel_step,
    std::optional<float> min_var,
    cv::Mat& ret,
    cv::Mat *corrmap
) {
    auto sz = raw_disp.size();

    ret.create(sz, cv::DataType<float>::type);
    ret.setTo(INVALID_DISP<float>);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const cv::Mat1s raw_row = raw_disp.row(row);
            cv::Mat ret_row = ret.row(row);

            for (int col = 0; col < sz.width; ++col) {
                const int16_t d = raw_row.at<int16_t>(col);

                if (is_invalid(d))
                    continue;

                const int col1 = col - d;

                if UNLIKELY(col1 < 0 || sz.width <= col1)
                    continue;

                if UNLIKELY(col1 == 0 || col1 == sz.width - 1) {
                    float nxc = nxcorr(
                        stack0.ptr<TInput>(row, col),
                        stack1.ptr<TInput>(row, col1),
                        n,
                        min_var
                    );

                    if (corrmap)
                        corrmap->at<float>(row, col) = nxc;

                    if (nxc < min_nxc)
                        continue;

                    ret_row.at<float>(col) = d;
                } else {
                    // clang-format off

                    thread_local std::vector<float> a, b; a.clear(); b.clear(); a.reserve(n); b.reserve(n);

                    const TInput *y0 = stack1.ptr<TInput>(row, col1 - 1),
                                 *y1 = stack1.ptr<TInput>(row, col1    ),
                                 *y2 = stack1.ptr<TInput>(row, col1 + 1);
                    
                    for (size_t t = 0; t < n; ++t) {
                        a.push_back(0.5f * ( y0[t] - 2.0f * y1[t] + y2[t]));
                        b.push_back(0.5f * (-y0[t]                + y2[t]));
                        // c[t] = y1[t];
                    }

                    // clang-format on

                    float best_x = 0.f, best_nxc = -1.f;

                    for (float x = -1.f; x <= 1.f; x += subpixel_step) {
                        thread_local std::vector<TInput> interp; interp.clear(); interp.reserve(n);
                        for (size_t t = 0; t < n; ++t)
                            interp.push_back((TInput)roundevenf(a[t] * x * x + b[t] * x + y1[t]));

                        float nxc = nxcorr(stack0.ptr<TInput>(row, col), interp.data(), n, min_var);

                        if (best_nxc < nxc) {
                            best_x = x;
                            best_nxc = nxc;
                        }
                    }

                    if (corrmap)
                        corrmap->at<float>(row, col) = best_nxc;

                    if (best_nxc < min_nxc)
                        continue;

                    ret_row.at<float>(col) = d - best_x;
                }
            }
        }
    });
}

} // namespace BICOS::impl::cpu
