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

namespace BICOS::impl::cpu {

template<typename T>
static double nxcorr(const T* pix0, const T* pix1, size_t n, std::optional<double> minvar) {
    double mean0 = 0.0, mean1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean0 += pix0[i];
        mean1 += pix1[i];
    }
    mean0 /= n;
    mean1 /= n;

    double covar = 0.0, var0 = 0.0, var1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff0 = pix0[i] - mean0, diff1 = pix1[i] - mean1;

        covar = std::fma(diff0, diff1, covar);
        var0 = std::fma(diff0, diff0, var0);
        var1 = std::fma(diff1, diff1, var1);
    }

    if (minvar.has_value() && (var0 < *minvar || var1 < *minvar))
        return -1.0;

    return covar / std::sqrt(var0 * var1);
}

template<typename TInput>
static void agree(
    const cv::Mat1s& raw_disp,
    const cv::Mat& stack0,
    const cv::Mat& stack1,
    size_t n_images,
    double nxcorr_threshold,
    std::optional<double> min_var,
    cv::Mat_<disparity_t>& ret
) {
    auto sz = raw_disp.size();

    ret.create(sz);
    ret.setTo(INVALID_DISP);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const cv::Mat1s raw_row = raw_disp.row(row);
            cv::Mat_<disparity_t> ret_row = ret.row(row);

            for (int col = 0; col < sz.width; ++col) {
                const int16_t d = raw_row.at<int16_t>(col);

                if (d == INVALID_DISP_<int16_t>)
                    continue;

                const int idx1 = col - d;

                if (idx1 < 0 || sz.width <= idx1)
                    continue;

                double nxc =
                    nxcorr(stack0.ptr<TInput>(row, col), stack1.ptr<TInput>(row, idx1), n_images, min_var);

                if (nxc < nxcorr_threshold)
                    continue;

                ret_row.at<disparity_t>(col) = d;
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
    double nxcorr_threshold,
    float subpixel_step,
    std::optional<double> min_var,
    cv::Mat_<disparity_t>& ret
) {
    auto sz = raw_disp.size();

    ret.create(sz);
    ret.setTo(INVALID_DISP);

    cv::parallel_for_(cv::Range(0, sz.height), [&](const cv::Range& r) {
        for (int row = r.start; row < r.end; ++row) {
            const cv::Mat1s raw_row = raw_disp.row(row);
            cv::Mat_<disparity_t> ret_row = ret.row(row);

            for (int col = 0; col < sz.width; ++col) {
                const int16_t d = raw_row.at<int16_t>(col);

                if (d == INVALID_DISP_<int16_t>)
                    continue;

                const int col1 = col - d;

                if (col1 < 0 || sz.width <= col1)
                    continue;

                if (col1 == 0 || col1 == sz.width - 1) {
                    double nxc = nxcorr(
                        stack0.ptr<TInput>(row, col),
                        stack1.ptr<TInput>(row, col1),
                        n,
                        min_var
                    );

                    if (nxc < nxcorr_threshold)
                        continue;

                    ret_row.at<disparity_t>(col) = d;
                } else {
                    // clang-format off

                    TInput interp[33];
                    float a[33], b[33], c[33];

                    const TInput *y0 = stack1.ptr<TInput>(row, col1 - 1),
                                 *y1 = stack1.ptr<TInput>(row, col1    ),
                                 *y2 = stack1.ptr<TInput>(row, col1 + 1);
                    
                    for (size_t t = 0; t < n; ++t) {
                        a[t] = 0.5f * ( y0[t] - 2.0f * y1[t] + y2[t]);
                        b[t] = 0.5f * (-y0[t]                + y2[t]);
                        c[t] = y1[t];
#ifdef BICOS_DEBUG
                        assert(t < 33);
#endif
                    }

                    // clang-format on

                    float best_x = 0.0f;
                    double best_nxcorr = -1.0;

                    for (float x = -1.0f; x <= 1.0f; x += subpixel_step) {
                        for (size_t t = 0; t < n; ++t)
                            interp[t] = (TInput)roundevenf(a[t] * x * x + b[t] * x + c[t]);

                        double nxc = nxcorr(stack0.ptr<TInput>(row, col), interp, n, min_var);

                        if (best_nxcorr < nxc) {
                            best_x = x;
                            best_nxcorr = nxc;
                        }
                    }

                    if (best_nxcorr < nxcorr_threshold)
                        continue;

                    ret_row.at<disparity_t>(col) = d + best_x;
                }
            }
        }
    });
}

} // namespace BICOS::impl::cpu
