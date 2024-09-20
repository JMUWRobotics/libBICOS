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

#include "bitfield.hpp"
#include "impl/common.hpp"
#include "stepbuf.hpp"

namespace BICOS::impl::cuda {

template<typename TInput, typename TDescriptor>
__global__ void transform_full_kernel(
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t _n,
    cv::Size size,
    StepBuf<TDescriptor>* out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (size.width <= col || size.height <= row)
        return;

    TInput pix[PIX_STACKSIZE];
    Bitfield<TDescriptor> bf;
    ssize_t n = (ssize_t)_n;

    float av = 0.0f;
    for (ssize_t i = 0; i < n; ++i) {
        av += pix[i] = stacks[i](row, col);
#ifdef BICOS_DEBUG
        if (i >= sizeof(pix))
            __trap();
#endif
    }
    av /= n;

    wider_t<TInput> pairsums[PIX_STACKSIZE];

    // clang-format off

    for (ssize_t t = 0; t < n - 2; ++t) {
        const TInput a = pix[t + 0],
                     b = pix[t + 1],
                     c = pix[t + 2];
        
        bf.set(a < b);
        bf.set(a < c);
        bf.set(a < av);

        pairsums[t] = wider_t<TInput>(pix[t]) + wider_t<TInput>(pix[t + 1]);
    }

    pairsums[n - 2] = wider_t<TInput>(pix[n - 2]) + wider_t<TInput>(pix[n - 1]);

    const TInput a = pix[n - 2],
                 b = pix[n - 1];

    bf.set(a < b);
    bf.set(a < av);
    bf.set(b < av);

    for (ssize_t t = 0; t < n - 1; ++t) {
        for (ssize_t i = 0; i < n - 1; ++i) {
            if (i == t || i == t - 1 || i == t + 1)
                continue;

            bf.set(pairsums[t] < pairsums[i]);
        }
    }

    // clang-format on

    out->row(row)[col] = bf.v;
}

template<typename TInput, typename TDescriptor>
__global__ void transform_limited_kernel(
    const cv::cuda::PtrStepSz<TInput>* stacks,
    size_t n,
    cv::Size size,
    StepBuf<TDescriptor>* out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (size.width <= col || size.height <= row)
        return;

    TInput pix[PIX_STACKSIZE];
    Bitfield<TDescriptor> bf;

    float av = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        av += pix[i] = stacks[i](row, col);
#ifdef BICOS_DEBUG
        if (i >= sizeof(pix))
            __trap();
#endif
    }
    av /= double(n);

    // clang-format off

    int prev_pair_sums[] = { -1, -1 };
    for (size_t t = 0; t < n - 2; ++t) {
        const TInput a = pix[t + 0],
                     b = pix[t + 1],
                     c = pix[t + 2];
        int& prev_pair_sum = prev_pair_sums[t % 2],
             current_sum   = a + b;
        
        bf.set(a < b);
        bf.set(a < c);
        bf.set(a < av);

        if (-1 != prev_pair_sum) {
            bf.set(prev_pair_sum < current_sum);
        }

        prev_pair_sum = current_sum;
    }

    const TInput a = pix[n - 2],
                 b = pix[n - 1];

    bf.set(a < b);
    bf.set(a < av);
    bf.set(b < av);
    bf.set(prev_pair_sums[(n - 2) % 2] < (a + b));

    // clang-format on

    out->row(row)[col] = bf.v;
}

} // namespace BICOS::impl::cuda