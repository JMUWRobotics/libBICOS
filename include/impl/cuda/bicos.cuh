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

namespace BICOS::impl::cuda {

static __device__ __forceinline__ int ham(uint32_t a, uint32_t b) {
    return __builtin_popcount(a ^ b);
}

static __device__ __forceinline__ int ham(uint64_t a, uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

static __device__ __forceinline__ int ham(uint128_t a, uint128_t b) {
    const uint128_t diff = a ^ b;
    int lo = __builtin_popcountll((uint64_t)(diff & 0xFFFFFFFFFFFFFFFFUL));
    int hi = __builtin_popcountll((uint64_t)(diff >> 64));
    return lo + hi;
}

template<typename TDescriptor, TDescriptor (*FLoad)(const TDescriptor*)>
static __device__ __forceinline__ int16_t
bicos_search(TDescriptor d0, const TDescriptor* row1, size_t cols) {
    int best_col1 = -1, min_cost = INT_MAX, num_duplicate_minima = 0;

    for (size_t col1 = 0; col1 < cols; ++col1) {
        const TDescriptor d1 = FLoad(row1 + col1);

        int cost = ham(d0, d1);

        if (cost < min_cost) {
            min_cost = cost;
            best_col1 = col1;
            num_duplicate_minima = 0;
        } else if (cost == min_cost) {
            num_duplicate_minima++;
        }
    }

    if (0 < num_duplicate_minima)
        return -1;

    return best_col1;
}

template<typename TDescriptor>
__global__ void bicos_kernel_smem(
    const StepBuf<TDescriptor>* descr0,
    const StepBuf<TDescriptor>* descr1,
    cv::cuda::PtrStepSz<int16_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.rows <= row)
        return;

    extern __shared__ char _row1[];
    TDescriptor* row1 = (TDescriptor*)_row1;

    for (size_t c = threadIdx.x; c < out.cols; c += blockDim.x)
        row1[c] = descr1->row(row)[c];

    if (out.cols <= col)
        return;

    __syncthreads();

    int best_col1 = bicos_search<TDescriptor, load_deref>(
        load_datacache(descr0->row(row) + col),
        row1,
        out.cols
    );

    if (best_col1 != -1)
        out(row, col) = abs(col - best_col1);
}

template<typename TDescriptor>
__global__ void bicos_kernel(
    const StepBuf<TDescriptor>* descr0,
    const StepBuf<TDescriptor>* descr1,
    cv::cuda::PtrStepSz<int16_t> out
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out.cols <= col || out.rows <= row)
        return;

    int best_col1 = bicos_search<TDescriptor, load_datacache>(
        load_datacache(descr0->row(row) + col),
        descr1->row(row),
        out.cols
    );

    if (best_col1 != -1)
        out(row, col) = abs(col - best_col1);
}

} // namespace BICOS::impl::cuda
