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

#include "impl/cuda/cutil.cuh"

#include <iostream>

using namespace BICOS;
using namespace impl;

#define UINT128_MAX (((__uint128_t)UINT64_MAX << 64) | UINT64_MAX)

__global__ void load_128bit_kernel(const __uint128_t *src, __uint128_t *dst) {
    *dst = cuda::load_datacache(src);
}

int main(void) {
    __uint128_t x, y, *py;
    __uint128_t *px;

    cudaMalloc(&py, sizeof(__uint128_t));
    cudaMalloc(&px, sizeof(__uint128_t));

    int i;
    for (i = 0, x = 0; x < UINT128_MAX; x |= __uint128_t(UINT16_MAX) << i++ * sizeof(uint16_t)) {
        assertCudaSuccess(cudaMemcpy(px, &x, sizeof(__uint128_t), cudaMemcpyHostToDevice));
        load_128bit_kernel<<<1,1>>>(px, py);
        assertCudaSuccess(cudaMemcpy(&y, py, sizeof(__uint128_t), cudaMemcpyDeviceToHost));
        assertCudaSuccess(cudaDeviceSynchronize());

        if (x != y) {
            std::cerr << std::hex << "x: " << uint64_t(x >> 64) << ' ' << uint64_t(x) << " y: " << uint64_t(y >> 64) << ' ' << uint64_t(y) << std::endl;
            return 1; 
        }
    }

    cudaHostUnregister(&y);
    cudaHostUnregister(&x);

    return 0;
}
