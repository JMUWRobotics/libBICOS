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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda/common.hpp>

#define PIX_STACKSIZE 33

namespace BICOS::impl::cuda {

#define assertCudaSuccess(call) \
    do { \
        cudaError_t err = (call); \
        if (cudaSuccess != err) { \
            std::cerr << "libBICOS CUDA error in " << __FILE__ << " [ " << __PRETTY_FUNCTION__ \
                      << " | L" << __LINE__ << " ]: " << cudaGetErrorString(err) << std::endl; \
            abort(); \
        } \
    } while (0)

#define create_grid(block, size) \
    dim3( \
        cv::cuda::device::divUp(size.width, block.x), \
        cv::cuda::device::divUp(size.height, block.y) \
    )

template<class T> 
inline dim3 max_blocksize(T fun, size_t smem_size = 0) {
    int _minGridSize, blockSize;
    assertCudaSuccess(cudaOccupancyMaxPotentialBlockSize(&_minGridSize, &blockSize, fun, smem_size));
    return dim3(blockSize);
}

template<typename T>
class RegisteredPtr {
private:
    T *_phost, *_pdev;

public:
    RegisteredPtr(T* phost, size_t n = 1, bool read_only = false): _phost(phost) {
        unsigned int flags = read_only ? cudaHostRegisterReadOnly : 0;

        assertCudaSuccess(cudaHostRegister(_phost, sizeof(T) * n, flags));
        assertCudaSuccess(cudaHostGetDevicePointer(&_pdev, _phost, 0));
    }
    ~RegisteredPtr() {
        assertCudaSuccess(cudaHostUnregister(_phost));
    }

    RegisteredPtr(const RegisteredPtr&) = delete;
    RegisteredPtr& operator=(const RegisteredPtr&) = delete;

    operator T*() {
        return _pdev;
    }
    operator const T*() {
        return _pdev;
    }

    T* operator+(int rhs) {
        return _pdev + rhs;
    }
    const T* operator+(int rhs) const {
        return _pdev + rhs;
    }
};

template<typename T>
__device__ __forceinline__
T load_datacache(const T* p) {
    return __ldg(p);
}

template<>
__device__ __forceinline__
__uint128_t load_datacache<__uint128_t>(const __uint128_t* _p) {
    auto p = reinterpret_cast<const uint64_t *>(_p);
    return (__uint128_t(__ldg(p + 1)) << 64) | __uint128_t(__ldg(p));
}

template<typename T>
__device__ __forceinline__
T load_deref(const T* p) {
    return *p;
}

} // namespace BICOS::impl::cuda
