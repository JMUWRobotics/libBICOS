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

#include "../common.hpp"
#include <cstdint>

namespace BICOS::impl {

template<typename T>
struct Wider;

template<>
struct Wider<uint8_t> {
    using type = uint16_t;
};

template<>
struct Wider<uint16_t> {
    using type = uint32_t;
};

template<typename T>
using wider_t = typename Wider<T>::type;

// clang-format off

constexpr int BICOSFLAGS_NODUPES     = (1 << 0),
              BICOSFLAGS_CONSISTENCY = (1 << 1);

// clang-format on

template<typename T>
constexpr auto baby_sqrt(T n) -> std::enable_if_t<std::is_integral_v<T>, T> {
    T x = n, y = T(1);
    while (x > y) {
        x = (x + y) / T(2);
        y = n / x;
    }
    return x;
}

template<typename TDescriptor, TransformMode mode>
struct MaxStacksize {};

template<typename TDescriptor>
struct MaxStacksize<TDescriptor, TransformMode::LIMITED> {
    static constexpr size_t value = (sizeof(TDescriptor) * 8 + 7) / 4;
};

template<typename TDescriptor>
struct MaxStacksize<TDescriptor, TransformMode::FULL> {
    static constexpr size_t value =
        size_t((2 + baby_sqrt(4 - 4 * (3 - sizeof(TDescriptor) * 8))) / 2.0);
};

template<typename TDescriptor, TransformMode mode>
constexpr size_t max_stacksize_v = MaxStacksize<TDescriptor, mode>::value;

} // namespace BICOS::impl