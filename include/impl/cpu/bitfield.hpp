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

#include <bitset>
#include <type_traits>

namespace BICOS::impl::cpu {

template<typename T>
struct is_bitset: std::false_type {};

template<size_t N>
struct is_bitset<std::bitset<N>>: std::true_type {};

template<typename T>
struct Bitfield {
    unsigned int i = 0u;
    T v = T(0);

    void set(bool value) {
#ifdef BICOS_DEBUG
        if constexpr (is_bitset<T>::value) {
            if (v.size() <= i)
                throw std::overflow_error("Bitfield overflow");
        } else {
            if (sizeof(T) * 8 <= i)
                throw std::overflow_error("Bitfield overflow");
        }
#endif
        if (value) {
            if constexpr (is_bitset<T>::value)
                v.set(i);
            else
                v |= T(1) << i;
        }

        i++;
    }
};

} // namespace BICOS::impl::cpu
