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

#include <fmt/format.h>

#include <opencv2/core.hpp>
#include <bitset>

template <> struct fmt::formatter<cv::Size> : formatter<string_view> {
    auto format(const cv::Size &sz, format_context& ctx) const -> format_context::iterator;
};

template <size_t N> struct fmt::formatter<std::bitset<N>> : formatter<string_view> {
    auto format(const std::bitset<N> &set, format_context& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{}", set.to_string());
    }
};
