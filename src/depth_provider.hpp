// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace Ort {
class Env;
}

namespace rclcpp {
class Node;
}

class DepthProvider {
   public:
    DepthProvider(rclcpp::Node& node, Ort::Env& env, std::size_t expected_output_size);
    ~DepthProvider();

    DepthProvider(const DepthProvider&) = delete;
    DepthProvider& operator=(const DepthProvider&) = delete;

    bool process_latest(std::vector<float>& output);
    void reset();

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};