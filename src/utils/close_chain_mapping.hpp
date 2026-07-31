// SPDX-License-Identifier: GPL-3.0
// Copyright (C) 2025-2026 Luo1imasi

#pragma once

#include <memory>
#include <string>
#include <Eigen/Dense>

struct JacobianResult
{
    Eigen::Matrix2d J_motor2Joint;
    Eigen::Matrix2d J_Joint2motor;
};

struct ForwardMappingResult
{
    Eigen::Vector2d ankle_joint_ori;
    JacobianResult Jac;
};

class Decouple
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual ~Decouple() = default;

    virtual void get_forwardQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel,
                                Eigen::Vector2d &tau, bool is_left) = 0;

    virtual void get_decoupleQVT(Eigen::Vector2d &q, Eigen::Vector2d &vel,
                                 Eigen::Vector2d &tau, bool is_left) = 0;

    static std::shared_ptr<Decouple> create(const std::string &type);
};
