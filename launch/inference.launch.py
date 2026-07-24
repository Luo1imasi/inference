# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

##launch file
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def policy_uses_depth(policy_config_path: str) -> bool:
    """Detect use_depth: true in policy yaml without requiring PyYAML."""
    try:
        with open(policy_config_path, "r", encoding="utf-8") as f:
            for line in f:
                if re.match(r"^\s*use_depth:\s*true\s*(#.*)?$", line):
                    return True
    except OSError:
        return False
    return False


def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration("robot").perform(context)
    policy = LaunchConfiguration("policy").perform(context)
    if not NAME_PATTERN.fullmatch(robot):
        raise ValueError(f"Invalid robot name: {robot}")
    if not NAME_PATTERN.fullmatch(policy):
        raise ValueError(f"Invalid policy name: {policy}")

    policy_file = policy if policy.endswith(".yaml") else f"{policy}.yaml"

    robot_dir = os.path.join(
        get_package_share_directory("roboparty_inference"),
        "robots",
        robot,
    )
    robot_config = os.path.join(robot_dir, "robot.yaml")
    policy_config = os.path.join(robot_dir, "configs", policy_file)
    model_dir = os.path.join(robot_dir, "models")
    motion_dir = os.path.join(robot_dir, "motions")

    if not os.path.isfile(robot_config):
        raise FileNotFoundError(f"Robot config not found: {robot_config}")
    if not os.path.isfile(policy_config):
        raise FileNotFoundError(f"Inference config not found: {policy_config}")

    nodes = [
        Node(
            package="roboparty_inference",
            executable="inference_node",
            name="inference_node",
            parameters=[
                policy_config,
                {
                    "robot_name": robot,
                    "policy_name": policy,
                    "robot_config": robot_config,
                    "model_dir": model_dir,
                    "motion_dir": motion_dir,
                },
            ],
            output="screen",
        ),
    ]

    if policy_uses_depth(policy_config):
        nodes.insert(
            0,
            Node(
                package="roboparty_inference",
                executable="depth_node",
                name="depth_node",
                parameters=[
                    policy_config,
                    {"model_dir": model_dir},
                ],
                output="screen",
            ),
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="rpo"),
            DeclareLaunchArgument("policy", default_value="default"),
            OpaqueFunction(function=launch_setup),
        ]
    )
