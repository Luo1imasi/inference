# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

"""Unified inference launch: inference_node, plus depth_node when use_depth is true.

Depth node params / encoder models come from the camera package.
Inference params / actor models remain under roboparty_inference.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
USE_DEPTH_PATTERN = re.compile(
    r"^\s*use_depth\s*:\s*(true|false|1|0)\s*(?:#.*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_use_depth(policy_config: str) -> bool:
    with open(policy_config, "r", encoding="utf-8") as f:
        text = f.read()
    match = USE_DEPTH_PATTERN.search(text)
    if not match:
        return False
    return match.group(1).lower() in ("true", "1")


def launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration("robot").perform(context)
    policy = LaunchConfiguration("policy").perform(context)
    use_depth_arg = LaunchConfiguration("use_depth").perform(context).strip().lower()

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

    if use_depth_arg in ("", "auto"):
        use_depth = _parse_use_depth(policy_config)
    elif use_depth_arg in ("true", "1"):
        use_depth = True
    elif use_depth_arg in ("false", "0"):
        use_depth = False
    else:
        raise ValueError(
            f"Invalid use_depth={use_depth_arg!r}; expected auto/true/false"
        )

    start_depth_arg = LaunchConfiguration("start_depth_node").perform(context).strip().lower()
    if start_depth_arg in ("", "auto"):
        start_depth_node = use_depth
    elif start_depth_arg in ("true", "1"):
        start_depth_node = True
    elif start_depth_arg in ("false", "0"):
        start_depth_node = False
    else:
        raise ValueError(
            f"Invalid start_depth_node={start_depth_arg!r}; expected auto/true/false"
        )

    nodes = []

    # depth_node may already be launched after the camera (see start_robot_depth.sh)
    # so that encoder.onnx + debug crop/downsample run without waiting for policy.
    if start_depth_node:
        camera_share = get_package_share_directory("camera")
        depth_config = os.path.join(camera_share, "configs", policy_file)
        depth_model_dir = os.path.join(camera_share, "models")
        if not os.path.isfile(depth_config):
            raise FileNotFoundError(f"Depth config not found: {depth_config}")
        nodes.append(
            Node(
                package="camera",
                executable="depth_node",
                name="depth_node",
                parameters=[
                    depth_config,
                    {"model_dir": depth_model_dir},
                ],
                output="screen",
            )
        )

    nodes.append(
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
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="rpo"),
            DeclareLaunchArgument("policy", default_value="default"),
            DeclareLaunchArgument(
                "use_depth",
                default_value="auto",
                description="auto: read use_depth from policy yaml; true/false to override",
            ),
            DeclareLaunchArgument(
                "start_depth_node",
                default_value="auto",
                description=(
                    "auto: start depth_node when use_depth is true; "
                    "false when depth_node is already launched separately"
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
