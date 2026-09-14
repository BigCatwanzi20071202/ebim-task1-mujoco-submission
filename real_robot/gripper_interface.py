"""Robotiq percentage-command adapter for the site gripper manager."""

import math
from typing import Any

from .config_real import GRIPPER_COMMAND_TOPICS


class GripperInterface:
    def __init__(self, motion_enabled: bool = False, *, node: Any = None) -> None:
        self.motion_enabled = bool(motion_enabled)
        self.node = node

    def command(self, arm: str, width_percent: float) -> None:
        if not self.motion_enabled:
            raise PermissionError("motion enable guard rejected gripper command")
        if arm not in GRIPPER_COMMAND_TOPICS:
            raise ValueError(f"unknown arm: {arm}")
        value = float(width_percent)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("gripper width percent must be finite and within [0, 1]")
        if self.node is None:
            raise RuntimeError("an initialized rclpy node is required")
        try:
            from std_msgs.msg import Float32
        except ImportError as exc:
            raise RuntimeError(f"ROS2 gripper interface unavailable: {exc}") from exc
        publisher = self.node.create_publisher(Float32, GRIPPER_COMMAND_TOPICS[arm], 1)
        try:
            message = Float32()
            message.data = value
            publisher.publish(message)
        finally:
            self.node.destroy_publisher(publisher)
