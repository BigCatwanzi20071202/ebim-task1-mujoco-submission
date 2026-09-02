"""ROS-facing FR3 status adapter; movement is intentionally not implemented."""

from dataclasses import dataclass
from typing import Any

from .config_real import ARMS, PTP_ACTION_TYPE


@dataclass(frozen=True)
class RobotCheck:
    ros2_available: bool
    detail: str


class FR3Interface:
    def __init__(self, motion_enabled: bool = False) -> None:
        self.motion_enabled = bool(motion_enabled)

    @staticmethod
    def check() -> RobotCheck:
        try:
            import rclpy  # noqa: F401
            import franka_msgs.action  # type: ignore # noqa: F401
        except ImportError as exc:
            return RobotCheck(False, f"ROS2/FR3 Python interfaces unavailable: {exc}")
        return RobotCheck(True, "ROS2 imports available; topic/action connectivity not probed")

    def current_pose_topic(self, arm: str) -> str:
        return ARMS[arm].pose_topic

    def send_ptp(self, arm: str, pose: Any) -> None:
        del pose
        if not self.motion_enabled:
            raise PermissionError("motion enable guard rejected PTP command")
        raise NotImplementedError(
            f"Movement disabled in v0.1: first inspect `ros2 interface show {PTP_ACTION_TYPE}`; "
            f"no goal fields are assumed for {ARMS[arm].ptp_action}"
        )
