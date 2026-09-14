"""ROS-facing FR3 PTP adapter with explicit opt-in and bounded waits."""

from dataclasses import dataclass
from typing import Any

from .config_real import ARMS, PTP_ACTION_TYPE
from .ptp_contract import JointPTPGoal, motion_result_reached


@dataclass(frozen=True)
class RobotCheck:
    ros2_available: bool
    detail: str


class FR3Interface:
    def __init__(self, motion_enabled: bool = False, *, node: Any = None,
                 server_timeout: float = 2.0, result_timeout: float = 30.0) -> None:
        self.motion_enabled = bool(motion_enabled)
        self.node = node
        self.server_timeout = float(server_timeout)
        self.result_timeout = float(result_timeout)
        if self.server_timeout <= 0 or self.result_timeout <= 0:
            raise ValueError("timeouts must be positive")

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

    def send_ptp(self, arm: str, goal: JointPTPGoal) -> Any:
        if not self.motion_enabled:
            raise PermissionError("motion enable guard rejected PTP command")
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if not isinstance(goal, JointPTPGoal):
            raise TypeError("goal must be a validated JointPTPGoal")
        if self.node is None:
            raise RuntimeError("an initialized rclpy node is required")
        try:
            import rclpy
            from action_msgs.msg import GoalStatus
            from franka_msgs.action import PTPMotion
            from rclpy.action import ActionClient
        except ImportError as exc:
            raise RuntimeError(f"ROS2/FR3 Python interfaces unavailable: {exc}") from exc
        client = ActionClient(self.node, PTPMotion, ARMS[arm].ptp_action)
        try:
            if not client.wait_for_server(timeout_sec=self.server_timeout):
                raise TimeoutError(f"PTP action server unavailable: {ARMS[arm].ptp_action}")
            send_future = client.send_goal_async(goal.to_ros_goal(PTPMotion))
            rclpy.spin_until_future_complete(self.node, send_future, timeout_sec=self.server_timeout)
            if not send_future.done():
                raise TimeoutError("PTP goal acknowledgement timed out")
            handle = send_future.result()
            if handle is None or not handle.accepted:
                raise RuntimeError("PTP goal was rejected")
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=self.result_timeout)
            if not result_future.done():
                handle.cancel_goal_async()
                raise TimeoutError("PTP result timed out; cancellation requested")
            wrapped = result_future.result()
            if wrapped is None:
                raise RuntimeError("PTP action returned no result")
            if not motion_result_reached(wrapped.status, wrapped.result):
                detail = getattr(wrapped.result, "error_message", "")
                status = getattr(wrapped.result.target_status, "status", None)
                raise RuntimeError(f"PTP target not reached: goal_status={wrapped.status}, target_status={status}, error={detail!r}")
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"unexpected PTP goal status: {wrapped.status}")
            return wrapped.result
        finally:
            client.destroy()
