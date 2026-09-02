"""Known real-cell configuration. Unknown values are deliberately absent."""

from dataclasses import dataclass


REQUIRES_ONSITE_VALIDATION = "REQUIRES_ONSITE_VALIDATION"


@dataclass(frozen=True)
class ArmConfig:
    namespace: str
    robot_ip: str
    pose_topic: str
    ptp_action: str


ARMS = {
    "left": ArmConfig("left", "172.16.16.12", "/left/franka_robot_state_broadcaster/current_pose", "/left/action_server/ptp_motion"),
    "right": ArmConfig("right", "172.16.16.11", "/right/franka_robot_state_broadcaster/current_pose", "/right/action_server/ptp_motion"),
}
PTP_ACTION_TYPE = "franka_msgs/action/PTPMotion"
CAMERA_TOPICS = {
    "head_rgb": "/head_camera/zed/rgb/color/rect/image",
    "left_rgb": "/wrist_camera_left/color/image_raw",
    "left_depth": "/wrist_camera_left/depth/image_rect_raw",
    "right_rgb": "/wrist_camera_right/color/image_raw",
    "right_depth": "/wrist_camera_right/depth/image_rect_raw",
}
MOTION_ENABLED = False
