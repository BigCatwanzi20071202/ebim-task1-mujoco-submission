"""Inspect the live ROS graph without publishing or sending action goals."""
import argparse
import json
from pathlib import Path

EXPECTED_TOPICS = {
    "/left/franka_robot_state_broadcaster/measured_joint_states": "sensor_msgs/msg/JointState",
    "/right/franka_robot_state_broadcaster/measured_joint_states": "sensor_msgs/msg/JointState",
    "/left/franka_robot_state_broadcaster/current_pose": "geometry_msgs/msg/PoseStamped",
    "/right/franka_robot_state_broadcaster/current_pose": "geometry_msgs/msg/PoseStamped",
    "/left/gripper/gripper_client/target_gripper_width_percent": "std_msgs/msg/Float32",
    "/right/gripper/gripper_client/target_gripper_width_percent": "std_msgs/msg/Float32",
}
EXPECTED_ACTIONS = {
    "/left/action_server/ptp_motion": "franka_msgs/action/PTPMotion",
    "/right/action_server/ptp_motion": "franka_msgs/action/PTPMotion",
    "/left/gripper/robotiq_gripper_controller/gripper_cmd": "control_msgs/action/GripperCommand",
    "/right/gripper/robotiq_gripper_controller/gripper_cmd": "control_msgs/action/GripperCommand",
}


def compare_graph(actual, expected):
    actual = {name: sorted(types) for name, types in actual}
    return {name: {"expected": kind, "actual": actual.get(name, []), "matches": kind in actual.get(name, [])}
            for name, kind in expected.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = {"commands_sent": 0, "publishers_created": 0, "action_clients_created": 0}
    try:
        import rclpy
        from franka_msgs.action import PTPMotion
    except ImportError as exc:
        result.update(status="ROS_ENVIRONMENT_REQUIRED", error=str(exc))
    else:
        fields = PTPMotion.Goal().get_fields_and_field_types()
        result["ptp_goal_fields"] = fields
        rclpy.init(args=None)
        node = rclpy.create_node("ggboy_interface_probe")
        try:
            result["topics"] = compare_graph(node.get_topic_names_and_types(), EXPECTED_TOPICS)
            result["actions"] = compare_graph(node.get_action_names_and_types(), EXPECTED_ACTIONS)
            result["all_required_interfaces_match"] = all(
                item["matches"] for group in (result["topics"], result["actions"])
                for item in group.values()
            )
            result["status"] = "READY_FOR_BOUNDED_INTERFACE_TEST" if result["all_required_interfaces_match"] else "INTERFACE_MISMATCH"
        finally:
            node.destroy_node()
            rclpy.shutdown()
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result.get("all_required_interfaces_match") else 2


if __name__ == "__main__":
    raise SystemExit(main())
