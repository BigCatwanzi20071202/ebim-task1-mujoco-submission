import unittest
from types import SimpleNamespace

from real_robot.fr3_interface import FR3Interface
from real_robot.gripper_interface import GripperInterface
from real_robot.ptp_contract import JointPTPGoal, motion_result_reached
from real_robot.site_probe import compare_graph
from real_robot.task_state_machine import TaskStateMachine


class Goal:
    def get_fields_and_field_types(self):
        return {
            "goal_joint_configuration": "sequence<double>",
            "maximum_joint_velocities": "sequence<double>",
            "goal_tolerance": "double",
        }


class RealInterfaceTests(unittest.TestCase):
    def make_goal(self):
        return JointPTPGoal.build(
            [0.0] * 7, [0.1] * 7, 0.01,
            joint_min=[-1.0] * 7, joint_max=[1.0] * 7,
            velocity_cap=[0.2] * 7,
        )

    def test_goal_contract(self):
        message = self.make_goal().to_ros_goal(SimpleNamespace(Goal=Goal))
        self.assertEqual(message.maximum_joint_velocities, [0.1] * 7)

    def test_result_requires_ros_and_franka_success(self):
        result = SimpleNamespace(target_status=SimpleNamespace(status=2), error_message="")
        self.assertTrue(motion_result_reached(4, result))
        self.assertFalse(motion_result_reached(6, result))

    def test_motion_defaults_off(self):
        with self.assertRaises(PermissionError):
            FR3Interface().send_ptp("left", self.make_goal())
        with self.assertRaises(PermissionError):
            GripperInterface().command("left", 0.5)

    def test_graph_mismatch_is_visible(self):
        checked = compare_graph([], {"/required": "pkg/msg/Type"})
        self.assertFalse(checked["/required"]["matches"])

    def test_preview_never_claims_success(self):
        states = TaskStateMachine(("o", "c", "y")).dry_run_states()
        self.assertFalse(any("SUCCEEDED" in state for state in states))


if __name__ == "__main__":
    unittest.main()
