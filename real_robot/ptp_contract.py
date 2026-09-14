"""Validated PTP goal construction matching upstream Franka ROS 2 Jazzy."""
from dataclasses import dataclass
import numpy as np


def vector(value):
    array = np.asarray(value, float)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError("seven finite joint values required")
    return array


@dataclass(frozen=True)
class JointPTPGoal:
    goal_joint_configuration: tuple
    maximum_joint_velocities: tuple
    goal_tolerance: float

    def __post_init__(self):
        positions = vector(self.goal_joint_configuration)
        velocities = vector(self.maximum_joint_velocities)
        if not (velocities > 0).all() or not np.isfinite(self.goal_tolerance) or self.goal_tolerance <= 0:
            raise ValueError("finite positive velocity bounds and tolerance required")
        object.__setattr__(self, "goal_joint_configuration", tuple(positions.tolist()))
        object.__setattr__(self, "maximum_joint_velocities", tuple(velocities.tolist()))

    @classmethod
    def build(cls, positions, velocities, tolerance, *, joint_min, joint_max, velocity_cap):
        q, v, lower, upper, cap = map(vector, (positions, velocities, joint_min, joint_max, velocity_cap))
        if not (lower < upper).all() or not (cap > 0).all():
            raise ValueError("invalid installation limits")
        if not ((q > lower) & (q < upper)).all():
            raise ValueError("target outside joint range")
        if not ((v > 0) & (v <= cap)).all():
            raise ValueError("invalid maximum joint velocities")
        if not np.isfinite(tolerance) or not 0 < tolerance < float(np.min(upper - lower)) / 2:
            raise ValueError("invalid goal tolerance")
        return cls(tuple(q.tolist()), tuple(v.tolist()), float(tolerance))

    def to_ros_goal(self, action_type):
        goal = action_type.Goal()
        fields = goal.get_fields_and_field_types()
        expected = {"goal_joint_configuration", "maximum_joint_velocities", "goal_tolerance"}
        if set(fields) != expected:
            raise ValueError("installed PTPMotion goal fields differ from inspected upstream")
        goal.goal_joint_configuration = list(self.goal_joint_configuration)
        goal.maximum_joint_velocities = list(self.maximum_joint_velocities)
        goal.goal_tolerance = self.goal_tolerance
        return goal


def motion_result_reached(ros_goal_status, result):
    return ros_goal_status == 4 and result.target_status.status == 2 and not result.error_message
