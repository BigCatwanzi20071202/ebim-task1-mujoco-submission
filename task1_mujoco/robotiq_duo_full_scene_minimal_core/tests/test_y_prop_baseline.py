from __future__ import annotations

import unittest

import numpy as np

from teleop import config
from teleop.y_prop_baseline import (
    BimanualMaterialPair,
    measure_y_inspection_hold,
    measure_y_prop_route,
    pair_is_reachable,
    plan_y_base_stance,
    plan_y_inspection_transport,
    plan_y_prop_hold,
    select_bimanual_material_pair,
)


class YPropBaselineTest(unittest.TestCase):
    def test_material_pair_uses_distinct_ordered_tail_points(self) -> None:
        points = np.column_stack(
            (np.linspace(-0.2, 0.2, 20), np.zeros(20), np.zeros(20))
        )
        pair = select_bimanual_material_pair(
            points,
            np.array([-0.1, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            min_gap=3,
            max_gap=8,
            tail_count=20,
            min_point_distance=0.0,
        )
        self.assertLess(pair.left_index, pair.right_index)
        self.assertGreaterEqual(pair.right_index - pair.left_index, 3)

    def test_reachability_requires_both_arms(self) -> None:
        pair = BimanualMaterialPair(
            1,
            5,
            np.array([0.2, 0.0, 0.0]),
            np.array([0.4, 0.0, 0.0]),
            0.0,
        )
        self.assertTrue(
            pair_is_reachable(pair, np.zeros(3), np.zeros(3), max_reach=0.5)
        )
        self.assertFalse(
            pair_is_reachable(pair, np.zeros(3), np.zeros(3), max_reach=0.3)
        )

    def test_material_pair_search_rejects_unreachable_assignments(self) -> None:
        points = np.array(
            [
                [-0.9, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.9, 0.0, 0.0],
            ]
        )
        with self.assertRaises(ValueError):
            select_bimanual_material_pair(
                points,
                np.zeros(3),
                np.zeros(3),
                min_gap=1,
                max_gap=1,
                min_point_distance=0.0,
                max_reference_distance=0.5,
            )

    def test_material_pair_accounts_for_planned_base_translation(self) -> None:
        points = np.array(
            [
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        pair = select_bimanual_material_pair(
            points,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            min_gap=1,
            max_gap=1,
            min_point_distance=0.0,
            max_reference_distance=0.25,
            right_alignment_point=np.array([0.2, 0.0, 0.0]),
        )
        self.assertEqual((pair.left_index, pair.right_index), (0, 1))

    def test_material_pair_rejects_folded_points_without_gripper_clearance(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.05, 0.00, 0.0],
                [0.10, 0.00, 0.0],
                [0.02, 0.00, 0.0],
                [0.20, 0.00, 0.0],
            ]
        )
        pair = select_bimanual_material_pair(
            points,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.02, 0.0, 0.0]),
            min_gap=3,
            max_gap=4,
            tail_count=5,
            min_point_distance=0.15,
        )
        self.assertEqual({pair.left_index, pair.right_index}, {0, 4})
        self.assertGreaterEqual(
            np.linalg.norm(pair.right_point - pair.left_point), 0.15
        )

    def test_base_stance_remains_outside_table_boundary(self) -> None:
        points = np.array(
            [
                [2.45, 0.50, 0.0],
                [2.35, 0.60, 0.0],
                [2.20, 0.72, 0.0],
                [2.10, 0.82, 0.0],
                [2.00, 0.90, 0.0],
            ]
        )
        stance = plan_y_base_stance(
            points,
            np.array([2.67, 0.90]),
            np.array([2.50, 0.50, 0.0]),
            np.array([2.10, 0.90, 0.0]),
            np.array([2.45, 0.50, 0.0]),
            np.array([2.00, 0.90, 0.0]),
            right_x=2.67,
            top_y=1.50,
            right_y_range=(0.5, 0.9),
            top_x_range=(2.0, 2.67),
            samples=4,
            max_reach=0.6,
            min_point_distance=0.0,
        )
        self.assertTrue(
            stance.base_xy[0] == 2.67 or stance.base_xy[1] == 1.50
        )
        self.assertGreaterEqual(len(stance.waypoints), 1)

    def test_base_stance_scores_tcp_travel_not_shoulder_distance(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )
        stance = plan_y_base_stance(
            points,
            np.array([0.0, 0.0]),
            np.array([-0.3, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
            np.array([-0.1, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            right_x=0.0,
            top_y=1.0,
            right_y_range=(0.0, 0.0),
            top_x_range=(1.0, 1.0),
            samples=2,
            max_reach=0.6,
            min_point_distance=0.0,
        )
        np.testing.assert_allclose(stance.base_xy, [0.0, 0.0])

    def test_plan_transforms_local_targets(self) -> None:
        origin = np.array([1.0, 2.0, 0.1])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        plan = plan_y_prop_hold(origin, rotation)
        self.assertAlmostEqual(plan.target_span, 0.11)
        np.testing.assert_allclose(plan.entry, [1.05, 2.0, 0.118])

    def test_inspection_transport_preserves_pose_with_small_tension(self) -> None:
        left = np.array([-0.10, 0.0, 0.05])
        right = np.array([0.10, 0.0, 0.05])
        plan = plan_y_inspection_transport(
            left, right, np.array([1.0, 2.0, 0.0]), np.eye(3)
        )

        np.testing.assert_allclose(
            0.5 * (plan.left_target + plan.right_target), plan.midpoint
        )
        np.testing.assert_allclose(
            plan.right_target - plan.left_target, plan.relative
        )
        self.assertAlmostEqual(
            plan.target_span,
            np.linalg.norm(right - left)
            + config.Y_BASELINE_INSPECTION_TENSION,
        )

    def test_inspection_requires_center_span_and_settled_cable(self) -> None:
        plan = plan_y_inspection_transport(
            np.array([-0.1, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.zeros(3),
            np.eye(3),
        )
        velocities = np.zeros((4, 3))
        metrics = measure_y_inspection_hold(
            plan.left_target, plan.right_target, velocities, plan
        )
        self.assertTrue(metrics.passed)

        moving = velocities.copy()
        moving[0, 0] = config.Y_BASELINE_MAX_CABLE_SPEED + 0.1
        metrics = measure_y_inspection_hold(
            plan.left_target, plan.right_target, moving, plan
        )
        self.assertFalse(metrics.passed)

    def test_valid_branched_route_passes(self) -> None:
        points = np.array(
            [
                [0.0, -0.05, 0.01],
                [0.0, 0.004, 0.01],
                [-0.045, 0.05, 0.01],
                [0.045, 0.05, 0.01],
            ]
        )
        metrics = measure_y_prop_route(
            points, np.zeros_like(points), np.zeros(3), np.eye(3)
        )
        self.assertTrue(metrics.passed)
        self.assertAlmostEqual(metrics.branch_span, 0.09)

    def test_one_sided_route_does_not_validate(self) -> None:
        points = np.array(
            [
                [0.0, -0.05, 0.01],
                [0.0, 0.004, 0.01],
                [-0.03, 0.03, 0.01],
            ]
        )
        metrics = measure_y_prop_route(
            points, np.zeros_like(points), np.zeros(3), np.eye(3)
        )
        self.assertFalse(metrics.passed)
        self.assertFalse(metrics.has_right_branch)

    def test_unsettled_route_does_not_validate(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.01],
                [-0.045, 0.05, 0.01],
                [0.045, 0.05, 0.01],
            ]
        )
        velocity = np.zeros_like(points)
        velocity[1, 0] = 0.5
        metrics = measure_y_prop_route(
            points, velocity, np.zeros(3), np.eye(3)
        )
        self.assertFalse(metrics.passed)


if __name__ == "__main__":
    unittest.main()
