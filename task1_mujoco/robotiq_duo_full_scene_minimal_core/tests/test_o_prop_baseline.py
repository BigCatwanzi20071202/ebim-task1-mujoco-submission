from __future__ import annotations

import math
import unittest

import numpy as np

from teleop import config
from teleop.o_prop_baseline import (
    plan_o_prop_route,
    plan_pregrasp_stance,
    signed_local_winding,
)


class SignedLocalWindingTest(unittest.TestCase):
    def test_counter_clockwise_arc_is_positive(self) -> None:
        angles = np.linspace(0.0, math.pi / 2.0, 9)
        points = np.column_stack((0.1 * np.cos(angles), 0.1 * np.sin(angles)))

        winding, nearby = signed_local_winding(points, np.zeros(2), 0.11)

        self.assertAlmostEqual(winding, math.pi / 2.0, places=6)
        self.assertEqual(nearby, len(points))

    def test_clockwise_arc_is_negative(self) -> None:
        angles = np.linspace(0.0, -math.pi, 13)
        points = np.column_stack((0.08 * np.cos(angles), 0.08 * np.sin(angles)))

        winding, _ = signed_local_winding(points, np.zeros(2), 0.1)

        self.assertAlmostEqual(winding, -math.pi, places=6)

    def test_distant_curve_does_not_contribute(self) -> None:
        points = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)

        winding, nearby = signed_local_winding(points, np.zeros(2), 0.2)

        self.assertEqual(winding, 0.0)
        self.assertEqual(nearby, 0)


class OPropPlanningTest(unittest.TestCase):
    def test_distant_prop_biases_pregrasp_posture_away_from_target(self) -> None:
        slot = np.array([0.4, 0.2, 0.1], dtype=np.float64)
        grasp = np.array([0.3, 0.2, 0.0], dtype=np.float64)
        prop = np.array([-0.3, 0.2, 0.0], dtype=np.float64)

        target = plan_pregrasp_stance(slot, grasp, prop)

        self.assertGreater(target[0], slot[0])
        self.assertAlmostEqual(target[1], slot[1])
        self.assertAlmostEqual(target[2], slot[2])
        self.assertAlmostEqual(
            np.linalg.norm(target[:2] - slot[:2]),
            config.BASELINE_STANCE_MAX_SHIFT,
        )

    def test_nearby_prop_preserves_verified_pregrasp_posture(self) -> None:
        slot = np.array([0.4, 0.2, 0.1], dtype=np.float64)
        grasp = np.array([0.3, 0.2, 0.0], dtype=np.float64)
        prop = np.array([0.4, 0.2, 0.0], dtype=np.float64)

        target = plan_pregrasp_stance(slot, grasp, prop)

        np.testing.assert_allclose(target, slot)

    def test_prewrap_translates_at_lift_height_then_descends(self) -> None:
        prop = np.array([1.2, 0.4, -0.01], dtype=np.float64)
        slot = np.array([1.8, 0.9, 0.12], dtype=np.float64)

        plan = plan_o_prop_route(prop, slot, 1.0)

        raised, high, low = plan.prewrap_targets
        np.testing.assert_allclose(raised[:2], slot[:2])
        np.testing.assert_allclose(high[:2], low[:2])
        self.assertAlmostEqual(
            raised[2], prop[2] + config.BASELINE_TRANSIT_CLEARANCE
        )
        self.assertAlmostEqual(high[2], raised[2])
        self.assertAlmostEqual(low[2], prop[2] + config.BASELINE_O_ROUTE_Z)

    def test_route_waypoints_follow_requested_direction(self) -> None:
        prop = np.zeros(3, dtype=np.float64)
        slot = np.array([0.2, 0.0, 0.1], dtype=np.float64)

        ccw = plan_o_prop_route(prop, slot, 1.0)
        cw = plan_o_prop_route(prop, slot, -1.0)
        ccw_angles = np.unwrap(
            [math.atan2(point[1], point[0]) for point in ccw.route_targets]
        )
        cw_angles = np.unwrap(
            [math.atan2(point[1], point[0]) for point in cw.route_targets]
        )

        self.assertTrue(np.all(np.diff(ccw_angles) > 0.0))
        self.assertTrue(np.all(np.diff(cw_angles) < 0.0))

    def test_route_uses_o_boundary_and_tangent_lead(self) -> None:
        prop = np.zeros(3, dtype=np.float64)
        slot = np.array([0.2, 0.0, 0.1], dtype=np.float64)

        plan = plan_o_prop_route(prop, slot, 1.0)
        first_angle = config.BASELINE_ROUTE_ARC / config.BASELINE_ROUTE_WAYPOINTS
        radial = np.array([math.cos(first_angle), math.sin(first_angle)])
        tangent = np.array([-math.sin(first_angle), math.cos(first_angle)])
        relative = plan.route_targets[0][:2] - prop[:2]

        self.assertAlmostEqual(float(relative @ radial), plan.boundary_radius)
        self.assertAlmostEqual(float(relative @ tangent), plan.tangent_lead)
        self.assertLess(np.linalg.norm(relative), 0.04)
        self.assertAlmostEqual(
            plan.required_length,
            plan.boundary_radius * plan.route_arc + plan.tangent_lead,
        )

    def test_nearby_prop_keeps_direct_prewrap(self) -> None:
        prop = np.zeros(3, dtype=np.float64)
        slot = np.array([0.1, 0.0, 0.1], dtype=np.float64)

        plan = plan_o_prop_route(prop, slot, 1.0)

        self.assertEqual(len(plan.prewrap_targets), 1)


if __name__ == "__main__":
    unittest.main()
