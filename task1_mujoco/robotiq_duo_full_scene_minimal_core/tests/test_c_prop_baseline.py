from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from teleop import config
from teleop.c_prop_baseline import measure_c_prop_route


def make_session(points: np.ndarray):
    cable_bodies = list(range(1, len(points) + 1))
    xpos = np.zeros((len(points) + 1, 3), dtype=np.float64)
    xpos[cable_bodies] = points
    xmat = np.tile(np.eye(3).reshape(1, 9), (len(points) + 1, 1))
    cvel = np.zeros((len(points) + 1, 6), dtype=np.float64)
    return SimpleNamespace(
        cable_bodies=cable_bodies,
        data=SimpleNamespace(xpos=xpos, xmat=xmat, cvel=cvel),
    )


class CPropRouteTest(unittest.TestCase):
    def test_continuous_edge_supplies_both_side_evidence(self) -> None:
        seat_y, seat_z = config.CLIP_SEAT_LOCAL
        channel_x = 0.5 * (config.CLIP_ZONE_LO[0] + config.CLIP_ZONE_HI[0])
        points = np.array(
            [
                [-0.05, seat_y, seat_z],
                [channel_x, seat_y, seat_z],
                [0.08, seat_y + 0.02, seat_z],
            ],
            dtype=np.float64,
        )

        metrics = measure_c_prop_route(make_session(points), 0)

        self.assertGreaterEqual(metrics.inside_segments, 1)
        self.assertGreaterEqual(metrics.crossing_edges, 1)
        self.assertTrue(metrics.has_left_side)
        self.assertTrue(metrics.has_right_side)

    def test_edge_above_channel_does_not_count(self) -> None:
        seat_y, seat_z = config.CLIP_SEAT_LOCAL
        points = np.array(
            [
                [-0.05, seat_y, seat_z + 0.04],
                [0.08, seat_y, seat_z + 0.04],
            ],
            dtype=np.float64,
        )

        metrics = measure_c_prop_route(make_session(points), 0)

        self.assertEqual(metrics.inside_segments, 0)
        self.assertEqual(metrics.crossing_edges, 0)


if __name__ == "__main__":
    unittest.main()
