import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy_core import HoldMonitor, validate_joint_state


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"joint{i}" for i in range(1, 8)]

    def test_valid_state(self):
        self.assertEqual(validate_joint_state(self.names, [0] * 7), (0.0,) * 7)

    def test_rejects_bad_state(self):
        invalid = [
            (self.names[:6], [0] * 6),
            (["same"] * 7, [0] * 7),
            (self.names, [0] * 6 + [float("nan")]),
        ]
        for names, positions in invalid:
            with self.subTest(names=names, positions=positions):
                with self.assertRaises(ValueError):
                    validate_joint_state(names, positions)

    def test_both_arms_required_and_stale_fails_closed(self):
        clock = FakeClock()
        monitor = HoldMonitor(stale_after_seconds=1.0, clock=clock)
        monitor.observe("left", self.names, [0] * 7)
        self.assertFalse(monitor.active())
        monitor.observe("right", self.names, [0] * 7)
        self.assertTrue(monitor.active())
        clock.now = 1.01
        self.assertFalse(monitor.active())

    def test_no_command_path(self):
        self.assertEqual(HoldMonitor().commands_sent(), 0)


if __name__ == "__main__":
    unittest.main()
