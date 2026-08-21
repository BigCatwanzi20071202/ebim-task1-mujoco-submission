from __future__ import annotations

import unittest
from types import SimpleNamespace

from teleop.baseline_args import parse_baseline_props
from teleop.baseline_sequence import BaselineSequenceRunner


class BaselineArgsParseTest(unittest.TestCase):
    def test_parse_baseline_props_list(self) -> None:
        args = SimpleNamespace(
            baseline_props="round_peg_0, round_peg_1,round_peg_2",
            baseline_prop="round_peg_3",
            baseline_repeats=1,
        )

        props = parse_baseline_props(args)

        self.assertEqual(props, ["round_peg_0", "round_peg_1", "round_peg_2"])

    def test_parse_baseline_repeats(self) -> None:
        args = SimpleNamespace(
            baseline_props=None,
            baseline_prop="round_peg_1",
            baseline_repeats=3,
        )

        props = parse_baseline_props(args)

        self.assertEqual(props, ["round_peg_1", "round_peg_1", "round_peg_1"])

    def test_parse_baseline_default_single_prop(self) -> None:
        args = SimpleNamespace(
            baseline_props=None,
            baseline_prop="round_peg_2",
            baseline_repeats=1,
        )

        props = parse_baseline_props(args)

        self.assertEqual(props, ["round_peg_2"])


class DummySession:
    def __init__(self) -> None:
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
        self.data = SimpleNamespace(time=0.0)
        self.step_count = 0

    def step_once(self, dt: float) -> None:
        self.step_count += 1
        self.data.time += dt


class DummyController:
    def __init__(self, session: DummySession, args: SimpleNamespace) -> None:
        self.args = args
        self.done = False
        self.result = None
        self._update_count = 0

    def update(self, dt: float) -> None:
        self._update_count += 1
        if self._update_count >= getattr(self.args, "finish_after", 3):
            self.done = True
            self.result = SimpleNamespace(success=True, reason="ok", metrics=None)


class BaselineSequenceRunnerTest(unittest.TestCase):
    def test_sequence_runner_runs_all_props(self) -> None:
        args = SimpleNamespace(
            baseline_props="round_peg_0,round_peg_1",
            baseline_prop="round_peg_0",
            baseline_repeats=1,
            render_hz=60.0,
            finish_after=2,
        )
        session = DummySession()
        events: list[str] = []

        def settle_callback(sess: DummySession, viewer: object | None) -> None:
            events.append("settle")
            sess.step_once(0.01)

        runner = BaselineSequenceRunner(
            session,
            args,
            controller_factory=lambda sess, a: DummyController(sess, a),
            settle_callback=settle_callback,
        )

        results = runner.run()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        self.assertEqual(events, ["settle"])


if __name__ == "__main__":
    unittest.main()
