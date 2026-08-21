"""Sequence runner for stage-1 baseline controllers.

This module is intentionally lightweight and independent of MuJoCo so the
sequence logic can be tested without simulator bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .baseline_args import parse_baseline_props
from . import log


@dataclass(frozen=True)
class BaselineSequenceResult:
    prop: str
    success: bool
    reason: str
    metrics: Any | None


class BaselineSequenceRunner:
    def __init__(
        self,
        session: Any,
        args: Any,
        controller_factory: Callable[[Any, Any], Any],
        settle_callback: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.session = session
        self.args = args
        self.controller_factory = controller_factory
        self.settle_callback = settle_callback
        self.props = parse_baseline_props(args)

    def _run_single(self, prop: str, viewer: Any = None) -> BaselineSequenceResult:
        self.args.baseline_prop = prop
        log(f"[baseline] running step prop={prop}")
        controller = self.controller_factory(self.session, self.args)
        dt = float(self.session.model.opt.timestep)
        next_render = float(self.session.data.time)
        render_period = 1.0 / max(float(controller.args.render_hz), 1.0)

        while not controller.done:
            controller.update(dt)
            if controller.done:
                break
            self.session.step_once(dt)
            if viewer is not None and float(self.session.data.time) >= next_render:
                if not viewer.is_running():
                    controller.fail("viewer was closed before completion")
                    break
                viewer.sync()
                next_render = float(self.session.data.time) + render_period

        result = controller.result
        if result is None:
            raise RuntimeError("baseline ended without a result")
        return BaselineSequenceResult(prop, result.success, result.reason, result.metrics)

    def run(self, viewer: Any = None) -> list[BaselineSequenceResult]:
        results: list[BaselineSequenceResult] = []
        for index, prop in enumerate(self.props, start=1):
            log(f"[baseline] step={index}/{len(self.props)} prop={prop}")
            result = self._run_single(prop, viewer)
            results.append(result)
            if not result.success:
                break
            if index < len(self.props) and self.settle_callback is not None:
                self.settle_callback(self.session, viewer)
        return results
