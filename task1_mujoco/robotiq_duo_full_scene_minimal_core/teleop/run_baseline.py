"""Runtime for the stage-1 autonomous O-prop controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import config, log
from .baseline_sequence import BaselineSequenceRunner
from .o_prop_baseline import BaselineResult, OPropBaseline
from .robot_arm import hard_hold_arm
from .session import TeleopSession

if TYPE_CHECKING:
    pass


def _step_until_done(
    session: TeleopSession, controller, viewer=None
) -> None:
    dt = float(session.model.opt.timestep)
    next_render = float(session.data.time)
    render_period = 1.0 / max(float(controller.args.render_hz), 1.0)
    while not controller.done:
        controller.update(dt)
        if controller.done:
            break
        session.step_once(dt)
        if viewer is not None and float(session.data.time) >= next_render:
            if not viewer.is_running():
                controller.fail("viewer was closed before completion")
                break
            viewer.sync()
            next_render = float(session.data.time) + render_period


def _settle_session(session: TeleopSession, viewer=None, steps: int = 50) -> None:
    dt = float(session.model.opt.timestep)
    for _ in range(steps):
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("viewer was closed during settle")
        for arm in session.arms.values():
            hard_hold_arm(session.model, session.data, arm)
        session.base_driver.drive(0.0, 0.0, 0.0, 0.0, dt)
        session.step_once(dt)


def _parse_baseline_props(args) -> list[str]:
    return parse_baseline_props(args)


def _run_baseline_step(
    session: TeleopSession,
    args,
    controller_type,
    viewer=None,
):
    controller = controller_type(session, args)
    if viewer is None:
        _step_until_done(session, controller)
    else:
        _step_until_done(session, controller, viewer)
    result = controller.result
    if result is None:
        raise RuntimeError("baseline ended without a result")
    return result


def run_controller(args, controller_type=OPropBaseline) -> None:
    # The generic teleop value intentionally squeezes hard for manual use.
    # This autonomous cable primitive stops at a lower measured two-pad force
    # so it can lift without first crushing the cable against a fingertip mesh.
    config.GRIPPER_FORCE_STOP = config.BASELINE_GRIPPER_FORCE_STOP
    session = TeleopSession(args)
    runner = BaselineSequenceRunner(
        session,
        args,
        controller_factory=controller_type,
        settle_callback=_settle_session,
    )
    if args.no_viewer:
        results = runner.run()
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(session.model, session.data) as viewer:
            session.setup_viewer_cam(viewer)
            results = runner.run(viewer)

    for index, result in enumerate(results, start=1):
        status = "success" if result.success else "failure"
        log(
            f"[baseline] RESULT step={index}/{len(results)} prop={result.prop} "
            f"status={status}: {result.reason}"
        )
        if not result.success:
            raise SystemExit(2)


def main(args) -> None:
    run_controller(args)
