"""Runtime for the stage-1 O-prop then C-prop autonomous sequence."""

from __future__ import annotations

from enum import Enum, auto

from . import config, log
from .c_prop_baseline import OCSequenceCPropBaseline
from .o_prop_baseline import OPropBaseline
from .robot_arm import hard_hold_arm
from .run_baseline import _step_until_done
from .run_c_baseline import configure_c_baseline
from .session import TeleopSession


class OCSequenceState(Enum):
    O_PROP = auto()
    TRANSITION = auto()
    C_PROP = auto()
    SUCCEEDED = auto()
    FAILED = auto()


def _run_sequence(session: TeleopSession, args, viewer=None) -> bool:
    state = OCSequenceState.O_PROP
    o_prop = args.baseline_prop
    c_prop = args.c_baseline_prop
    log(f"[baseline-oc] state={state.name} prop={o_prop}")
    o_controller = OPropBaseline(session, args)
    _step_until_done(session, o_controller, viewer)
    if o_controller.result is None or not o_controller.result.success:
        log("[baseline-oc] state=FAILED reason=O-prop primitive failed")
        return False

    state = OCSequenceState.TRANSITION
    log(f"[baseline-oc] state={state.name}")
    c_arm = session.arms[config.OC_C_BASELINE_ARM]
    if c_arm.grasped_body is None:
        log("[baseline-oc] state=FAILED reason=O released cable before C")
        return False
    for arm in session.arms.values():
        hard_hold_arm(session.model, session.data, arm)

    configure_c_baseline()
    config.BASELINE_MAX_CABLE_GRASP_FORCE = (
        config.OC_C_BASELINE_MAX_CABLE_GRASP_FORCE
    )
    config.BASELINE_MAX_OBSTACLE_FORCE = config.OC_C_BASELINE_MAX_OBSTACLE_FORCE
    config.GRASP_ASSIST_MAX_FORCE = config.OC_C_BASELINE_ASSIST_MAX_FORCE
    args.baseline_prop = c_prop
    state = OCSequenceState.C_PROP
    log(f"[baseline-oc] state={state.name} prop={c_prop}")
    c_controller = OCSequenceCPropBaseline(session, args)
    try:
        c_controller.resume_existing_grasp()
    except ValueError as exc:
        log(f"[baseline-oc] state=FAILED reason={exc}")
        return False
    _step_until_done(session, c_controller, viewer)
    if c_controller.result is None or not c_controller.result.success:
        log("[baseline-oc] state=FAILED reason=C-prop primitive failed")
        return False

    state = OCSequenceState.SUCCEEDED
    log(
        f"[baseline-oc] state={state.name} "
        f"O=({o_controller.result.reason}) C=({c_controller.result.reason})"
    )
    return True


def main(args) -> None:
    config.GRIPPER_FORCE_STOP = config.BASELINE_GRIPPER_FORCE_STOP
    session = TeleopSession(args)
    if args.no_viewer:
        success = _run_sequence(session, args)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(session.model, session.data) as viewer:
            session.setup_viewer_cam(viewer)
            success = _run_sequence(session, args, viewer)
    if not success:
        raise SystemExit(2)
