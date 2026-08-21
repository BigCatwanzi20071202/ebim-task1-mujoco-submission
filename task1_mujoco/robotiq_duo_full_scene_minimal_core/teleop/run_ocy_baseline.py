"""Runtime for the fixed-layout O -> C -> bimanual Y baseline."""

from __future__ import annotations

import mujoco

from . import config, log
from .robot_arm import hard_hold_arm, seed_arm, set_arm_ready_pose
from .run_baseline import _step_until_done
from .run_oc_baseline import _run_sequence
from .session import TeleopSession
from .y_prop_controller import YPropController


def _run_ocy(session: TeleopSession, args, viewer=None) -> bool:
    if not _run_sequence(session, args, viewer):
        log("[baseline-ocy] state=FAILED reason=O-C prefix failed")
        return False
    for arm in session.arms.values():
        set_arm_ready_pose(session.model, session.data, arm)
        session.data.ctrl[arm.gripper_act] = config.GRIPPER_OPEN
        session.data.qvel[arm.dof_ids] = 0.0
    left = session.arms["left"]
    for joint_id, qpos in zip(
        left.joint_ids, config.Y_BASELINE_LEFT_STOW_QPOS
    ):
        session.data.qpos[session.model.jnt_qposadr[joint_id]] = qpos
    mujoco.mj_forward(session.model, session.data)
    for arm in session.arms.values():
        seed_arm(session.model, session.data, arm)
        hard_hold_arm(session.model, session.data, arm)
    log("[baseline-ocy] state=Y_PROP")
    controller = YPropController(session, args)
    _step_until_done(session, controller, viewer)
    if controller.result is None or not controller.result.success:
        reason = (
            controller.result.reason
            if controller.result is not None
            else "Y controller ended without result"
        )
        log(f"[baseline-ocy] state=FAILED reason={reason}")
        return False
    log(f"[baseline-ocy] state=SUCCEEDED {controller.result.reason}")
    return True


def main(args) -> None:
    config.GRIPPER_FORCE_STOP = config.BASELINE_GRIPPER_FORCE_STOP
    session = TeleopSession(args)
    if args.no_viewer:
        success = _run_ocy(session, args)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            session.model, session.data
        ) as viewer:
            session.setup_viewer_cam(viewer)
            success = _run_ocy(session, args, viewer)
    if not success:
        raise SystemExit(2)
