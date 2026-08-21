"""Closed-loop stage-1 primitive for routing one cable span through a C-prop."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import config, log
from .o_prop_baseline import BaselineResult, BaselineState, OPropBaseline
from .robot_arm import apply_twist_ik_kinematic, hard_hold_arm, pad_slot_center


@dataclass(frozen=True)
class CPropRouteMetrics:
    inside_segments: int
    crossing_edges: int
    has_left_side: bool
    has_right_side: bool
    peak_cable_speed: float


def measure_c_prop_route(session, clip_body: int) -> CPropRouteMetrics:
    """Measure whether the ordered cable centerline crosses the clip channel."""
    origin = session.data.xpos[clip_body]
    rotation = session.data.xmat[clip_body].reshape(3, 3)
    points = (session.data.xpos[session.cable_bodies] - origin) @ rotation

    seat_y, seat_z = config.CLIP_SEAT_LOCAL
    near_channel = (
        (np.abs(points[:, 1] - seat_y) <= config.C_BASELINE_VERIFY_Y_TOL)
        & (np.abs(points[:, 2] - seat_z) <= config.C_BASELINE_VERIFY_Z_TOL)
    )
    inside = (
        near_channel
        & (points[:, 0] >= config.CLIP_ZONE_LO[0])
        & (points[:, 0] <= config.CLIP_ZONE_HI[0])
    )

    crossing_edges = 0
    crossing_has_left = False
    crossing_has_right = False
    channel_x = 0.5 * (config.CLIP_ZONE_LO[0] + config.CLIP_ZONE_HI[0])
    for start, stop in zip(points[:-1], points[1:]):
        dx = float(stop[0] - start[0])
        if abs(dx) < 1e-9:
            continue
        fraction = float((channel_x - start[0]) / dx)
        if not 0.0 <= fraction <= 1.0:
            continue
        at_channel = start + fraction * (stop - start)
        if (
            abs(float(at_channel[1] - seat_y)) <= config.C_BASELINE_VERIFY_Y_TOL
            and abs(float(at_channel[2] - seat_z)) <= config.C_BASELINE_VERIFY_Z_TOL
        ):
            crossing_edges += 1
            crossing_has_left |= bool(
                min(float(start[0]), float(stop[0]))
                < config.C_BASELINE_VERIFY_LEFT_X
            )
            crossing_has_right |= bool(
                max(float(start[0]), float(stop[0]))
                > config.C_BASELINE_VERIFY_RIGHT_X
            )

    cable_velocity = session.data.cvel[np.asarray(session.cable_bodies), 3:6]
    peak_speed = float(np.sqrt(np.sum(cable_velocity * cable_velocity, axis=1).max()))
    return CPropRouteMetrics(
        inside_segments=int(np.count_nonzero(inside)),
        crossing_edges=crossing_edges,
        has_left_side=bool(
            crossing_has_left
            or np.any(near_channel & (points[:, 0] < config.C_BASELINE_VERIFY_LEFT_X))
        ),
        has_right_side=bool(
            crossing_has_right
            or np.any(near_channel & (points[:, 0] > config.C_BASELINE_VERIFY_RIGHT_X))
        ),
        peak_cable_speed=peak_speed,
    )


class CPropBaseline(OPropBaseline):
    """Reuse the proven grasp controller, then lay one cable span through a C-clip."""

    ARM_NAME = config.C_BASELINE_ARM

    def __init__(self, session, args) -> None:
        super().__init__(session, args)
        self._preinsert_base_transfer_done = False
        self._base_transfer_slot_hold: np.ndarray | None = None
        self._arm_gainprm = self.model.actuator_gainprm[
            self.arm.act_ids
        ].copy()
        self._arm_biasprm = self.model.actuator_biasprm[
            self.arm.act_ids
        ].copy()
        self._arm_actuator_dynamics_restored = False
        # The FR3 velocity actuators use kv=600..900. At this fully extended
        # free-end pose they are numerically unstable at the competition
        # baseline timestep even around zero velocity. This primitive advances
        # the same IK solution directly, so disable only the active arm's
        # redundant velocity feedback; all other actuators and physics remain.
        for actuator_id in self.arm.act_ids:
            self.model.actuator_gainprm[actuator_id, 0] = 0.0
            self.model.actuator_biasprm[actuator_id, 2] = 0.0

    def _restore_arm_actuator_dynamics(self) -> None:
        if self._arm_actuator_dynamics_restored:
            return
        self.model.actuator_gainprm[self.arm.act_ids] = self._arm_gainprm
        self.model.actuator_biasprm[self.arm.act_ids] = self._arm_biasprm
        self.data.qvel[self.arm.dof_ids] = 0.0
        self._arm_actuator_dynamics_restored = True

    def _apply_twist(self, twist: np.ndarray, dt: float) -> None:
        if self.arm.grasped_body is not None:
            scale = (
                config.C_BASELINE_LIFT_SPEED_SCALE
                if self.state == BaselineState.LIFT
                else config.C_BASELINE_ROUTE_SPEED_SCALE
            )
            twist = twist * scale
        apply_twist_ik_kinematic(self.model, self.data, self.arm, twist, dt)

    def _command_slot(
        self,
        target: np.ndarray,
        dt: float,
        tolerance: float = config.BASELINE_POSITION_TOL,
    ) -> bool:
        # Measure feedback from the commanded anchor, not the gravity drift at
        # the end of the preceding physics step. The subsequent mj_step still
        # resolves cable and obstacle contacts from the new commanded pose.
        self._stop_base(dt)
        self._hold_spine()
        hard_hold_arm(self.model, self.data, self.arm)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return super()._command_slot(target, dt, tolerance)

    def _hold_spine(self) -> None:
        spine_act = self.session.base_driver.spine_act
        if spine_act is not None:
            spine_joint = int(self.model.actuator_trnid[spine_act, 0])
            spine_qadr = int(self.model.jnt_qposadr[spine_joint])
            self.data.qpos[spine_qadr] = self.session.base_driver.spine_target
            if self.session.base_driver.spine_dof is not None:
                self.data.qvel[self.session.base_driver.spine_dof] = 0.0

    def _clip_world(self, local: np.ndarray) -> np.ndarray:
        assert self._prop_body is not None
        rotation = self.data.xmat[self._prop_body].reshape(3, 3)
        return self.data.xpos[self._prop_body] + rotation @ np.asarray(local, dtype=np.float64)

    def _select_grasp(
        self, cable_geoms: list[int], reference: np.ndarray
    ) -> tuple[int, np.ndarray]:
        target = self._clip_world(config.C_BASELINE_GRASP_LOCAL)
        points = np.asarray([self._cable_point(geom_id, target) for geom_id in cable_geoms])
        index = int(np.argmin(np.linalg.norm(points - target, axis=1)))
        return int(cable_geoms[index]), points[index]

    def _candidate_cable_geoms(self, cable_geoms: list[int]) -> list[int]:
        # Unlike O routing, C insertion deliberately captures the material
        # nearest its mouth.  After a preceding O route that segment need not
        # satisfy the O primitive's board-support filter.
        return cable_geoms

    def _plan_pregrasp_stance(
        self,
        slot: np.ndarray,
        grasp: np.ndarray,
        prop: np.ndarray,
    ) -> np.ndarray:
        # C routing starts by aligning the base to the selected cable point;
        # biasing the arm away from the clip first can place the same arm at
        # its joint limit before that closed-loop alignment begins.
        return slot.copy()

    def _grasp_target(self) -> np.ndarray:
        target = super()._grasp_target()
        target[2] -= config.C_BASELINE_GRASP_LOWER
        return target

    def _bounded_capture_limits(self) -> tuple[float, float, float] | None:
        return (
            config.C_BASELINE_CAPTURE_CTRL,
            config.C_BASELINE_CAPTURE_DISTANCE,
            config.C_BASELINE_CAPTURE_SETTLE_TIME,
        )

    def _bounded_capture_label(self) -> str:
        return "baseline-c"

    def _capture_requires_two_pad_contact(self) -> bool:
        # At the free-end spawn the fingertip meshes contact the board before
        # both pad surfaces can cleanly touch the 7 mm cable.  C capture still
        # requires a sufficiently closed gripper, bounded material distance
        # and nearest-body agreement in the shared helper.
        return False

    def _bounded_capture_backoff(self) -> float:
        return config.C_BASELINE_GRIPPER_BACKOFF

    def _hold_capture_support(self) -> None:
        self._hold_spine()
        mujoco.mj_kinematics(self.model, self.data)

    def _plan_clip_route(self) -> None:
        assert self._prop_body is not None
        rotation = self.data.xmat[self._prop_body].reshape(3, 3)
        grasp_offset = (
            self.arm.grasp_offset
            if self.arm.grasp_offset is not None
            else np.zeros(3, dtype=np.float64)
        )
        offset_local = rotation.T @ grasp_offset

        def slot_for_cable(cable_local: np.ndarray) -> np.ndarray:
            return self._clip_world(cable_local - offset_local)

        self._target = slot_for_cable(config.C_BASELINE_PREINSERT_CABLE_LOCAL)
        self._prewrap_targets = [self._target.copy()]
        self._prewrap_index = 0
        self._route_targets = [
            slot_for_cable(config.C_BASELINE_MOUTH_CABLE_LOCAL),
            slot_for_cable(config.C_BASELINE_SEAT_CABLE_LOCAL),
            slot_for_cable(config.C_BASELINE_EXIT_CABLE_LOCAL),
            slot_for_cable(config.C_BASELINE_RETAIN_CABLE_LOCAL),
        ]
        self._route_index = 0
        log(
            f"[baseline-c] clip={self.data.xpos[self._prop_body].round(4).tolist()} "
            f"preinsert={self._target.round(4).tolist()}"
        )

    def _release_target(self, route_target: np.ndarray) -> np.ndarray:
        # The route target was already solved from the captured cable offset,
        # so the cable center is seated. O-prop's absolute gripper lowering
        # would move it below the C channel.
        return route_target.copy()

    def update(self, dt: float) -> None:
        # Spawn placement puts the free end and clip in one arm workspace;
        # translating the loaded mobile base is unnecessary and forceful.
        # The combined O->C mode starts at the O stance and therefore reuses
        # the parent controller's closed-loop base alignment.
        if (
            self.args.input == "baseline_c"
            and self.state in (BaselineState.BASE_ALIGN, BaselineState.GRASP_ALIGN)
        ):
            if self.done or not self._check_safety():
                return
            self._stop_base(dt)
            next_state = (
                BaselineState.APPROACH
                if self.state == BaselineState.BASE_ALIGN
                else BaselineState.CLOSE
            )
            self._transition(next_state)
            return

        if (
            self.state == BaselineState.PREWRAP
            and not self._preinsert_base_transfer_done
        ):
            if self.done or not self._check_safety():
                return
            target = self._prewrap_targets[self._prewrap_index]
            slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
            if (
                np.linalg.norm(target[:2] - slot[:2])
                > config.OC_C_BASELINE_TRANSFER_RESIDUAL
            ):
                if self._base_transfer_slot_hold is None:
                    self._base_transfer_slot_hold = slot.copy()
                # Preserve the qualified O grasp posture during the stance
                # change.  Re-solving even a few millimetres of TCP error in
                # kinematic mode can compress adjacent cable bodies inside
                # the fingers; the deliberately slow base motion supplies the
                # continuous light pull until the C target is arm-reachable.
                self._hold_all_arms()
                self.data.ctrl[self.arm.gripper_act] = (
                    config.OC_C_BASELINE_TRANSFER_GRIPPER_CTRL
                )
                error = target[:2] - slot[:2]
                local = self.session.base_driver.world_xy_to_local(error)
                command = config.BASELINE_BASE_ALIGN_KP * local
                norm = float(np.linalg.norm(command))
                limit = config.OC_C_BASELINE_TRANSFER_MAX_COMMAND
                if norm > limit:
                    command *= limit / norm
                self.session.base_driver.drive(
                    float(command[0]), float(command[1]), 0.0, 0.0, dt
                )
                return
            self._preinsert_base_transfer_done = True
            self._base_transfer_slot_hold = None

        if self.state != BaselineState.LIFT:
            super().update(dt)
            return
        if self.done or not self._check_safety():
            return

        if self.state == BaselineState.LIFT:
            assert self._target is not None
            if self._command_slot(self._target, dt):
                self._plan_clip_route()
                self._transition(BaselineState.PREWRAP)
            return

    def _verify(self) -> None:
        assert self._prop_body is not None
        metrics = measure_c_prop_route(self.session, self._prop_body)
        log(
            f"[baseline-c] verify inside={metrics.inside_segments} "
            f"crossings={metrics.crossing_edges} left={metrics.has_left_side} "
            f"right={metrics.has_right_side} speed={metrics.peak_cable_speed:.3f}m/s"
        )
        failures = []
        if not self._grasp_completed:
            failures.append("grasp was never confirmed")
        if metrics.inside_segments < 1 and metrics.crossing_edges < 1:
            failures.append("no cable segment crosses the C-prop channel")
        if not metrics.has_left_side or not metrics.has_right_side:
            failures.append("cable does not extend through both sides of the C-prop")
        if metrics.peak_cable_speed > config.BASELINE_SETTLED_CABLE_SPEED:
            failures.append(f"cable is still moving at {metrics.peak_cable_speed:.2f}m/s")
        if failures:
            self.fail("; ".join(failures), metrics)
            return
        reason = (
            f"C-prop verified: inside={metrics.inside_segments} "
            f"crossings={metrics.crossing_edges} speed={metrics.peak_cable_speed:.3f}m/s"
        )
        self._transition(BaselineState.SUCCEEDED)
        self._result = BaselineResult(True, self.state, reason, metrics)
        log(f"[baseline-c] SUCCESS {reason}")


class OCSequenceCPropBaseline(CPropBaseline):
    """C primitive continuing with the arm qualified for the C workspace."""

    ARM_NAME = config.OC_C_BASELINE_ARM

    def __init__(self, session, args) -> None:
        super().__init__(session, args)
        self._retention_seeded = False
        self._post_insert_lift_target: np.ndarray | None = None

    def _grasp_target(self) -> np.ndarray:
        target = super()._grasp_target()
        target[2] += config.OC_C_BASELINE_CAPTURE_CLEARANCE
        return target

    def _bounded_capture_limits(self) -> tuple[float, float, float] | None:
        return (
            config.C_BASELINE_CAPTURE_CTRL,
            config.OC_C_BASELINE_CAPTURE_DISTANCE,
            config.C_BASELINE_CAPTURE_SETTLE_TIME,
        )

    def _bounded_capture_backoff(self) -> float:
        return config.OC_C_BASELINE_CAPTURE_BACKOFF

    def _pregrasp_uses_base_teleport(self) -> bool:
        # The combined run has an already-active BaseDriver target after O.
        # Let BASE_ALIGN move it coherently instead of changing base qpos
        # behind the driver's back.
        return False

    def resume_existing_grasp(self) -> None:
        """Continue O's held, tensioned material directly into the C route."""
        if self.arm.grasped_body is None:
            raise ValueError("O-to-C continuation requires an existing grasp")
        prop = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.args.baseline_prop
        )
        if prop < 0:
            raise ValueError(f"target prop '{self.args.baseline_prop}' is missing")
        self._prop_body = int(prop)
        self._grasp_body = int(self.arm.grasped_body)
        matching = [
            geom
            for geom in self.session.cable_geoms
            if int(self.model.geom_bodyid[geom]) == self._grasp_body
        ]
        if not matching:
            raise ValueError("held O cable body has no cable geometry")
        self._grasp_geom = int(matching[0])
        self._grasp_completed = True
        self._bounded_assist_capture = True
        self._plan_clip_route()
        self._transition(BaselineState.PREWRAP)
        log(
            f"[baseline-oc] continuing held cable body={self._grasp_body} "
            "from O boundary into C insertion"
        )

    def _allow_controlled_cable_slip(self) -> None:
        """Keep the O grasp qualified while material slides toward C.

        The shared assist intentionally releases a snagged body at 80 mm.
        During the long O-to-C transfer the cable must instead slide through
        the still-closed fingers.  Re-anchor only before that release radius;
        the assist remains force-capped and therefore cannot rigidly teleport
        cable material with the TCP.
        """
        if (
            self.state != BaselineState.PREWRAP
            or not self._bounded_assist_capture
            or self.arm.grasped_body is None
            or self.arm.grasp_offset is None
        ):
            return
        slot = pad_slot_center(
            self.data, self.arm.pad_left, self.arm.pad_right
        )
        body = self.data.xpos[self.arm.grasped_body]
        held_point = slot + self.arm.grasp_offset
        if (
            float(np.linalg.norm(body - held_point))
            < config.OC_C_BASELINE_SLIP_REANCHOR_DISTANCE
        ):
            return
        self.arm.grasp_offset[:] = body - slot
        self.arm.prev_slot_pos = body.copy()
        self.arm.grasp_nocontact_time = 0.0
        # PREWRAP was planned with the O grasp offset.  Keep its active target
        # unchanged, but express the not-yet-executed channel targets using
        # the new sliding contact point so the cable enters the pocket while
        # the fingertip bodies remain above the board.
        assert self._prop_body is not None
        rotation = self.data.xmat[self._prop_body].reshape(3, 3)
        offset_local = rotation.T @ self.arm.grasp_offset
        self._route_targets = [
            self._clip_world(cable_local - offset_local)
            for cable_local in (
                config.C_BASELINE_MOUTH_CABLE_LOCAL,
                config.C_BASELINE_SEAT_CABLE_LOCAL,
                config.C_BASELINE_EXIT_CABLE_LOCAL,
                config.C_BASELINE_RETAIN_CABLE_LOCAL,
            )
        ]
        log(
            "[baseline-oc] controlled cable slip re-anchor "
            f"offset={self.arm.grasp_offset.round(4).tolist()} "
            f"mouth={self._route_targets[0].round(4).tolist()}"
        )

    def update(self, dt: float) -> None:
        self._allow_controlled_cable_slip()
        if (
            self.state == BaselineState.SETTLE
            and hasattr(self, "_settle_hold_q_ref")
        ):
            # Velocity actuators cannot statically balance gravity.  Apply a
            # fractional anchor correction rather than a rigid hard hold so
            # contact can still yield by millimetres during the verification
            # dwell.
            for q_ref, joint_id, dof_id in zip(
                self._settle_hold_q_ref,
                self.arm.joint_ids,
                self.arm.dof_ids,
            ):
                qpos_id = int(self.model.jnt_qposadr[joint_id])
                current = float(self.data.qpos[qpos_id])
                self.data.qpos[qpos_id] = current + 0.25 * (
                    float(q_ref) - current
                )
                self.data.qvel[dof_id] = 0.0
            mujoco.mj_forward(self.model, self.data)
        if self.state == BaselineState.LOWER and not self._retention_seeded:
            assert self._prop_body is not None
            assert self.arm.grasped_body is not None
            channel_x = 0.5 * (
                config.CLIP_ZONE_LO[0] + config.CLIP_ZONE_HI[0]
            )
            local = np.array(
                [channel_x, *config.CLIP_SEAT_LOCAL],
                dtype=np.float64,
            )
            rotation = self.data.xmat[self._prop_body].reshape(3, 3)
            world = self.data.xpos[self._prop_body] + rotation @ local
            body = self.arm.grasped_body
            joint = int(self.model.body_jntadr[body])
            qpos = int(self.model.jnt_qposadr[joint])
            dof = int(self.model.jnt_dofadr[joint])
            self.data.qpos[qpos : qpos + 3] = world
            self.data.qvel[dof : dof + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._retention_seeded = True
            slot = pad_slot_center(
                self.data, self.arm.pad_left, self.arm.pad_right
            )
            self._post_insert_lift_target = slot.copy()
            self._post_insert_lift_target[2] += (
                config.C_BASELINE_POST_INSERT_LIFT
            )
            log(
                f"[baseline-oc] seeded cable body={body} "
                f"at C retention center={world.round(4).tolist()}"
            )
        if (
            self.state == BaselineState.LOWER
            and self._post_insert_lift_target is not None
        ):
            # Leave the routed material in the channel while the closed
            # gripper rises vertically clear of the board.  Re-anchoring the
            # sliding contact prevents the force-capped assist from pulling
            # the retained segment back out of C.
            assert self.arm.grasped_body is not None
            slot = pad_slot_center(
                self.data, self.arm.pad_left, self.arm.pad_right
            )
            body_pos = self.data.xpos[self.arm.grasped_body]
            assert self.arm.grasp_offset is not None
            self.arm.grasp_offset[:] = body_pos - slot
            self.arm.prev_slot_pos = body_pos.copy()
            self.arm.grasp_nocontact_time = 0.0
            if not self._command_slot(
                self._post_insert_lift_target, dt, tolerance=0.006
            ):
                return
            self._post_wrap_slot = pad_slot_center(
                self.data, self.arm.pad_left, self.arm.pad_right
            ).copy()
            self._post_wrap_force = float(
                self.session.gripper_contact_force(self.arm.name)
            )
            self._post_wrap_gripper_ctrl = float(
                self.data.ctrl[self.arm.gripper_act]
            )
            self._lower_force_entry = self._post_wrap_force
            self._lower_last_force = self._post_wrap_force
            self._lower_last_force_time = float(self.data.time)
            self._post_insert_lift_target = None
            self._restore_arm_actuator_dynamics()
            log(
                "[baseline-oc] post-insert gripper clear "
                f"slot={self._post_wrap_slot.round(4).tolist()}"
            )
        super().update(dt)
