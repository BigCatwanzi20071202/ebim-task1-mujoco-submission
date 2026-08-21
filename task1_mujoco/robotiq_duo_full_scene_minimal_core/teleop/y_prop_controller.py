"""Closed-loop bimanual controller for the fixed-layout Y checkpoint proxy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import mujoco
import numpy as np

from . import config, log
from .grasping import release_grasp
from .robot_arm import (
    apply_twist_ik_kinematic,
    clamp_tcp_twist_for_contact,
    hard_hold_arm,
    pad_slot_center,
    seed_arm,
    write_arm_ctrl,
)
from .y_prop_baseline import (
    BimanualMaterialPair,
    YInspectionMetrics,
    YInspectionPlan,
    measure_y_inspection_hold,
    pair_is_reachable,
    plan_y_base_stance,
    plan_y_inspection_transport,
)


class YPropState(Enum):
    LOCATE = auto()
    BASE_ALIGN = auto()
    APPROACH_RIGHT = auto()
    CAPTURE_RIGHT = auto()
    APPROACH_LEFT = auto()
    CAPTURE_LEFT = auto()
    LIFT = auto()
    TRANSPORT = auto()
    INSPECTION_HOLD = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class YPropResult:
    success: bool
    state: YPropState
    reason: str
    metrics: YInspectionMetrics | None


class YPropController:
    """Select -> staged capture -> Y route -> tension -> verify.

    Capture uses the same bounded grasp-assist representation as the current
    C primitive.  It remains deliberately capped and distance-gated; replacing
    this development capture with physical two-pad contact does not change the
    remaining state machine.
    """

    def __init__(self, session, args) -> None:
        self.session = session
        self.args = args
        self.model = session.model
        self.data = session.data
        self.left = session.arms["left"]
        self.right = session.arms["right"]
        self.state = YPropState.LOCATE
        self._run_started = float(self.data.time)
        self._state_started = float(self.data.time)
        self._prop_body: int | None = None
        self._pair: BimanualMaterialPair | None = None
        self._left_body: int | None = None
        self._right_body: int | None = None
        self._left_target: np.ndarray | None = None
        self._right_target: np.ndarray | None = None
        self._base_waypoints: list[np.ndarray] = []
        self._capture_started: float | None = None
        self._tension_started: float | None = None
        self._inspection_plan: YInspectionPlan | None = None
        self._transport_base_target: np.ndarray | None = None
        self._transport_final_base_target: np.ndarray | None = None
        self._transport_base_phase = "ESCAPE"
        self._frozen_left_grasp_offset: np.ndarray | None = None
        self._frozen_right_grasp_offset: np.ndarray | None = None
        self._compensated_left_target: np.ndarray | None = None
        self._compensated_right_target: np.ndarray | None = None
        self._inspection_settle_stable_time = 0.0
        self._arms_previous_slots: dict[str, np.ndarray] = {}
        self._arms_tcp_speeds: dict[str, float] = {}
        self._arms_command_speeds: dict[str, float] = {}
        self._arms_start_slots: dict[str, np.ndarray] = {}
        self._arms_progress = 0.0
        self._arms_midpoint_progress = 0.0
        self._arms_span_progress = 0.0
        self._arms_initial_midpoint: np.ndarray | None = None
        self._arms_final_midpoint: np.ndarray | None = None
        self._arms_initial_span: np.ndarray | None = None
        self._arms_final_span: np.ndarray | None = None
        self._arms_previous_path_targets: dict[str, np.ndarray] = {}
        self._arms_span_speed = 0.0
        self._arms_speed_scale = 1.0
        self._arms_cable_peak_speed = 0.0
        self._arms_cable_peak_body = ""
        self._result: YPropResult | None = None
        self._last_log = -1.0
        self._transport_debug_time: float | None = None
        self._transport_debug_error: float | None = None
        self._transport_debug_left_offset: np.ndarray | None = None
        self._transport_debug_right_offset: np.ndarray | None = None
        for arm in (self.left, self.right):
            stale_body = arm.grasped_body
            arm.close_ramp = False
            release_grasp(self.data, arm)
            arm.prev_slot_pos = pad_slot_center(
                self.data, arm.pad_left, arm.pad_right
            ).copy()
            if stale_body is not None:
                log(
                    f"[baseline-y] cleared inherited {arm.name} grasp "
                    f"body={stale_body}"
                )
        for arm in (self.left, self.right):
            for actuator_id in arm.act_ids:
                self.model.actuator_gainprm[actuator_id, 0] = 0.0
                self.model.actuator_biasprm[actuator_id, 2] = 0.0
        enabled = 0
        for geom_name in (
            "yclip_0_stem",
            "yclip_0_left_branch",
            "yclip_0_right_branch",
        ):
            geom = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom >= 0:
                self.model.geom_contype[geom] = 1
                self.model.geom_conaffinity[geom] = 3
                enabled += 1
        mujoco.mj_forward(self.model, self.data)
        if enabled != 3:
            self.fail(f"Y checkpoint collision geometry incomplete ({enabled}/3)")
        log(f"[baseline-y] state={self.state.name} prop={args.y_baseline_prop}")

    @property
    def done(self) -> bool:
        return self.state in (YPropState.SUCCEEDED, YPropState.FAILED)

    @property
    def result(self) -> YPropResult | None:
        return self._result

    def _transition(self, state: YPropState) -> None:
        self.state = state
        self._state_started = float(self.data.time)
        self._capture_started = None
        if state == YPropState.INSPECTION_HOLD:
            for arm in (self.left, self.right):
                arm.grasp_assist_force_scale = (
                    config.Y_BASELINE_INSPECTION_ASSIST_SCALE
                )
            self.session.cable_velocity_decay = (
                config.Y_BASELINE_INSPECTION_CABLE_VELOCITY_DECAY
            )
            log(
                "[baseline-y] HOLD grasp-assist scale="
                f"{config.Y_BASELINE_INSPECTION_ASSIST_SCALE:.3f} "
                f"cable-decay={config.Y_BASELINE_INSPECTION_CABLE_VELOCITY_DECAY:.3f}/s"
            )
        log(f"[baseline-y] state={state.name} t={self.data.time:.3f}s")

    def _stop_base(self, dt: float) -> None:
        self.session.base_driver.drive(0.0, 0.0, 0.0, 0.0, dt)

    def _hold_spine(self) -> None:
        driver = self.session.base_driver
        if driver.spine_act is None:
            return
        joint = int(self.model.actuator_trnid[driver.spine_act, 0])
        self.data.qpos[self.model.jnt_qposadr[joint]] = driver.spine_target
        if driver.spine_dof is not None:
            self.data.qvel[driver.spine_dof] = 0.0

    def _safe_hold(self, dt: float) -> None:
        self._stop_base(dt)
        for arm in (self.left, self.right):
            hard_hold_arm(self.model, self.data, arm)

    def fail(self, reason: str) -> None:
        if self.done:
            return
        self.state = YPropState.FAILED
        self._result = YPropResult(False, self.state, reason, None)
        for arm in (self.left, self.right):
            arm.close_ramp = False
            seed_arm(self.model, self.data, arm)
            write_arm_ctrl(self.model, self.data, arm)
        log(f"[baseline-y] FAILED {reason}")

    def _check_safety(self, dt: float) -> bool:
        if float(self.data.time) - self._run_started > config.Y_BASELINE_TOTAL_TIMEOUT:
            self.fail("global timeout")
            self._safe_hold(dt)
            return False
        if (
            float(self.data.time) - self._state_started
            > config.Y_BASELINE_STATE_TIMEOUT
        ):
            self.fail(f"state timeout in {self.state.name}")
            self._safe_hold(dt)
            return False
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            self.fail("non-finite simulation state")
            self._safe_hold(dt)
            return False
        for arm in (self.left, self.right):
            total = self.session.gripper_contact_force(arm.name)
            if total <= config.Y_BASELINE_MAX_OBSTACLE_FORCE:
                continue
            owned_material: set[int] = set()
            if arm.grasped_body in self.session.cable_bodies:
                owned_index = self.session.cable_bodies.index(
                    arm.grasped_body
                )
                owned_material = set(
                    self.session.cable_bodies[
                        max(
                            0,
                            owned_index - config.Y_BASELINE_PAIR_MAX_GAP,
                        ) : owned_index + config.Y_BASELINE_PAIR_MAX_GAP + 1
                    ]
                )
            cable_force = 0.0
            obstacle_force = 0.0
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if (
                    g1 not in self.session.haptic_geoms[arm.name]
                    and g2 not in self.session.haptic_geoms[arm.name]
                ):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                force = abs(float(wrench[0]))
                other = g2 if g1 in self.session.haptic_geoms[arm.name] else g1
                if other in self.session.cable_geoms:
                    # The simulator-only bounded assist already caps the force
                    # on its one owned material body.  Counting that body's
                    # simultaneous rigid pad contact here double-counts the
                    # same grasp constraint; adjacent cable bodies remain
                    # fully covered by this independent contact-force gate.
                    if (
                        int(self.model.geom_bodyid[other])
                        in owned_material
                    ):
                        continue
                    cable_force += force
                else:
                    obstacle_force += force
            if obstacle_force > config.Y_BASELINE_MAX_OBSTACLE_FORCE:
                self.fail(
                    f"{arm.name} obstacle contact force {obstacle_force:.1f} N"
                )
                self._safe_hold(dt)
                return False
            if cable_force > config.Y_BASELINE_MAX_CABLE_GRASP_FORCE:
                self.fail(f"{arm.name} cable grasp force {cable_force:.1f} N")
                self._safe_hold(dt)
                return False
        return True

    def _command_arm(
        self,
        arm,
        target: np.ndarray,
        dt: float,
        tolerance: float | None = None,
        terminal_slowdown: bool = False,
    ) -> bool:
        tolerance = config.Y_BASELINE_POSITION_TOL if tolerance is None else tolerance
        self._hold_spine()
        hard_hold_arm(self.model, self.data, arm)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        slot = pad_slot_center(self.data, arm.pad_left, arm.pad_right)
        error = np.asarray(target) - slot
        distance = float(np.linalg.norm(error))
        if terminal_slowdown:
            previous_slot = self._arms_previous_slots.get(arm.name)
            tcp_speed = (
                0.0
                if previous_slot is None
                else float(np.linalg.norm(slot - previous_slot)) / max(dt, 1e-9)
            )
            self._arms_previous_slots[arm.name] = slot.copy()
            self._arms_tcp_speeds[arm.name] = tcp_speed
        if distance <= tolerance:
            seed_arm(self.model, self.data, arm)
            write_arm_ctrl(self.model, self.data, arm)
            if terminal_slowdown:
                self._arms_command_speeds[arm.name] = 0.0
                return (
                    self._arms_tcp_speeds[arm.name]
                    <= config.Y_BASELINE_ARMS_HOLD_ENTRY_SPEED
                )
            return True
        if float(self.data.time) - self._last_log >= 1.0:
            self._last_log = float(self.data.time)
            log(
                f"[baseline-y] progress state={self.state.name} "
                f"arm={arm.name} error={distance:.4f}m "
                f"slot={slot.round(4).tolist()} "
                f"target={np.asarray(target).round(4).tolist()}"
            )
        speed = min(
            config.Y_BASELINE_MAX_TCP_SPEED,
            config.Y_BASELINE_SERVO_KP * distance,
        )
        if terminal_slowdown:
            slowdown_span = max(
                config.Y_BASELINE_ARMS_SLOWDOWN_DISTANCE - tolerance,
                1e-9,
            )
            slowdown_phase = np.clip(
                (distance - tolerance) / slowdown_span, 0.0, 1.0
            )
            smooth_phase = slowdown_phase**2 * (3.0 - 2.0 * slowdown_phase)
            terminal_speed = config.Y_BASELINE_ARMS_TERMINAL_SPEED
            speed = min(
                speed,
                terminal_speed
                + (config.Y_BASELINE_MAX_TCP_SPEED - terminal_speed)
                * smooth_phase,
            )
            self._arms_command_speeds[arm.name] = speed
        twist = np.zeros(6)
        twist[:3] = error * speed / max(distance, 1e-9)
        twist = clamp_tcp_twist_for_contact(
            self.model, twist, arm.grasped_body is not None
        )
        apply_twist_ik_kinematic(self.model, self.data, arm, twist, dt)
        return False

    def _command_both(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        dt: float,
        terminal_slowdown: bool = False,
    ) -> bool:
        left_done = self._command_arm(
            self.left, left_target, dt, terminal_slowdown=terminal_slowdown
        )
        right_done = self._command_arm(
            self.right, right_target, dt, terminal_slowdown=terminal_slowdown
        )
        self._stop_base(dt)
        if float(self.data.time) - self._last_log >= 1.0:
            self._last_log = float(self.data.time)
            left_slot = pad_slot_center(
                self.data, self.left.pad_left, self.left.pad_right
            )
            right_slot = pad_slot_center(
                self.data, self.right.pad_left, self.right.pad_right
            )
            log(
                f"[baseline-y] progress state={self.state.name} "
                f"left_error={np.linalg.norm(left_target-left_slot):.4f}m "
                f"right_error={np.linalg.norm(right_target-right_slot):.4f}m"
            )
        return left_done and right_done

    def _command_synchronized_inspection(self, dt: float) -> bool:
        """Interleave independent midpoint and delayed-span progress paths."""
        assert self._compensated_left_target is not None
        assert self._compensated_right_target is not None
        assert self._arms_initial_midpoint is not None
        assert self._arms_final_midpoint is not None
        assert self._arms_initial_span is not None
        assert self._arms_final_span is not None
        targets = {
            "left": self._compensated_left_target,
            "right": self._compensated_right_target,
        }
        arms = {"left": self.left, "right": self.right}
        self._hold_spine()
        for arm in arms.values():
            hard_hold_arm(self.model, self.data, arm)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        slots = {
            name: pad_slot_center(self.data, arm.pad_left, arm.pad_right)
            for name, arm in arms.items()
        }
        for name, slot in slots.items():
            previous_slot = self._arms_previous_slots[name]
            self._arms_tcp_speeds[name] = (
                float(np.linalg.norm(slot - previous_slot)) / max(dt, 1e-9)
            )
            self._arms_previous_slots[name] = slot.copy()

        remaining_distance = max(
            float(np.linalg.norm(targets[name] - slots[name])) for name in arms
        )
        tolerance = config.Y_BASELINE_POSITION_TOL
        slowdown_span = max(
            config.Y_BASELINE_ARMS_SLOWDOWN_DISTANCE - tolerance, 1e-9
        )
        slowdown_phase = np.clip(
            (remaining_distance - tolerance) / slowdown_span, 0.0, 1.0
        )
        smooth_phase = slowdown_phase**2 * (3.0 - 2.0 * slowdown_phase)
        shared_speed = (
            config.Y_BASELINE_ARMS_TERMINAL_SPEED
            + (
                config.Y_BASELINE_MAX_TCP_SPEED
                - config.Y_BASELINE_ARMS_TERMINAL_SPEED
            )
            * smooth_phase
        )
        cable_bodies = np.asarray(self.session.cable_bodies)
        cable_speeds = np.linalg.norm(
            self.data.cvel[cable_bodies, 3:6], axis=1
        )
        peak_index = int(np.argmax(cable_speeds))
        self._arms_cable_peak_speed = float(cable_speeds[peak_index])
        peak_body_id = int(cable_bodies[peak_index])
        self._arms_cable_peak_body = (
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, peak_body_id
            )
            or str(peak_body_id)
        )
        if self._arms_cable_peak_speed <= 1.0:
            target_speed_scale = 1.0
        elif self._arms_cable_peak_speed <= 2.0:
            phase = self._arms_cable_peak_speed - 1.0
            smooth = phase**2 * (3.0 - 2.0 * phase)
            target_speed_scale = 1.0 - 0.25 * smooth
        else:
            phase = np.clip(
                (self._arms_cable_peak_speed - 2.0) / 2.0, 0.0, 1.0
            )
            smooth = phase**2 * (3.0 - 2.0 * phase)
            target_speed_scale = 0.75 - 0.15 * smooth
        scale_alpha = 1.0 - np.exp(-dt / 0.25)
        self._arms_speed_scale += scale_alpha * (
            target_speed_scale - self._arms_speed_scale
        )
        shared_speed *= self._arms_speed_scale

        midpoint_delta = (
            self._arms_final_midpoint - self._arms_initial_midpoint
        )
        span_delta = self._arms_final_span - self._arms_initial_span
        time_progress = self._arms_progress
        midpoint_derivative = (
            30.0 * time_progress**2 * (1.0 - time_progress) ** 2
        )
        if time_progress <= 0.25:
            span_time = 0.0
            span_derivative = 0.0
        else:
            span_time = np.clip((time_progress - 0.25) / 0.75, 0.0, 1.0)
            span_derivative = (
                30.0 * span_time**2 * (1.0 - span_time) ** 2 / 0.75
            )
        left_path_derivative = (
            midpoint_delta * midpoint_derivative
            + 0.5 * span_delta * span_derivative
        )
        right_path_derivative = (
            midpoint_delta * midpoint_derivative
            - 0.5 * span_delta * span_derivative
        )
        derivative_floor = 0.10 * max(
            float(np.linalg.norm(midpoint_delta)),
            0.5 * float(np.linalg.norm(span_delta)),
            1e-6,
        )
        path_derivative_norm = max(
            float(np.linalg.norm(left_path_derivative)),
            float(np.linalg.norm(right_path_derivative)),
            derivative_floor,
        )
        if self._arms_progress < 1.0:
            progress_rate = shared_speed / path_derivative_norm
            self._arms_progress = min(
                1.0, self._arms_progress + progress_rate * dt
            )
        midpoint_time = self._arms_progress
        self._arms_midpoint_progress = (
            10.0 * midpoint_time**3
            - 15.0 * midpoint_time**4
            + 6.0 * midpoint_time**5
        )
        if midpoint_time <= 0.25:
            span_time = 0.0
        else:
            span_time = np.clip((midpoint_time - 0.25) / 0.75, 0.0, 1.0)
        self._arms_span_progress = (
            10.0 * span_time**3
            - 15.0 * span_time**4
            + 6.0 * span_time**5
        )
        path_midpoint = (
            self._arms_initial_midpoint
            + self._arms_midpoint_progress * midpoint_delta
        )
        path_span = (
            self._arms_initial_span
            + self._arms_span_progress * span_delta
        )
        path_targets = {
            "left": path_midpoint + 0.5 * path_span,
            "right": path_midpoint - 0.5 * path_span,
        }
        self._arms_span_speed = float(
            np.linalg.norm(
                path_span
                - (
                    self._arms_previous_path_targets["left"]
                    - self._arms_previous_path_targets["right"]
                )
            )
            / max(dt, 1e-9)
        )
        for name, arm in arms.items():
            feedforward = (
                path_targets[name]
                - self._arms_previous_path_targets[name]
            ) / max(dt, 1e-9)
            correction = config.Y_BASELINE_SERVO_KP * (
                path_targets[name] - slots[name]
            )
            command = feedforward + correction
            command_norm = float(np.linalg.norm(command))
            arm_speed_limit = shared_speed
            if command_norm > arm_speed_limit > 0.0:
                command *= arm_speed_limit / command_norm
            self._arms_command_speeds[name] = float(np.linalg.norm(command))
            twist = np.zeros(6)
            twist[:3] = command
            twist = clamp_tcp_twist_for_contact(
                self.model, twist, arm.grasped_body is not None
            )
            apply_twist_ik_kinematic(self.model, self.data, arm, twist, dt)
        self._arms_previous_path_targets = {
            name: target.copy() for name, target in path_targets.items()
        }

        final_errors = {
            name: float(np.linalg.norm(targets[name] - slots[name]))
            for name in arms
        }
        phase_position_done = self._arms_progress >= 1.0 and all(
            final_errors[name] <= tolerance for name in arms
        )
        converged = phase_position_done and all(
            self._arms_tcp_speeds[name]
            <= config.Y_BASELINE_ARMS_HOLD_ENTRY_SPEED
            for name in arms
        )
        if converged:
            for arm in arms.values():
                seed_arm(self.model, self.data, arm)
                write_arm_ctrl(self.model, self.data, arm)
            for name in arms:
                self._arms_command_speeds[name] = 0.0
        return converged

    def _locate(self) -> None:
        prop = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.args.y_baseline_prop
        )
        if prop < 0:
            self.fail(f"target prop '{self.args.y_baseline_prop}' is missing")
            return
        self._prop_body = int(prop)
        left_reference = self.data.xanchor[self.left.joint_ids[0]]
        right_reference = self.data.xanchor[self.right.joint_ids[0]]
        left_slot = pad_slot_center(
            self.data, self.left.pad_left, self.left.pad_right
        )
        right_slot = pad_slot_center(
            self.data, self.right.pad_left, self.right.pad_right
        )
        points = self.data.xpos[np.asarray(self.session.cable_bodies)].copy()
        try:
            stance = plan_y_base_stance(
                points,
                self.data.xpos[self.session.base_driver.base_body][:2],
                left_reference,
                right_reference,
                left_slot,
                right_slot,
            )
        except ValueError as exc:
            self.fail(str(exc))
            return
        self._pair = stance.pair
        self._base_waypoints = [point.copy() for point in stance.waypoints]
        self._left_body = int(
            self.session.cable_bodies[self._pair.left_index]
        )
        self._right_body = int(
            self.session.cable_bodies[self._pair.right_index]
        )
        self._left_target = self._pair.left_point + np.array(
            [0.0, 0.0, config.Y_BASELINE_APPROACH_HEIGHT]
        )
        self._right_target = self._pair.right_point + np.array(
            [0.0, 0.0, config.Y_BASELINE_APPROACH_HEIGHT]
        )
        log(
            f"[baseline-y] selected material left={self._pair.left_index} "
            f"right={self._pair.right_index} gap="
            f"{abs(self._pair.right_index-self._pair.left_index)} "
            f"score={self._pair.score:.3f} "
            f"base={stance.base_xy.round(3).tolist()}"
        )
        self._transition(YPropState.BASE_ALIGN)

    def _base_align(self, dt: float) -> None:
        assert self._pair is not None
        if not self._base_waypoints:
            self.fail("base stance has no waypoint")
            return
        base_xy = self.data.xpos[self.session.base_driver.base_body][:2]
        target = self._base_waypoints[0]
        error = target - base_xy
        distance = float(np.linalg.norm(error))
        if float(self.data.time) - self._last_log >= 1.0:
            self._last_log = float(self.data.time)
            log(
                f"[baseline-y] progress state={self.state.name} "
                f"error={distance:.4f}m "
                f"target_xy={target.round(4).tolist()} "
                f"base_xy={base_xy.round(4).tolist()}"
            )
        for arm in (self.left, self.right):
            hard_hold_arm(self.model, self.data, arm)
        if distance <= config.Y_BASELINE_BASE_ALIGN_TOL:
            self._stop_base(dt)
            self._base_waypoints.pop(0)
            if self._base_waypoints:
                return
            left_shoulder = self.data.xanchor[self.left.joint_ids[0]]
            right_shoulder = self.data.xanchor[self.right.joint_ids[0]]
            live_pair = BimanualMaterialPair(
                self._pair.left_index,
                self._pair.right_index,
                self.data.xpos[self._left_body].copy(),
                self.data.xpos[self._right_body].copy(),
                self._pair.score,
            )
            if not pair_is_reachable(
                live_pair, left_shoulder, right_shoulder
            ):
                self.fail("selected material pair is outside conservative arm reach")
                return
            self._transition(YPropState.APPROACH_RIGHT)
            return
        local = self.session.base_driver.world_xy_to_local(error)
        command = config.Y_BASELINE_BASE_ALIGN_KP * local
        norm = float(np.linalg.norm(command))
        if norm > config.Y_BASELINE_BASE_ALIGN_MAX_COMMAND:
            command *= config.Y_BASELINE_BASE_ALIGN_MAX_COMMAND / norm
        self.session.base_driver.drive(
            float(command[0]), float(command[1]), 0.0, 0.0, dt
        )

    def _capture(self, arm, body: int) -> bool:
        slot = pad_slot_center(self.data, arm.pad_left, arm.pad_right)
        distance = float(np.linalg.norm(self.data.xpos[body] - slot))
        target_index = self.session.cable_bodies.index(body)
        target_neighborhood = set(
            self.session.cable_bodies[
                max(0, target_index - 1) : target_index + 2
            ]
        )
        # ``update_grasp`` runs after this controller and may consume the
        # two-pad contact while leaving its selected material body attached.
        # Preserve that one-tick contact evidence when it belongs to the
        # requested segment neighborhood.
        target_contact = arm.grasped_body in target_neighborhood
        gripper_geoms = self.session.haptic_geoms[arm.name]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in gripper_geoms:
                other = g2
            elif g2 in gripper_geoms:
                other = g1
            else:
                continue
            if (
                other in self.session.cable_geoms
                and int(self.model.geom_bodyid[other])
                in target_neighborhood
            ):
                target_contact = True
                break
        # At the protected 45 mm slot height the cable can be centered in the
        # closing envelope without sustaining a rigid-pad contact.  Treat
        # that bounded, nearest-material envelope as the safe target-contact
        # region so capture completes before the fingers squeeze the board-
        # supported cable into a high-force contact.
        target_contact = target_contact or (
            distance <= config.Y_BASELINE_CAPTURE_DISTANCE
        )
        if (
            distance <= config.Y_BASELINE_CAPTURE_DISTANCE
            and target_contact
            and self._capture_started is None
        ):
            arm.grasped_body = body
            arm.grasped_neighbors = []
            assert arm.grasp_offset is not None
            arm.grasp_offset[:] = self.data.xpos[body] - slot
            arm.prev_slot_pos = self.data.xpos[body].copy()
            arm.grasp_assist_age = 0.0
            arm.grasp_nocontact_time = 0.0
            arm.close_ramp = False
            hold_ctrl = config.Y_BASELINE_GRIPPER_HOLD_CTRL
            self.data.ctrl[arm.gripper_act] = hold_ctrl
            self._capture_started = float(self.data.time)
            log(
                f"[baseline-y] {arm.name} bounded capture body={body} "
                f"distance={distance:.4f}m hold_ctrl={hold_ctrl:.3f}"
            )
        if arm.grasped_body != body or self._capture_started is None:
            return False
        arm.grasp_nocontact_time = 0.0
        return (
            float(self.data.time) - self._capture_started
            >= config.Y_BASELINE_CAPTURE_SETTLE_TIME
        )

    def _capture_target(self, body: int) -> np.ndarray:
        cable = self.data.xpos[body]
        target = cable + np.array(
            [0.0, 0.0, config.Y_BASELINE_APPROACH_HEIGHT]
        )
        target[2] = max(
            float(target[2]),
            config.Y_BASELINE_CAPTURE_MIN_SLOT_Z,
        )
        return target

    def _try_state_capture_before_safety(self) -> None:
        if self.state == YPropState.CAPTURE_RIGHT and self._right_body is not None:
            self._capture(self.right, self._right_body)
        elif self.state == YPropState.CAPTURE_LEFT and self._left_body is not None:
            self._capture(self.left, self._left_body)

    def _require_dual_grasp(self) -> bool:
        if (
            self.left.grasped_body != self._left_body
            or self.right.grasped_body != self._right_body
        ):
            self.fail("dual grasp was lost")
            return False
        return True

    def _retain_bimanual_grasps(self) -> None:
        """Allow bounded material slip without dropping either cable end."""
        for arm in (self.left, self.right):
            if arm.grasped_body is None or arm.grasp_offset is None:
                continue
            arm.grasp_nocontact_time = 0.0
            slot = pad_slot_center(
                self.data, arm.pad_left, arm.pad_right
            )
            body = self.data.xpos[arm.grasped_body]
            held_point = slot + arm.grasp_offset
            if float(np.linalg.norm(body - held_point)) < 0.060:
                continue
            if self._frozen_left_grasp_offset is not None:
                continue
            arm.grasp_offset[:] = body - slot
            arm.prev_slot_pos = body.copy()
            log(
                f"[baseline-y] {arm.name} controlled cable slip "
                f"offset={arm.grasp_offset.round(4).tolist()}"
            )

    def _freeze_inspection_grasps(self) -> None:
        """Freeze material anchors and convert body targets to slot targets."""
        if self._frozen_left_grasp_offset is not None:
            return
        assert self._inspection_plan is not None
        assert self.left.grasp_offset is not None
        assert self.right.grasp_offset is not None
        self._frozen_left_grasp_offset = self.left.grasp_offset.copy()
        self._frozen_right_grasp_offset = self.right.grasp_offset.copy()
        self.left.grasp_release_locked = True
        self.right.grasp_release_locked = True
        self._compensated_left_target = (
            self._inspection_plan.left_target
            - self._frozen_left_grasp_offset
        )
        self._compensated_right_target = (
            self._inspection_plan.right_target
            - self._frozen_right_grasp_offset
        )
        self._arms_start_slots = {
            "left": pad_slot_center(
                self.data, self.left.pad_left, self.left.pad_right
            ).copy(),
            "right": pad_slot_center(
                self.data, self.right.pad_left, self.right.pad_right
            ).copy(),
        }
        self._arms_previous_slots = {
            name: slot.copy() for name, slot in self._arms_start_slots.items()
        }
        self._arms_initial_midpoint = 0.5 * (
            self._arms_start_slots["left"]
            + self._arms_start_slots["right"]
        )
        self._arms_final_midpoint = 0.5 * (
            self._compensated_left_target
            + self._compensated_right_target
        )
        self._arms_initial_span = (
            self._arms_start_slots["left"]
            - self._arms_start_slots["right"]
        )
        self._arms_final_span = (
            self._compensated_left_target
            - self._compensated_right_target
        )
        self._arms_previous_path_targets = {
            name: slot.copy() for name, slot in self._arms_start_slots.items()
        }
        self._arms_progress = 0.0
        self._arms_midpoint_progress = 0.0
        self._arms_span_progress = 0.0
        self._arms_span_speed = 0.0
        self._arms_speed_scale = 1.0
        self._arms_cable_peak_speed = 0.0
        self._arms_cable_peak_body = ""
        log(
            "[baseline-y] ARMS grasp offsets frozen "
            f"left_frozen_offset={self._frozen_left_grasp_offset.round(4).tolist()} "
            f"right_frozen_offset={self._frozen_right_grasp_offset.round(4).tolist()} "
            f"left_compensated_tcp_target={self._compensated_left_target.round(4).tolist()} "
            f"right_compensated_tcp_target={self._compensated_right_target.round(4).tolist()}"
        )

    def _metrics(self) -> YInspectionMetrics:
        assert self._inspection_plan is not None
        velocities = self.data.cvel[
            np.asarray(self.session.cable_bodies), 3:6
        ]
        return measure_y_inspection_hold(
            self.data.xpos[self._left_body],
            self.data.xpos[self._right_body],
            velocities,
            self._inspection_plan,
        )

    def _log_transport_debug(
        self, phase: str, base_distance: float
    ) -> None:
        now = float(self.data.time)
        if self._transport_debug_time is not None:
            elapsed = max(now - self._transport_debug_time, 1e-9)
            error_delta = base_distance - float(self._transport_debug_error)
            error_rate = error_delta / elapsed
        else:
            error_delta = 0.0
            error_rate = 0.0
        left_slot = pad_slot_center(
            self.data, self.left.pad_left, self.left.pad_right
        )
        right_slot = pad_slot_center(
            self.data, self.right.pad_left, self.right.pad_right
        )
        left_offset = (
            np.zeros(3)
            if self.left.grasp_offset is None
            else self.left.grasp_offset.copy()
        )
        right_offset = (
            np.zeros(3)
            if self.right.grasp_offset is None
            else self.right.grasp_offset.copy()
        )
        left_offset_delta = (
            0.0
            if self._transport_debug_left_offset is None
            else float(
                np.linalg.norm(left_offset - self._transport_debug_left_offset)
            )
        )
        right_offset_delta = (
            0.0
            if self._transport_debug_right_offset is None
            else float(
                np.linalg.norm(right_offset - self._transport_debug_right_offset)
            )
        )
        assert self._transport_base_target is not None
        assert self._inspection_plan is not None
        base_xy = self.data.xpos[
            self.session.base_driver.base_body
        ][:2]
        spine_support_contacts = 0
        spine_support_peak = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            names = {
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                ),
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                ),
            }
            if names != {
                "franka_spine_collision",
                "table_cable_support_back",
            }:
                continue
            spine_support_contacts += 1
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model, self.data, contact_index, contact_force
            )
            spine_support_peak = max(
                spine_support_peak,
                float(np.linalg.norm(contact_force[:3])),
            )
        log(
            f"[baseline-y-debug] phase={phase} "
            f"base_target={self._transport_base_target.round(4).tolist()} "
            f"base_xy={base_xy.round(4).tolist()} "
            f"base_error={base_distance:.4f}m "
            f"error_delta={error_delta:+.4f}m "
            f"error_rate={error_rate:+.4f}m/s "
            f"left_target={self._inspection_plan.left_target.round(4).tolist()} "
            f"left_tcp={left_slot.round(4).tolist()} "
            f"right_target={self._inspection_plan.right_target.round(4).tolist()} "
            f"right_tcp={right_slot.round(4).tolist()} "
            f"left_grasp_offset={left_offset.round(4).tolist()} "
            f"left_offset_delta={left_offset_delta:.4f}m "
            f"right_grasp_offset={right_offset.round(4).tolist()} "
            f"right_offset_delta={right_offset_delta:.4f}m "
            f"spine_support_contacts={spine_support_contacts} "
            f"spine_support_peak={spine_support_peak:.1f}N"
        )
        self._transport_debug_time = now
        self._transport_debug_error = base_distance
        self._transport_debug_left_offset = left_offset
        self._transport_debug_right_offset = right_offset

    def update(self, dt: float) -> None:
        self._retain_bimanual_grasps()
        self._try_state_capture_before_safety()
        if self.done or not self._check_safety(dt):
            return
        if self.state == YPropState.LOCATE:
            self._locate()
            return
        if self.state == YPropState.BASE_ALIGN:
            self._base_align(dt)
            return
        assert self._left_body is not None and self._right_body is not None
        if self.state == YPropState.APPROACH_RIGHT:
            left_target = self._capture_target(self._left_body)
            right_target = self._capture_target(self._right_body)
            # Pre-position the second arm before the first capture.  Holding
            # board-supported cable in one hand while the other traverses from
            # its high parking pose loads adjacent cable bodies unnecessarily.
            self._command_arm(self.left, left_target, dt, tolerance=0.010)
            if self._command_arm(
                self.right, right_target, dt, tolerance=0.010
            ):
                self.right.close_ramp = True
                self._transition(YPropState.CAPTURE_RIGHT)
                self._capture(self.right, self._right_body)
            self._stop_base(dt)
            return
        if self.state == YPropState.CAPTURE_RIGHT:
            hard_hold_arm(self.model, self.data, self.left)
            self._command_arm(
                self.right,
                self._capture_target(self._right_body),
                dt,
            )
            self._stop_base(dt)
            if self._capture(self.right, self._right_body):
                self._transition(YPropState.APPROACH_LEFT)
            return
        if self.state == YPropState.APPROACH_LEFT:
            hard_hold_arm(self.model, self.data, self.right)
            target = self._capture_target(self._left_body)
            if self._command_arm(self.left, target, dt, tolerance=0.010):
                self.left.close_ramp = True
                self._transition(YPropState.CAPTURE_LEFT)
                self._capture(self.left, self._left_body)
            self._stop_base(dt)
            return
        if self.state == YPropState.CAPTURE_LEFT:
            hard_hold_arm(self.model, self.data, self.right)
            self._command_arm(
                self.left,
                self._capture_target(self._left_body),
                dt,
            )
            self._stop_base(dt)
            if self._capture(self.left, self._left_body):
                left_slot = pad_slot_center(
                    self.data, self.left.pad_left, self.left.pad_right
                )
                right_slot = pad_slot_center(
                    self.data, self.right.pad_left, self.right.pad_right
                )
                self._left_target = left_slot + np.array(
                    [0.0, 0.0, config.Y_BASELINE_LIFT_HEIGHT]
                )
                self._right_target = right_slot + np.array(
                    [0.0, 0.0, config.Y_BASELINE_LIFT_HEIGHT]
                )
                self._transition(YPropState.LIFT)
            return
        if not self._require_dual_grasp():
            self._safe_hold(dt)
            return
        assert self._prop_body is not None
        rotation = self.data.xmat[self._prop_body].reshape(3, 3)
        if self.state == YPropState.LIFT:
            assert self._left_target is not None and self._right_target is not None
            if self._command_both(
                self._left_target, self._right_target, dt
            ):
                left_slot = pad_slot_center(
                    self.data, self.left.pad_left, self.left.pad_right
                )
                right_slot = pad_slot_center(
                    self.data, self.right.pad_left, self.right.pad_right
                )
                self._inspection_plan = plan_y_inspection_transport(
                    left_slot,
                    right_slot,
                    self.data.xpos[self._prop_body],
                    rotation,
                )
                self._transport_final_base_target = np.array(
                    [2.80, 1.00], dtype=np.float64
                )
                self._transport_base_target = np.array(
                    [3.10, 0.92], dtype=np.float64
                )
                self._transport_base_phase = "ESCAPE"
                log(
                    "[baseline-y] TRANSPORT phase=ESCAPE "
                    f"target={self._transport_base_target.tolist()} "
                    f"final={self._transport_final_base_target.round(4).tolist()}"
                )
                self._transport_debug_time = None
                self._transport_debug_error = None
                self._transport_debug_left_offset = None
                self._transport_debug_right_offset = None
                self._transition(YPropState.TRANSPORT)
            return
        assert self._inspection_plan is not None
        if self.state == YPropState.TRANSPORT:
            self.data.ctrl[self.left.gripper_act] = config.Y_BASELINE_GRIPPER_HOLD_CTRL
            self.data.ctrl[self.right.gripper_act] = config.Y_BASELINE_GRIPPER_HOLD_CTRL
            assert self._transport_base_target is not None
            base_xy = self.data.xpos[
                self.session.base_driver.base_body
            ][:2]
            base_error = self._transport_base_target - base_xy
            base_distance = float(np.linalg.norm(base_error))
            if base_distance > config.Y_BASELINE_BASE_ALIGN_TOL:
                # Move the loaded bimanual frame as one rigid pose first;
                # this preserves hand separation and avoids asking either arm
                # to reach across the full board to the Y checkpoint.
                hard_hold_arm(self.model, self.data, self.left)
                hard_hold_arm(self.model, self.data, self.right)
                local = self.session.base_driver.world_xy_to_local(
                    base_error
                )
                command = config.Y_BASELINE_BASE_ALIGN_KP * local
                norm = float(np.linalg.norm(command))
                if norm > config.Y_BASELINE_BASE_ALIGN_MAX_COMMAND:
                    command *= (
                        config.Y_BASELINE_BASE_ALIGN_MAX_COMMAND / norm
                    )
                self.session.base_driver.drive(
                    float(command[0]),
                    float(command[1]),
                    0.0,
                    0.0,
                    dt,
                )
                if float(self.data.time) - self._last_log >= 1.0:
                    self._last_log = float(self.data.time)
                    self._log_transport_debug(
                        self._transport_base_phase, base_distance
                    )
                return
            if self._transport_base_phase == "ESCAPE":
                assert self._transport_final_base_target is not None
                self._stop_base(dt)
                self._transport_base_target = (
                    self._transport_final_base_target.copy()
                )
                self._transport_base_phase = "FINAL"
                self._transport_debug_time = None
                self._transport_debug_error = None
                log(
                    "[baseline-y] TRANSPORT phase=FINAL "
                    f"target={self._transport_base_target.round(4).tolist()}"
                )
                return
            self._freeze_inspection_grasps()
            assert self._compensated_left_target is not None
            assert self._compensated_right_target is not None
            left_slot = pad_slot_center(
                self.data, self.left.pad_left, self.left.pad_right
            )
            right_slot = pad_slot_center(
                self.data, self.right.pad_left, self.right.pad_right
            )
            if float(self.data.time) - self._last_log >= 1.0:
                self._last_log = float(self.data.time)
                self._log_transport_debug("ARMS", base_distance)
                midpoint = 0.5 * (left_slot + right_slot)
                span_vector = left_slot - right_slot
                span_length = float(np.linalg.norm(span_vector))
                log(
                    "[baseline-y] ARMS error "
                    f"midpoint_progress={self._arms_midpoint_progress:.4f} "
                    f"span_progress={self._arms_span_progress:.4f} "
                    f"midpoint={midpoint.round(4).tolist()} "
                    f"span_length={span_length:.4f}m "
                    f"span_vector={span_vector.round(4).tolist()} "
                    f"span_speed={self._arms_span_speed:.4f}m/s "
                    f"speed_scale={self._arms_speed_scale:.4f} "
                    f"left_tcp_speed={self._arms_tcp_speeds.get('left', 0.0):.4f}m/s "
                    f"right_tcp_speed={self._arms_tcp_speeds.get('right', 0.0):.4f}m/s "
                    f"left_command_speed={self._arms_command_speeds.get('left', 0.0):.4f}m/s "
                    f"right_command_speed={self._arms_command_speeds.get('right', 0.0):.4f}m/s "
                    f"cable_peak_speed={self._arms_cable_peak_speed:.4f}m/s "
                    f"peak_body={self._arms_cable_peak_body}"
                )
            if self._command_synchronized_inspection(dt):
                left_slot = pad_slot_center(
                    self.data, self.left.pad_left, self.left.pad_right
                )
                right_slot = pad_slot_center(
                    self.data, self.right.pad_left, self.right.pad_right
                )
                log(
                    "[baseline-y] ARMS converged "
                    f"midpoint_progress={self._arms_midpoint_progress:.4f} "
                    f"span_progress={self._arms_span_progress:.4f} "
                    f"left_error={np.linalg.norm(self._compensated_left_target-left_slot):.4f}m "
                    f"right_error={np.linalg.norm(self._compensated_right_target-right_slot):.4f}m "
                    f"speed_scale={self._arms_speed_scale:.4f} "
                    f"left_tcp_speed={self._arms_tcp_speeds.get('left', 0.0):.4f}m/s "
                    f"right_tcp_speed={self._arms_tcp_speeds.get('right', 0.0):.4f}m/s "
                    f"left_command_speed={self._arms_command_speeds.get('left', 0.0):.4f}m/s "
                    f"right_command_speed={self._arms_command_speeds.get('right', 0.0):.4f}m/s"
                )
                self._transition(YPropState.INSPECTION_HOLD)
            return
        if self.state == YPropState.INSPECTION_HOLD:
            self.data.ctrl[self.left.gripper_act] = config.Y_BASELINE_GRIPPER_HOLD_CTRL
            self.data.ctrl[self.right.gripper_act] = config.Y_BASELINE_GRIPPER_HOLD_CTRL
            self._safe_hold(dt)
            if (
                float(self.data.time) - self._state_started
                < config.Y_BASELINE_INSPECTION_HOLD_TIME
            ):
                return
            cable_bodies = np.asarray(self.session.cable_bodies)
            velocities = self.data.cvel[cable_bodies, 3:6]
            cable_speeds = np.linalg.norm(velocities, axis=1)
            peak_index = int(np.argmax(cable_speeds))
            max_speed = float(cable_speeds[peak_index])
            peak_body_id = int(cable_bodies[peak_index])
            peak_body = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                peak_body_id,
            ) or str(peak_body_id)
            if max_speed < config.Y_BASELINE_INSPECTION_SETTLE_MAX_SPEED:
                self._inspection_settle_stable_time += float(dt)
            else:
                self._inspection_settle_stable_time = 0.0
            elapsed = float(self.data.time) - self._state_started
            if float(self.data.time) - self._last_log >= 1.0:
                self._last_log = float(self.data.time)
                log(
                    "[baseline-y] INSPECTION_SETTLE "
                    f"max_speed={max_speed:.3f}m/s "
                    f"peak_body={peak_body} "
                    f"stable_time={self._inspection_settle_stable_time:.3f}/"
                    f"{config.Y_BASELINE_INSPECTION_SETTLE_TIME:.2f}s "
                    f"elapsed={elapsed:.3f}s"
                )
            if (
                self._inspection_settle_stable_time
                < config.Y_BASELINE_INSPECTION_SETTLE_TIME
            ):
                return
            log(
                "[baseline-y] INSPECTION_SETTLE passed "
                f"max_speed={max_speed:.3f}m/s "
                f"peak_body={peak_body} "
                f"stable_time={self._inspection_settle_stable_time:.3f}s "
                f"elapsed={elapsed:.3f}s"
            )
            metrics = self._metrics()
            if not metrics.passed:
                self.fail(
                    "Y inspection failed: "
                    f"center_error={metrics.center_error:.3f}m "
                    f"span_error={metrics.span_error:.3f}m "
                    f"speed={metrics.peak_cable_speed:.3f}m/s"
                )
                return
            self.state = YPropState.SUCCEEDED
            reason = (
                f"Y inspection held {config.Y_BASELINE_INSPECTION_HOLD_TIME:.1f}s: "
                f"center_error={metrics.center_error:.3f}m "
                f"span_error={metrics.span_error:.3f}m "
                f"speed={metrics.peak_cable_speed:.3f}m/s"
            )
            self._result = YPropResult(True, self.state, reason, metrics)
            log(f"[baseline-y] SUCCESS {reason}")
