"""Closed-loop stage-1 baseline for one round (O) cable-routing prop.

The locator deliberately reads MuJoCo state.  Motion, grasping, contact
limits, and physics all reuse the simulator's existing control stack.  A
future vision locator can replace :meth:`_locate` without changing the state
machine or low-level controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import mujoco
import numpy as np

from . import config, log
from .grasping import open_gripper, pad_cable_contacts
from .mjutil import nearest_bodies_to_point
from .scene import (
    board_safe_cable_geoms,
    teleport_base_slot_translation_to_target,
)
from .robot_arm import (
    Arm,
    apply_twist_ik,
    apply_twist_ik_kinematic,
    clamp_tcp_twist_for_contact,
    hard_hold_arm,
    pad_slot_center,
    seed_arm,
    write_arm_ctrl,
)


class BaselineState(Enum):
    LOCATE = auto()
    BASE_ALIGN = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP_ALIGN = auto()
    CLOSE = auto()
    LIFT = auto()
    PREWRAP = auto()
    WRAP = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    SETTLE = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class OPropRouteMetrics:
    signed_winding: float
    nearest_distance: float
    nearby_segments: int
    peak_cable_speed: float


@dataclass(frozen=True)
class OPropPlan:
    prewrap_targets: tuple[np.ndarray, ...]
    route_targets: tuple[np.ndarray, ...]
    route_arc: float
    boundary_radius: float
    tangent_lead: float
    required_length: float


@dataclass(frozen=True)
class BaselineResult:
    success: bool
    state: BaselineState
    reason: str
    metrics: OPropRouteMetrics | None


def signed_local_winding(
    points_xy: np.ndarray, center_xy: np.ndarray, radius: float
) -> tuple[float, int]:
    """Signed cable angle accumulated close to a prop.

    Cable points must be ordered from the fixed end to the free end.  Only
    edges with at least one endpoint inside ``radius`` contribute, preventing
    distant S-curves elsewhere on the board from affecting the result.
    Positive is counter-clockwise when viewed from above.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return 0.0, 0
    rel = points - center
    distance = np.linalg.norm(rel, axis=1)
    edge_near = np.minimum(distance[:-1], distance[1:]) <= float(radius)
    if not np.any(edge_near):
        return 0.0, 0
    a = rel[:-1]
    b = rel[1:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = np.sum(a * b, axis=1)
    delta = np.arctan2(cross, dot)
    return float(np.sum(delta[edge_near])), int(
        np.count_nonzero(distance <= float(radius))
    )


def measure_o_prop_route(
    session, prop_body: int, verify_radius: float
) -> OPropRouteMetrics:
    points = session.data.xpos[session.cable_bodies, :2]
    center = session.data.xpos[prop_body, :2]
    winding, nearby = signed_local_winding(points, center, verify_radius)
    segment_start = points[:-1]
    segment_delta = points[1:] - segment_start
    segment_length_sq = np.sum(segment_delta * segment_delta, axis=1)
    projection = np.sum((center - segment_start) * segment_delta, axis=1)
    projection /= np.maximum(segment_length_sq, 1e-12)
    projection = np.clip(projection, 0.0, 1.0)
    closest = segment_start + projection[:, None] * segment_delta
    nearest = float(np.min(np.linalg.norm(closest - center, axis=1)))
    cable_velocity = session.data.cvel[np.asarray(session.cable_bodies), 3:6]
    peak_speed = float(np.sqrt(np.sum(cable_velocity * cable_velocity, axis=1).max()))
    return OPropRouteMetrics(winding, nearest, nearby, peak_speed)


def plan_o_prop_route(
    prop: np.ndarray,
    lifted_slot: np.ndarray,
    direction_sign: float,
) -> OPropPlan:
    """Plan a taut tangent lead whose trailing cable follows the O boundary.

    Targets describe the held cable point, not the centre of the gripper.  At
    every route sample it sits a fixed distance ahead on the circle tangent.
    The shortest taut path from upstream material therefore consists of the
    O-boundary arc followed by that tangent, rather than a large clearance arc.
    """
    prop = np.asarray(prop, dtype=np.float64)
    slot = np.asarray(lifted_slot, dtype=np.float64)
    start_angle = math.atan2(float(slot[1] - prop[1]), float(slot[0] - prop[0]))
    boundary_radius = float(
        config.BASELINE_O_OUTER_RADIUS
        + config.BASELINE_CABLE_RADIUS
        + config.BASELINE_O_CONTACT_GAP
    )
    tangent_lead = float(config.BASELINE_O_TANGENT_LEAD)
    route_z = float(prop[2] + config.BASELINE_O_ROUTE_Z)

    def tangent_target(angle: float) -> np.ndarray:
        radial = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        tangent = float(direction_sign) * np.array(
            [-math.sin(angle), math.cos(angle)], dtype=np.float64
        )
        xy = prop[:2] + boundary_radius * radial + tangent_lead * tangent
        return np.array([xy[0], xy[1], route_z], dtype=np.float64)

    low = np.array(
        tangent_target(start_angle), dtype=np.float64
    )
    transit_z = max(
        float(slot[2]),
        float(prop[2] + config.BASELINE_TRANSIT_CLEARANCE),
    )
    raised = slot.copy()
    raised[2] = transit_z
    high = low.copy()
    high[2] = transit_z
    arc = config.BASELINE_ROUTE_ARC
    angles = start_angle + float(direction_sign) * np.linspace(
        arc / config.BASELINE_ROUTE_WAYPOINTS,
        arc,
        config.BASELINE_ROUTE_WAYPOINTS,
    )
    route = tuple(tangent_target(float(angle)) for angle in angles)
    transfer_distance = float(np.linalg.norm(low[:2] - slot[:2]))
    prewrap = (
        (low,)
        if transfer_distance <= config.BASELINE_DIRECT_TRANSIT_DISTANCE
        else (raised, high, low)
    )
    required_length = boundary_radius * arc + tangent_lead
    return OPropPlan(
        prewrap, route, arc, boundary_radius, tangent_lead, required_length
    )


def plan_pregrasp_stance(
    slot: np.ndarray,
    grasp: np.ndarray,
    prop: np.ndarray,
) -> np.ndarray:
    """Bias the empty arm away from a distant target before base alignment.

    The subsequent base motion brings the biased slot back over ``grasp``.
    This changes the arm posture at capture without changing the cable point,
    leaving joint workspace in the direction of the prop.
    """
    slot = np.asarray(slot, dtype=np.float64)
    grasp = np.asarray(grasp, dtype=np.float64)
    prop = np.asarray(prop, dtype=np.float64)
    travel = prop[:2] - grasp[:2]
    distance = float(np.linalg.norm(travel))
    target = slot.copy()
    if distance <= config.BASELINE_STANCE_TRIGGER_DISTANCE:
        return target
    shift = min(
        config.BASELINE_STANCE_MAX_SHIFT,
        distance - config.BASELINE_STANCE_TRIGGER_DISTANCE,
    )
    target[:2] -= shift * travel / distance
    return target


class OPropBaseline:
    """Finite-state controller for grasp -> route -> verify -> safe stop."""

    ARM_NAME = config.BASELINE_ARM

    def __init__(self, session, args) -> None:
        self.session = session
        self.args = args
        self.model = session.model
        self.data = session.data
        self.arm: Arm = session.arms[self.ARM_NAME]
        self.direction_sign = 1.0 if args.baseline_direction == "ccw" else -1.0
        self.state = BaselineState.LOCATE
        self._run_started = float(self.data.time)
        self._state_started = float(self.data.time)
        self._prop_body: int | None = None
        self._grasp_body: int | None = None
        self._grasp_geom: int | None = None
        self._grasp_along = 0.0
        self._free_tail_segments = 0
        self._uses_pregrasp_stance = False
        self._assist_capture_started: float | None = None
        self._assist_capture_ctrl = 0.0
        self._bounded_assist_capture = False
        self._bounded_lift_start_z: float | None = None
        self._target: np.ndarray | None = None
        self._prewrap_targets: list[np.ndarray] = []
        self._prewrap_index = 0
        self._route_targets: list[np.ndarray] = []
        self._route_index = 0
        self._initial_metrics: OPropRouteMetrics | None = None
        self._grasp_completed = False
        self._result: BaselineResult | None = None
        self._last_progress_log = -1.0
        self._last_close_log = -1.0
        self._last_grasp_target_log = -1.0
        log(
            f"[baseline] state={self.state.name} prop={args.baseline_prop} "
            f"direction={args.baseline_direction} arm={self.arm.name}"
        )

    @property
    def done(self) -> bool:
        return self.state in (BaselineState.SUCCEEDED, BaselineState.FAILED)

    @property
    def result(self) -> BaselineResult | None:
        return self._result

    def _elapsed(self) -> float:
        return float(self.data.time) - self._state_started

    def _transition(self, state: BaselineState) -> None:
        self.state = state
        self._state_started = float(self.data.time)
        log(f"[baseline] state={state.name} t={self.data.time:.3f}s")
        if state == BaselineState.CLOSE:
            self.arm.close_ramp = True

    def _stop_base(self, dt: float) -> None:
        self.session.base_driver.drive(0.0, 0.0, 0.0, 0.0, dt)

    def _hold_other_arm(self) -> None:
        for name, arm in self.session.arms.items():
            if name != self.arm.name:
                hard_hold_arm(self.model, self.data, arm)

    def _hold_all_arms(self) -> None:
        for arm in self.session.arms.values():
            hard_hold_arm(self.model, self.data, arm)

    def _capture_motion_end(self) -> None:
        if self.arm.grasped_body is None:
            seed_arm(self.model, self.data, self.arm)
            return
        self.arm.q_ref[:] = [
            float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
            for joint_id in self.arm.joint_ids
        ]
        write_arm_ctrl(self.model, self.data, self.arm)

    def _cable_point(self, geom_id: int, reference: np.ndarray) -> np.ndarray:
        """Closest point on a cable capsule's centerline to ``reference``."""
        center = self.data.geom_xpos[geom_id]
        axis = self.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        half_length = float(self.model.geom_size[geom_id, 1])
        along = float(
            np.clip(np.dot(reference - center, axis), -half_length, half_length)
        )
        return center + along * axis

    def _tracked_grasp_point(self) -> np.ndarray:
        assert self._grasp_geom is not None
        center = self.data.geom_xpos[self._grasp_geom]
        axis = self.data.geom_xmat[self._grasp_geom].reshape(3, 3)[:, 2]
        return center + self._grasp_along * axis

    def _initialize_pregrasp_stance(
        self,
        target: np.ndarray,
        grasp: np.ndarray,
    ) -> bool:
        """Set the distant-prop arm posture before the first physics step."""
        initial = pad_slot_center(
            self.data, self.arm.pad_left, self.arm.pad_right
        )
        dt = float(self.model.opt.timestep)
        max_steps = int(
            math.ceil(
                2.0
                * config.BASELINE_STANCE_MAX_SHIFT
                / (config.BASELINE_PREGRASP_MAX_TCP_SPEED * dt)
            )
        )
        for _ in range(max_steps):
            current = pad_slot_center(
                self.data, self.arm.pad_left, self.arm.pad_right
            )
            error = np.asarray(target, dtype=np.float64) - current
            distance = float(np.linalg.norm(error))
            if distance <= config.BASELINE_POSITION_TOL:
                break
            speed = min(
                config.BASELINE_PREGRASP_MAX_TCP_SPEED,
                config.BASELINE_SERVO_KP * distance,
            )
            twist = np.zeros(6, dtype=np.float64)
            twist[:3] = error * (speed / max(distance, 1e-9))
            apply_twist_ik_kinematic(
                self.model,
                self.data,
                self.arm,
                twist,
                dt,
            )
            mujoco.mj_kinematics(self.model, self.data)
        else:
            self.fail("pre-grasp stance IK did not converge")
            return False

        mujoco.mj_forward(self.model, self.data)
        if self._pregrasp_uses_base_teleport():
            if not teleport_base_slot_translation_to_target(
                self.model,
                self.data,
                self.arm,
                grasp,
            ):
                self.fail("pre-grasp base alignment initialization failed")
                return False
        seed_arm(self.model, self.data, self.arm)
        log(
            f"[baseline] initialized distant-prop stance "
            f"shift={np.linalg.norm(np.asarray(target)[:2] - initial[:2]):.4f}m"
        )
        return True

    def _pregrasp_uses_base_teleport(self) -> bool:
        return True

    def _command_slot(
        self,
        target: np.ndarray,
        dt: float,
        tolerance: float = config.BASELINE_POSITION_TOL,
    ) -> bool:
        target = np.asarray(target, dtype=np.float64)
        current = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        error = target - current
        distance = float(np.linalg.norm(error))
        waypoint_control = self.state in (
            BaselineState.PREWRAP,
            BaselineState.WRAP,
        )
        # Arrival is stateful: entering the radius (or passing a close local
        # minimum below) irreversibly starts a dwell.  Contact-driven
        # millimetre oscillations therefore cannot clear it; only changing the
        # waypoint resets the latch.
        previous_target = getattr(self, "_slot_arrival_target", None)
        if waypoint_control and (
            previous_target is None
            or not np.allclose(previous_target, target, atol=1e-9, rtol=0.0)
        ):
            self._slot_arrival_target = target.copy()
            self._slot_arrival_started = None
            self._slot_arrival_latched = False
            self._slot_arrival_min_distance = math.inf
            self._slot_arrival_last_improved = float(self.data.time)
        # Cable/prop contact leaves a small compliant tracking offset.  Use a
        # 18 mm entry band for route waypoints (the nominal tolerance is
        # 12 mm), then require the dwell below before advancing.
        enter_radius = (
            max(float(tolerance), 0.018)
            if waypoint_control
            else float(tolerance)
        )
        stable_time = 0.15
        if waypoint_control and not self._slot_arrival_latched:
            previous_min = getattr(
                self, "_slot_arrival_min_distance", math.inf
            )
            if distance < previous_min - 0.0005:
                self._slot_arrival_last_improved = float(self.data.time)
            self._slot_arrival_min_distance = min(
                previous_min, distance
            )
        if (
            waypoint_control
            and distance <= enter_radius
            and not self._slot_arrival_latched
        ):
            self._slot_arrival_latched = True
            self._slot_arrival_started = float(self.data.time)
        elif (
            waypoint_control
            and not self._slot_arrival_latched
            and self._slot_arrival_min_distance <= 0.040
            and float(self.data.time)
            - getattr(self, "_slot_arrival_last_improved", float(self.data.time))
            >= 1.0
        ):
            self._slot_arrival_latched = True
            self._slot_arrival_started = float(self.data.time)
        elif (
            waypoint_control
            and not self._slot_arrival_latched
            and self._slot_arrival_min_distance <= 0.040
            and distance >= self._slot_arrival_min_distance + 0.003
        ):
            # Under cable/prop contact the commanded slot can reach its closest
            # feasible point and then be pushed away.  Detect that local
            # minimum instead of continually driving back through it.
            self._slot_arrival_latched = True
            self._slot_arrival_started = float(self.data.time)
        if float(self.data.time) - self._last_progress_log >= 1.0:
            self._last_progress_log = float(self.data.time)
            log(
                f"[baseline] progress state={self.state.name} error={distance:.4f}m "
                f"arrival={getattr(self, '_slot_arrival_latched', False)} "
                f"slot={current.round(4).tolist()} target={target.round(4).tolist()}"
            )
        if waypoint_control and self._slot_arrival_latched:
            self._capture_motion_end()
            if (
                self._slot_arrival_started is not None
                and float(self.data.time) - self._slot_arrival_started
                >= stable_time
            ):
                return True
            self._hold_other_arm()
            self._stop_base(dt)
            return False
        if not waypoint_control and distance <= tolerance:
            self._capture_motion_end()
            return True
        max_speed = config.BASELINE_MAX_TCP_SPEED
        if (
            self.state == BaselineState.LIFT
            and self._uses_pregrasp_stance
            and self._bounded_assist_capture
            and self.arm.grasped_body is not None
            and self._bounded_lift_start_z is not None
            and current[2]
            < self._bounded_lift_start_z
            + config.BASELINE_STANCE_LIFTOFF_HEIGHT
        ):
            max_speed = min(
                max_speed,
                config.BASELINE_STANCE_LIFTOFF_MAX_TCP_SPEED,
            )
        # Reduce both the cap and proportional gain near a waypoint.  The old
        # controller could still command ~48 mm/s at the 12 mm arrival radius,
        # enough to cross the target before the next contact-resolved step.
        if waypoint_control and distance < 0.04:
            max_speed = min(max_speed, max(0.006, 0.5 * distance))
            servo_kp = min(config.BASELINE_SERVO_KP, 1.0)
        else:
            servo_kp = config.BASELINE_SERVO_KP
        speed = min(max_speed, servo_kp * distance)
        twist = np.zeros(6, dtype=np.float64)
        twist[:3] = error * (speed / max(distance, 1e-9))
        twist = clamp_tcp_twist_for_contact(
            self.model, twist, self.arm.grasped_body is not None
        )
        self._apply_twist(twist, dt)
        self._hold_other_arm()
        self._stop_base(dt)
        return False

    def _apply_twist(self, twist: np.ndarray, dt: float) -> None:
        """Low-level arm servo hook shared by autonomous primitives."""
        if self._bounded_assist_capture and self.state == BaselineState.LIFT:
            apply_twist_ik_kinematic(
                self.model, self.data, self.arm, twist, dt
            )
        else:
            apply_twist_ik(self.model, self.data, self.arm, twist)

    def _grasp_target(self) -> np.ndarray:
        """Slot target that keeps the moving pad tips clear of the board."""
        assert self._grasp_geom is not None
        slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        lowest_pad_z = math.inf
        pad_geoms = self.arm.pad_left_contact | self.arm.pad_right_contact
        for geom_id in pad_geoms:
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                z_extent = float(
                    np.dot(np.abs(rotation[2]), self.model.geom_size[geom_id])
                )
                geom_lowest = float(self.data.geom_xpos[geom_id, 2] - z_extent)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(self.model.mesh_vertadr[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                stop = start + int(self.model.mesh_vertnum[mesh_id])
                world_vertices = (
                    self.model.mesh_vert[start:stop] @ rotation.T
                    + self.data.geom_xpos[geom_id]
                )
                geom_lowest = float(np.min(world_vertices[:, 2]))
            else:
                continue
            lowest_pad_z = min(lowest_pad_z, geom_lowest)
        cable = self._tracked_grasp_point()
        desired_lowest_z = float(
            cable[2]
            - config.BASELINE_CABLE_RADIUS
            + config.BASELINE_PAD_SURFACE_CLEARANCE
            - config.BASELINE_GRASP_CLOSE_OFFSET
        )
        target = cable.copy()
        target[2] = float(slot[2] + desired_lowest_z - lowest_pad_z)
        # This target is recomputed every physics step while descending and
        # closing.  Keep enough feedback for diagnosis without flooding
        # container logs (and slowing a headless/cloud run) at 500 Hz.
        if float(self.data.time) - self._last_grasp_target_log >= 0.5:
            self._last_grasp_target_log = float(self.data.time)
            log(
                f"[baseline-debug] grasp_target slot={slot.round(4).tolist()} "
                f"cable={cable.round(4).tolist()} target={target.round(4).tolist()} "
                f"lowest_pad_z={lowest_pad_z:.4f} desired_lowest_z={desired_lowest_z:.4f}"
            )
        return target

    def _locate(self) -> None:
        prop = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.args.baseline_prop
        )
        if prop < 0:
            self.fail(f"target prop '{self.args.baseline_prop}' is missing")
            return
        self._prop_body = int(prop)
        slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        dynamic_bodies = set(self.session.cable_bodies)
        cable_geoms = sorted(
            geom_id
            for geom_id in self.session.cable_geoms
            if int(self.model.geom_bodyid[geom_id]) in dynamic_bodies
        )
        all_cable_geoms = cable_geoms
        cable_geoms = self._candidate_cable_geoms(cable_geoms)
        self._free_tail_segments = len(all_cable_geoms) - len(cable_geoms)
        self._grasp_geom, grasp = self._select_grasp(cable_geoms, slot)
        self._grasp_body = int(self.model.geom_bodyid[self._grasp_geom])
        axis = self.data.geom_xmat[self._grasp_geom].reshape(3, 3)[:, 2]
        self._grasp_along = float(
            np.dot(grasp - self.data.geom_xpos[self._grasp_geom], axis)
        )
        self._initial_metrics = measure_o_prop_route(
            self.session,
            self._prop_body,
            config.BASELINE_VERIFY_RADIUS,
        )
        grasp = grasp.copy()
        self._target = grasp.copy()
        self._target[2] += config.BASELINE_GRASP_APPROACH_HEIGHT
        stance = self._plan_pregrasp_stance(
            slot, grasp, self.data.xpos[self._prop_body]
        )
        log(
            f"[baseline] located prop={self.data.xpos[self._prop_body].round(4).tolist()} "
            f"cable_body={self._grasp_body} cable_geom={self._grasp_geom} "
            f"cable={grasp.round(4).tolist()} "
            f"free_tail_segments={self._free_tail_segments} "
            f"initial_winding={math.degrees(self._initial_metrics.signed_winding):+.1f}deg"
        )
        if np.linalg.norm(stance[:2] - slot[:2]) > 1e-9:
            self._uses_pregrasp_stance = True
            if not self._initialize_pregrasp_stance(stance, grasp):
                return
        self._transition(BaselineState.BASE_ALIGN)

    def _candidate_cable_geoms(self, cable_geoms: list[int]) -> list[int]:
        safe_geoms = board_safe_cable_geoms(
            self.model,
            self.data,
            config.BASELINE_BACK_SUPPORT_MARGIN,
        )
        return safe_geoms or cable_geoms

    def _plan_pregrasp_stance(
        self,
        slot: np.ndarray,
        grasp: np.ndarray,
        prop: np.ndarray,
    ) -> np.ndarray:
        return plan_pregrasp_stance(slot, grasp, prop)

    def _select_grasp(
        self, cable_geoms: list[int], reference: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """Return a dynamic cable geom and material point for this primitive."""
        # Geoms are ordered fixed-to-free.  Grasp the last board-safe segment
        # explicitly, minimizing uncontrolled material beyond the gripper.
        geom_id = int(cable_geoms[-1])
        return geom_id, self._cable_point(geom_id, reference)

    def _bounded_capture_limits(self) -> tuple[float, float, float] | None:
        """Return ctrl, distance and settle limits for assisted capture."""
        if not self._uses_pregrasp_stance:
            return None
        return (
            config.BASELINE_STANCE_CAPTURE_CTRL,
            config.BASELINE_STANCE_CAPTURE_DISTANCE,
            config.BASELINE_STANCE_CAPTURE_SETTLE_TIME,
        )

    def _bounded_capture_label(self) -> str:
        return "baseline"

    def _bounded_capture_backoff(self) -> float:
        return (
            config.BASELINE_STANCE_CAPTURE_BACKOFF
            if self._uses_pregrasp_stance
            else config.BASELINE_GRIPPER_BACKOFF
        )

    def _hold_capture_support(self) -> None:
        pass

    def _capture_requires_two_pad_contact(self) -> bool:
        return True

    def _capture_with_existing_assist(self) -> None:
        """Capture a cable already inside the closed fingertip envelope."""
        limits = self._bounded_capture_limits()
        if (
            limits is None
            or self.arm.grasped_body is not None
            or self._grasp_body is None
        ):
            return
        capture_ctrl, capture_distance, _ = limits
        if float(self.data.ctrl[self.arm.gripper_act]) < capture_ctrl:
            return
        both, _, _ = pad_cable_contacts(
            self.model,
            self.data,
            self.arm.pad_left_contact,
            self.arm.pad_right_contact,
            self.session.cable_geoms,
        )
        if self._capture_requires_two_pad_contact() and not both:
            return
        slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        distance = float(np.linalg.norm(self._tracked_grasp_point() - slot))
        if distance > capture_distance:
            return
        picked = nearest_bodies_to_point(
            self.data, self.session.cable_bodies, slot, count=3
        )
        if self._grasp_body not in picked:
            return
        self.arm.grasped_body = picked[0]
        self.arm.grasped_neighbors = []
        capture_offset = (
            self.data.xpos[self.arm.grasped_body] - slot
        ).copy()
        assert self.arm.grasp_offset is not None
        self.arm.grasp_offset[:] = capture_offset
        self.arm.grasp_assist_age = 0.0
        self.arm.grasp_nocontact_time = 0.0
        self.arm.prev_slot_pos = (slot + capture_offset).copy()
        self.arm.close_ramp = False
        self._assist_capture_started = float(self.data.time)
        capture_ctrl = float(self.data.ctrl[self.arm.gripper_act])
        gripper_q = float(
            self.data.qpos[
                self.model.jnt_qposadr[self.arm.gripper_joint]
            ]
        )
        backoff = self._bounded_capture_backoff()
        self._assist_capture_ctrl = (
            max(config.GRIPPER_OPEN, gripper_q - backoff)
            if backoff > 0.0
            else capture_ctrl
        )
        self.data.ctrl[self.arm.gripper_act] = self._assist_capture_ctrl
        self._bounded_assist_capture = True
        log(
            f"[{self._bounded_capture_label()}] bounded grasp capture "
            f"body={self.arm.grasped_body} material_distance={distance:.4f}m "
            f"body_distance={np.linalg.norm(capture_offset):.4f}m "
            f"capture_ctrl={capture_ctrl:.3f} "
            f"hold_ctrl={self._assist_capture_ctrl:.3f}"
        )

    def _update_assist_capture(self, dt: float) -> bool:
        if self._assist_capture_started is None:
            return False
        if self.arm.grasped_body is None:
            self.fail("bounded grasp capture was lost")
            return True
        limits = self._bounded_capture_limits()
        assert limits is not None
        _, _, settle_time = limits
        self.arm.grasp_nocontact_time = 0.0
        self._hold_all_arms()
        self._stop_base(dt)
        self._hold_capture_support()
        self.data.ctrl[self.arm.gripper_act] = self._assist_capture_ctrl
        if float(self.data.time) - self._assist_capture_started >= settle_time:
            self._grasp_completed = True
            slot = pad_slot_center(
                self.data, self.arm.pad_left, self.arm.pad_right
            )
            self._bounded_lift_start_z = float(slot[2])
            self._target = slot + np.array(
                [0.0, 0.0, config.BASELINE_LIFT_HEIGHT]
            )
            self._transition(BaselineState.LIFT)
        return True

    def _align_base(
        self, dt: float, next_state: BaselineState, tolerance: float
    ) -> None:
        grasp = self._tracked_grasp_point()
        self._drive_base_to_xy(grasp[:2], dt, next_state, tolerance)

    def _drive_base_to_xy(
        self,
        target_xy: np.ndarray,
        dt: float,
        next_state: BaselineState,
        tolerance: float,
        max_command: float = config.BASELINE_BASE_ALIGN_MAX_COMMAND,
    ) -> None:
        """Move the base until the active gripper slot reaches a world XY."""
        slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        error = np.asarray(target_xy, dtype=np.float64) - slot[:2]
        distance = float(np.linalg.norm(error))
        if float(self.data.time) - self._last_progress_log >= 1.0:
            self._last_progress_log = float(self.data.time)
            log(
                f"[baseline] progress state={self.state.name} error={distance:.4f}m "
                f"slot_xy={slot[:2].round(4).tolist()} "
                f"target_xy={np.asarray(target_xy).round(4).tolist()}"
            )
        self._hold_all_arms()
        if distance <= tolerance:
            self._stop_base(dt)
            self._transition(next_state)
            return
        local = self.session.base_driver.world_xy_to_local(error)
        command = config.BASELINE_BASE_ALIGN_KP * local
        norm = float(np.linalg.norm(command))
        if norm > max_command:
            command *= max_command / norm
        self.session.base_driver.drive(
            float(command[0]), float(command[1]), 0.0, 0.0, dt
        )

    def _plan_route(self) -> None:
        assert self._prop_body is not None
        assert self._initial_metrics is not None
        prop = self.data.xpos[self._prop_body].copy()
        slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
        grasp_offset = (
            self.arm.grasp_offset
            if self.arm.grasp_offset is not None
            else np.zeros(3, dtype=np.float64)
        )
        plan = plan_o_prop_route(
            prop,
            slot + grasp_offset,
            self.direction_sign,
        )
        assert self._grasp_body is not None
        grasp_index = self.session.cable_bodies.index(self._grasp_body)
        material = self.data.xpos[
            np.asarray(self.session.cable_bodies[: grasp_index + 1])
        ]
        available_length = float(
            np.sum(np.linalg.norm(np.diff(material, axis=0), axis=1))
        )
        if (
            available_length
            < plan.required_length
            + config.BASELINE_O_MIN_AVAILABLE_LENGTH_MARGIN
        ):
            self.fail(
                "insufficient cable length for O boundary route: "
                f"available={available_length:.3f}m "
                f"required={plan.required_length:.3f}m"
            )
            return
        self._prewrap_targets = [
            target - grasp_offset for target in plan.prewrap_targets
        ]
        self._prewrap_index = 0
        self._route_targets = [
            target - grasp_offset for target in plan.route_targets
        ]
        self._route_index = 0
        log(
            f"[baseline] plan prewrap={len(self._prewrap_targets)} "
            f"arc={math.degrees(plan.route_arc):.1f}deg "
            f"boundary={plan.boundary_radius:.4f}m "
            f"lead={plan.tangent_lead:.4f}m "
            f"available={available_length:.3f}m"
        )

    def _release_target(self, route_target: np.ndarray) -> np.ndarray:
        """Return the final held pose before opening the gripper."""
        target = route_target.copy()
        assert self._prop_body is not None
        grasp_offset_z = (
            float(self.arm.grasp_offset[2])
            if self.arm.grasp_offset is not None
            else 0.0
        )
        target[2] = float(
            self.data.xpos[self._prop_body, 2]
            + config.BASELINE_LOWER_CLEARANCE
            - grasp_offset_z
        )
        return target

    def _retain_grasp_after_route(self) -> bool:
        return self.args.input in ("baseline_oc", "baseline_ocy")

    def _check_safety(self) -> bool:
        if float(self.data.time) - self._run_started > float(
            self.args.baseline_timeout
        ):
            self.fail("global timeout")
            return False
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(
            np.isfinite(self.data.qvel)
        ):
            self.fail("non-finite simulation state")
            return False
        contact_force = self.session.gripper_contact_force(self.arm.name)
        if contact_force > min(
            config.BASELINE_MAX_OBSTACLE_FORCE,
            config.BASELINE_MAX_CABLE_GRASP_FORCE,
        ):
            contacts: list[tuple[float, str, str, float]] = []
            gripper_geoms = self.session.haptic_geoms[self.arm.name]
            obstacle_force = 0.0
            cable_force = 0.0
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if geom1 not in gripper_geoms and geom2 not in gripper_geoms:
                    continue
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                name1 = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1
                ) or str(geom1)
                name2 = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2
                ) or str(geom2)
                force = abs(float(wrench[0]))
                contacts.append((force, name1, name2, float(contact.dist)))
                other = geom2 if geom1 in gripper_geoms else geom1
                if other in self.session.cable_geoms:
                    if (
                        self._bounded_assist_capture
                        and self.arm.grasped_body is not None
                        and int(self.model.geom_bodyid[other])
                        == self.arm.grasped_body
                    ):
                        continue
                    cable_force += force
                else:
                    obstacle_force += force
            contacts.sort(reverse=True)
            if self._bounded_assist_capture and self.state == BaselineState.PREWRAP:
                cable_force = 0.0
            detail = ", ".join(
                f"{a}/{b}:{force:.1f}N@{distance * 1000.0:.1f}mm"
                for force, a, b, distance in contacts[:3]
            )
            if obstacle_force > config.BASELINE_MAX_OBSTACLE_FORCE:
                self.fail(
                    f"gripper-obstacle force {obstacle_force:.1f}N exceeds safety limit"
                    + (f" ({detail})" if detail else "")
                )
                return False
            if cable_force > config.BASELINE_MAX_CABLE_GRASP_FORCE:
                self.fail(
                    f"cable grasp force {cable_force:.1f}N exceeds safety limit"
                    + (f" ({detail})" if detail else "")
                )
                return False
        if (
            self.state
            in (
                BaselineState.LIFT,
                BaselineState.PREWRAP,
                BaselineState.WRAP,
                BaselineState.LOWER,
            )
            and self.arm.grasped_body is None
        ):
            self.fail("cable grasp was lost")
            return False
        timeout = (
            config.BASELINE_GRASP_TIMEOUT
            if self.state == BaselineState.CLOSE
            else (
                max(config.BASELINE_STATE_TIMEOUT, config.BASELINE_PREWRAP_TIMEOUT)
                if self.state == BaselineState.PREWRAP
                else config.BASELINE_STATE_TIMEOUT
            )
        )
        if self.state not in (
            BaselineState.LOCATE,
            BaselineState.SUCCEEDED,
            BaselineState.FAILED,
        ):
            if self._elapsed() > timeout:
                self.fail(f"state {self.state.name} timed out")
                return False
        return True

    def update(self, dt: float) -> None:
        if self._bounded_assist_capture and self.arm.grasped_body is not None:
            self.arm.grasp_nocontact_time = 0.0
        if self.done or not self._check_safety():
            return
        if self.state == BaselineState.LOCATE:
            self._locate()
            return

        assert self._grasp_body is not None and self._grasp_geom is not None
        if self.state == BaselineState.CLOSE and self._update_assist_capture(dt):
            return

        if self.state == BaselineState.BASE_ALIGN:
            self._align_base(dt, BaselineState.APPROACH, config.BASELINE_BASE_ALIGN_TOL)
            return

        if self.state == BaselineState.APPROACH:
            grasp = self._tracked_grasp_point()
            grasp[2] += config.BASELINE_GRASP_APPROACH_HEIGHT
            if self._command_slot(grasp, dt):
                self._transition(BaselineState.DESCEND)
            return

        if self.state == BaselineState.DESCEND:
            if self._command_slot(
                self._grasp_target(), dt, config.BASELINE_GRASP_POSITION_TOL
            ):
                self._transition(BaselineState.GRASP_ALIGN)
            return

        if self.state == BaselineState.GRASP_ALIGN:
            self._align_base(
                dt, BaselineState.CLOSE, config.BASELINE_GRASP_BASE_ALIGN_TOL
            )
            return

        if self.state == BaselineState.CLOSE:
            align_tolerance = (
                config.BASELINE_STANCE_GRASP_ALIGN_TOL
                if self._uses_pregrasp_stance
                else config.BASELINE_GRASP_BASE_ALIGN_TOL
            )
            if self._command_slot(
                self._grasp_target(), dt, align_tolerance
            ):
                hard_hold_arm(self.model, self.data, self.arm)
            self._hold_other_arm()
            self._stop_base(dt)
            if float(self.data.time) - self._last_close_log >= 1.0:
                self._last_close_log = float(self.data.time)
                both, force, count = pad_cable_contacts(
                    self.model,
                    self.data,
                    self.arm.pad_left_contact,
                    self.arm.pad_right_contact,
                    self.session.cable_geoms,
                )
                gripper_q = float(
                    self.data.qpos[self.model.jnt_qposadr[self.arm.gripper_joint]]
                )
                log(
                    f"[baseline] progress state=CLOSE ctrl={self.data.ctrl[self.arm.gripper_act]:.3f} "
                    f"gripper_q={gripper_q:.4f} pad_contacts={count} both={both} force={force:.2f}N "
                    f"pads={[self.data.geom_xpos[self.arm.pad_left].round(4).tolist(), self.data.geom_xpos[self.arm.pad_right].round(4).tolist()]} "
                    f"slot={pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right).round(4).tolist()} "
                    f"cable={self._tracked_grasp_point().round(4).tolist()}"
                )
            if self.arm.grasped_body is not None and not self.arm.close_ramp:
                self._grasp_completed = True
                gripper_q = float(
                    self.data.qpos[self.model.jnt_qposadr[self.arm.gripper_joint]]
                )
                self.data.ctrl[self.arm.gripper_act] = max(
                    config.GRIPPER_OPEN,
                    gripper_q - config.BASELINE_GRIPPER_BACKOFF,
                )
                slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
                self._target = slot + np.array([0.0, 0.0, config.BASELINE_LIFT_HEIGHT])
                self._transition(BaselineState.LIFT)
            else:
                self._capture_with_existing_assist()
            return

        if self.state == BaselineState.LIFT:
            assert self._target is not None
            if self._command_slot(self._target, dt):
                self._plan_route()
                if self.done:
                    return
                self._transition(BaselineState.PREWRAP)
            return

        if self.state == BaselineState.PREWRAP:
            target = self._prewrap_targets[self._prewrap_index]
            if self._command_slot(target, dt):
                self._prewrap_index += 1
                self._slot_arrival_started = None
                self._slot_arrival_latched = False
                self._state_started = float(self.data.time)
                log(
                    f"[baseline] prewrap waypoint={self._prewrap_index}/"
                    f"{len(self._prewrap_targets)}"
                )
                if self._prewrap_index == len(self._prewrap_targets):
                    self._transition(BaselineState.WRAP)
            return

        if self.state == BaselineState.WRAP:
            target = self._route_targets[self._route_index]
            if self._command_slot(target, dt):
                self._route_index += 1
                self._slot_arrival_started = None
                self._slot_arrival_latched = False
                self._state_started = float(self.data.time)
                log(
                    f"[baseline] wrap waypoint={self._route_index}/{len(self._route_targets)}"
                )
                if self._route_index == len(self._route_targets):
                    if self._retain_grasp_after_route():
                        self._post_wrap_slot = pad_slot_center(
                            self.data, self.arm.pad_left, self.arm.pad_right
                        ).copy()
                        self._post_wrap_gripper_ctrl = float(
                            self.data.ctrl[self.arm.gripper_act]
                        )
                        self._post_wrap_force = float(
                            self.session.gripper_contact_force(self.arm.name)
                        )
                        self._target = self._post_wrap_slot.copy()
                        self._lower_force_entry = self._post_wrap_force
                        self._lower_last_force = self._post_wrap_force
                        self._lower_last_force_time = float(self.data.time)
                    else:
                        self._target = self._release_target(target)
                    self._transition(BaselineState.LOWER)
            return

        if self.state == BaselineState.LOWER:
            assert self._target is not None
            if self._retain_grasp_after_route():
                force = float(self.session.gripper_contact_force(self.arm.name))
                now = float(self.data.time)
                force_dt = max(now - self._lower_last_force_time, 1e-6)
                force_rate = (force - self._lower_last_force) / force_dt
                hard_limit = config.BASELINE_MAX_CABLE_GRASP_FORCE
                soft_limit = min(
                    0.6 * hard_limit,
                    self._lower_force_entry + 25.0,
                )
                if force >= hard_limit:
                    self.fail(
                        f"post-wrap LOWER contact force {force:.1f}N exceeds "
                        f"safety limit {hard_limit:.1f}N"
                    )
                    return

                # First retained-grasp version deliberately performs no Z
                # descent.  It also issues no XY correction, so the compliant
                # arm cannot pull an already wrapped cable back toward the
                # final route waypoint.
                write_arm_ctrl(self.model, self.data, self.arm)
                self.data.ctrl[self.arm.gripper_act] = (
                    self._post_wrap_gripper_ctrl
                )
                self._hold_other_arm()
                self._stop_base(dt)
                log(
                    f"[baseline] post-wrap LOWER hold force={force:.1f}N "
                    f"delta={force - self._lower_force_entry:+.1f}N "
                    f"rate={force_rate:+.1f}N/s soft={soft_limit:.1f}N"
                )
                self._settle_gripper_ctrl = self._post_wrap_gripper_ctrl
                self._settle_last_force = force
                self._settle_last_force_time = now
                self._settle_stable_since = None
                self._transition(BaselineState.SETTLE)
                return
            if self._command_slot(self._target, dt):
                open_gripper(self.data, self.arm)
                self._transition(BaselineState.RELEASE)
            return

        if self.state == BaselineState.RELEASE:
            self._hold_all_arms()
            self._stop_base(dt)
            if self._elapsed() >= config.BASELINE_RELEASE_WAIT:
                slot = pad_slot_center(self.data, self.arm.pad_left, self.arm.pad_right)
                self._target = slot + np.array(
                    [0.0, 0.0, config.BASELINE_RETREAT_HEIGHT]
                )
                self._transition(BaselineState.RETREAT)
            return

        if self.state == BaselineState.RETREAT:
            assert self._target is not None
            if self._command_slot(self._target, dt):
                self._transition(BaselineState.SETTLE)
            return

        if self.state == BaselineState.SETTLE:
            if self._retain_grasp_after_route():
                if self.arm.grasped_body is None:
                    self.fail("cable grasp was lost before C-prop handoff")
                    return
                force = float(self.session.gripper_contact_force(self.arm.name))
                now = float(self.data.time)
                if not hasattr(self, "_settle_hold_q_ref"):
                    self._settle_hold_q_ref = np.array(
                        [
                            float(
                                self.data.qpos[
                                    self.model.jnt_qposadr[joint_id]
                                ]
                            )
                            for joint_id in self.arm.joint_ids
                        ],
                        dtype=np.float64,
                    )
                    self._settle_hold_tcp = pad_slot_center(
                        self.data, self.arm.pad_left, self.arm.pad_right
                    ).copy()
                    self._settle_hold_gripper_ctrl = float(
                        max(self.data.ctrl[self.arm.gripper_act], 0.35)
                    )
                    self._settle_hold_force = force
                    self._settle_filtered_force = force
                    self._settle_last_filtered_force = force
                    self._settle_last_force_time = now
                    self._settle_stable_since = None
                    log(
                        f"[baseline] SETTLE compliant hold "
                        f"force={force:.1f}N "
                        f"slot={self._settle_hold_tcp.round(4).tolist()}"
                    )
                force_dt = max(now - self._settle_last_force_time, 1e-6)
                force_alpha = min(1.0, force_dt / 0.05)
                self._settle_filtered_force += force_alpha * (
                    force - self._settle_filtered_force
                )
                force_rate = (
                    self._settle_filtered_force
                    - self._settle_last_filtered_force
                ) / force_dt
                hard_limit = config.BASELINE_MAX_CABLE_GRASP_FORCE
                soft_limit = min(
                    0.6 * hard_limit,
                    self._lower_force_entry + 25.0,
                )
                if force >= hard_limit:
                    self.fail(
                        f"post-wrap SETTLE contact force {force:.1f}N exceeds "
                        f"safety limit {hard_limit:.1f}N"
                    )
                    return

                current_q = np.array(
                    [
                        float(
                            self.data.qpos[
                                self.model.jnt_qposadr[joint_id]
                            ]
                        )
                        for joint_id in self.arm.joint_ids
                    ],
                    dtype=np.float64,
                )
                current_tcp = pad_slot_center(
                    self.data, self.arm.pad_left, self.arm.pad_right
                )
                tcp_drift = float(
                    np.linalg.norm(current_tcp - self._settle_hold_tcp)
                )
                if tcp_drift > 0.020:
                    self.fail(
                        f"post-wrap SETTLE TCP drift {tcp_drift:.3f}m "
                        "exceeds compliant hold limit"
                    )
                    return

                # These are velocity actuators, so a bounded joint position
                # error produces a low-gain corrective velocity.  More drift
                # increases the gain; rising contact force reduces it so the
                # arm can yield by a few millimetres without becoming free.
                hold_gain = 1.2
                hold_limit = 0.12
                if tcp_drift > 0.008:
                    hold_gain = 2.0
                    hold_limit = 0.18
                if force >= soft_limit or force_rate > 30.0:
                    hold_gain = min(hold_gain, 0.5)
                    hold_limit = min(hold_limit, 0.04)
                joint_ctrl = np.clip(
                    hold_gain * (self._settle_hold_q_ref - current_q),
                    -hold_limit,
                    hold_limit,
                )
                for control, actuator_id in zip(
                    joint_ctrl, self.arm.act_ids
                ):
                    self.data.ctrl[actuator_id] = float(control)

                # Keep the closing command captured on SETTLE entry.  Never
                # raise it in response to drift or loss of pad contact.
                self.data.ctrl[self.arm.gripper_act] = (
                    self._settle_hold_gripper_ctrl
                )
                self.arm.grasp_nocontact_time = 0.0
                for other in self.session.arms.values():
                    if other.name == self.arm.name:
                        continue
                    for index, actuator_id in enumerate(other.act_ids):
                        joint_id = other.joint_ids[index]
                        q_error = float(other.q_ref[index]) - float(
                            self.data.qpos[self.model.jnt_qposadr[joint_id]]
                        )
                        self.data.ctrl[actuator_id] = float(
                            np.clip(1.2 * q_error, -0.12, 0.12)
                        )
                self._stop_base(dt)

                stable_envelope = (
                    force <= soft_limit and tcp_drift <= 0.010
                )
                stable = stable_envelope and abs(force_rate) <= 10.0
                if now - self._last_progress_log >= 1.0:
                    self._last_progress_log = now
                    log(
                        f"[baseline] SETTLE force={force:.1f}N "
                        f"rate={force_rate:+.1f}N/s "
                        f"drift={tcp_drift * 1000.0:.1f}mm "
                        f"stable={stable}"
                    )
                if not stable_envelope:
                    self._settle_stable_since = None
                elif stable and self._settle_stable_since is None:
                    # A single filtered near-zero rate starts the dwell.  Once
                    # started, solver-scale force-rate spikes do not clear it
                    # unless force or TCP drift actually leaves the safe
                    # compliant envelope.
                    self._settle_stable_since = now
                self._settle_last_force = force
                self._settle_last_filtered_force = (
                    self._settle_filtered_force
                )
                self._settle_last_force_time = now
                if (
                    self._settle_stable_since is not None
                    and now - self._settle_stable_since >= 1.0
                ):
                    self._verify()
                return
            self._hold_all_arms()
            self._stop_base(dt)
            if self._elapsed() >= config.BASELINE_SETTLE_TIME:
                self._verify()

    def _verify(self) -> None:
        assert self._prop_body is not None
        assert self._initial_metrics is not None
        metrics = measure_o_prop_route(
            self.session, self._prop_body, config.BASELINE_VERIFY_RADIUS
        )
        winding_gain = self.direction_sign * (
            metrics.signed_winding - self._initial_metrics.signed_winding
        )
        failures = []
        if not self._grasp_completed:
            failures.append("grasp was never confirmed")
        if winding_gain < config.BASELINE_MIN_WINDING_GAIN:
            failures.append(
                f"directed winding gain {math.degrees(winding_gain):.1f}deg < "
                f"{math.degrees(config.BASELINE_MIN_WINDING_GAIN):.1f}deg"
            )
        if metrics.nearby_segments < 2:
            failures.append("cable is not close enough to the prop")
        if metrics.nearest_distance > config.BASELINE_O_MAX_VERIFIED_DISTANCE:
            failures.append(
                f"nearest cable distance {metrics.nearest_distance:.3f}m exceeds "
                f"O boundary limit {config.BASELINE_O_MAX_VERIFIED_DISTANCE:.3f}m"
            )
        if (
            not self._retain_grasp_after_route()
            and metrics.peak_cable_speed > config.BASELINE_SETTLED_CABLE_SPEED
        ):
            failures.append(
                f"cable is still moving at {metrics.peak_cable_speed:.2f}m/s"
            )
        if failures:
            self.fail("; ".join(failures), metrics)
            return
        reason = (
            f"route verified: winding={math.degrees(metrics.signed_winding):+.1f}deg "
            f"gain={math.degrees(winding_gain):+.1f}deg nearest={metrics.nearest_distance:.3f}m"
        )
        self._transition(BaselineState.SUCCEEDED)
        self._result = BaselineResult(True, self.state, reason, metrics)
        log(f"[baseline] SUCCESS {reason}")

    def fail(self, reason: str, metrics: OPropRouteMetrics | None = None) -> None:
        if self.done:
            return
        finite = np.all(np.isfinite(self.data.qpos)) and np.all(
            np.isfinite(self.data.qvel)
        )
        self.arm.close_ramp = False
        self.data.xfrc_applied[:, :] = 0.0
        if finite:
            for arm in self.session.arms.values():
                seed_arm(self.model, self.data, arm)
                hard_hold_arm(self.model, self.data, arm)
            self._stop_base(max(float(self.model.opt.timestep), 1e-6))
        else:
            self.data.ctrl[:] = 0.0
        self._transition(BaselineState.FAILED)
        self._result = BaselineResult(False, self.state, reason, metrics)
        log(f"[baseline] SAFE_STOP {reason}")
