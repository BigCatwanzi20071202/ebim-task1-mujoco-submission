"""Geometry and planning boundary for the stage-1 Y checkpoint.

The current public scene has no official F3/Y fixture asset.  The companion
XML therefore contains a clearly labelled proxy body.  These functions use
only ordered cable points and the fixture transform: camera perception can
later replace the privileged MuJoCo locator without changing the success
criterion or the bimanual hold targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class YPropPlan:
    entry: np.ndarray
    left_hold: np.ndarray
    right_hold: np.ndarray
    target_span: float


@dataclass(frozen=True)
class YPropMetrics:
    junction_distance: float
    has_left_branch: bool
    has_right_branch: bool
    branch_span: float
    peak_cable_speed: float
    passed: bool


@dataclass(frozen=True)
class YInspectionPlan:
    left_target: np.ndarray
    right_target: np.ndarray
    midpoint: np.ndarray
    relative: np.ndarray
    target_span: float


@dataclass(frozen=True)
class YInspectionMetrics:
    center_error: float
    span_error: float
    peak_cable_speed: float
    passed: bool


@dataclass(frozen=True)
class BimanualMaterialPair:
    left_index: int
    right_index: int
    left_point: np.ndarray
    right_point: np.ndarray
    score: float


@dataclass(frozen=True)
class YBaseStance:
    base_xy: np.ndarray
    waypoints: tuple[np.ndarray, ...]
    pair: BimanualMaterialPair
    score: float


def select_bimanual_material_pair(
    cable_points: np.ndarray,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    *,
    min_gap: int = config.Y_BASELINE_PAIR_MIN_GAP,
    max_gap: int = config.Y_BASELINE_PAIR_MAX_GAP,
    tail_count: int = config.Y_BASELINE_PAIR_TAIL_COUNT,
    min_point_distance: float = config.Y_BASELINE_PAIR_MIN_DISTANCE,
    max_reference_distance: float | None = None,
    right_alignment_point: np.ndarray | None = None,
) -> BimanualMaterialPair:
    """Select two ordered, distinct cable points for a bimanual handover.

    The search is restricted to the free-tail material and penalizes crossed
    arm assignments.  This is deterministic and independent of MuJoCo, making
    material selection testable before any robot motion is commanded.
    """
    points = np.asarray(cable_points, dtype=np.float64)
    left = np.asarray(left_reference, dtype=np.float64)
    right = np.asarray(right_reference, dtype=np.float64)
    alignment = (
        None
        if right_alignment_point is None
        else np.asarray(right_alignment_point, dtype=np.float64)
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("cable_points must have shape (N, 3)")
    if len(points) <= min_gap:
        raise ValueError("not enough cable material for bimanual selection")
    start = max(0, len(points) - int(tail_count))
    best: BimanualMaterialPair | None = None
    for i in range(start, len(points)):
        for j in range(i + int(min_gap), min(len(points), i + int(max_gap) + 1)):
            if float(np.linalg.norm(points[j] - points[i])) < min_point_distance:
                continue
            for left_index, right_index in ((i, j), (j, i)):
                left_point = points[left_index]
                right_point = points[right_index]
                translation = np.zeros(3)
                if alignment is not None:
                    translation[:2] = right_point[:2] - alignment[:2]
                shifted_left = left + translation
                shifted_right = right + translation
                left_distance = float(np.linalg.norm(left_point - shifted_left))
                right_distance = float(np.linalg.norm(right_point - shifted_right))
                if max_reference_distance is not None and (
                    left_distance > max_reference_distance
                    or right_distance > max_reference_distance
                ):
                    continue
                candidate = BimanualMaterialPair(
                    left_index,
                    right_index,
                    left_point.copy(),
                    right_point.copy(),
                    left_distance + right_distance,
                )
                if best is None or candidate.score < best.score:
                    best = candidate
    if best is None:
        raise ValueError("no cable pair satisfies the material-gap limits")
    return best


def pair_is_reachable(
    pair: BimanualMaterialPair,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    max_reach: float = config.Y_BASELINE_MAX_ARM_REACH,
) -> bool:
    """Conservative pre-motion reach gate for both selected material points."""
    left_distance = float(
        np.linalg.norm(np.asarray(pair.left_point) - np.asarray(left_shoulder))
    )
    right_distance = float(
        np.linalg.norm(np.asarray(pair.right_point) - np.asarray(right_shoulder))
    )
    return left_distance <= max_reach and right_distance <= max_reach


def plan_y_base_stance(
    cable_points: np.ndarray,
    base_xy: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_slot: np.ndarray | None = None,
    right_slot: np.ndarray | None = None,
    *,
    right_x: float = config.Y_BASELINE_BASE_RIGHT_X,
    top_y: float = config.Y_BASELINE_BASE_TOP_Y,
    right_y_range: tuple[float, float] = config.Y_BASELINE_BASE_RIGHT_Y_RANGE,
    top_x_range: tuple[float, float] = config.Y_BASELINE_BASE_TOP_X_RANGE,
    samples: int = config.Y_BASELINE_BASE_STANCE_SAMPLES,
    max_reach: float = config.Y_BASELINE_MAX_ARM_REACH,
    min_point_distance: float = config.Y_BASELINE_PAIR_MIN_DISTANCE,
) -> YBaseStance:
    """Jointly choose an outside-table base stance and reachable cable pair."""
    base = np.asarray(base_xy, dtype=np.float64)
    left = np.asarray(left_shoulder, dtype=np.float64)
    right = np.asarray(right_shoulder, dtype=np.float64)
    left_tcp = left if left_slot is None else np.asarray(left_slot, dtype=np.float64)
    right_tcp = (
        right if right_slot is None else np.asarray(right_slot, dtype=np.float64)
    )
    candidates: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    for y in np.linspace(*right_y_range, int(samples)):
        stance = np.array([right_x, y], dtype=np.float64)
        candidates.append((stance, (stance.copy(),)))
    corner = np.array([right_x, top_y], dtype=np.float64)
    for x in np.linspace(*top_x_range, int(samples)):
        stance = np.array([x, top_y], dtype=np.float64)
        candidates.append((stance, (corner.copy(), stance.copy())))

    best: YBaseStance | None = None
    for stance, waypoints in candidates:
        translation = stance - base
        shifted_left = left.copy()
        shifted_right = right.copy()
        shifted_left[:2] += translation
        shifted_right[:2] += translation
        shifted_left_tcp = left_tcp.copy()
        shifted_right_tcp = right_tcp.copy()
        shifted_left_tcp[:2] += translation
        shifted_right_tcp[:2] += translation
        try:
            pair = select_bimanual_material_pair(
                cable_points,
                shifted_left_tcp,
                shifted_right_tcp,
                min_point_distance=min_point_distance,
            )
        except ValueError:
            continue
        if not pair_is_reachable(pair, shifted_left, shifted_right, max_reach):
            continue
        travel = float(
            np.linalg.norm(waypoints[0] - base)
            + sum(
                np.linalg.norm(b - a)
                for a, b in zip(waypoints[:-1], waypoints[1:])
            )
        )
        score = pair.score + config.Y_BASELINE_BASE_TRAVEL_WEIGHT * travel
        candidate = YBaseStance(stance.copy(), waypoints, pair, score)
        if best is None or candidate.score < best.score:
            best = candidate
    if best is None:
        raise ValueError("no collision-free outside-table bimanual stance exists")
    return best


def plan_y_prop_hold(origin: np.ndarray, rotation: np.ndarray) -> YPropPlan:
    """Transform the proxy-independent local Y route into world coordinates."""
    origin = np.asarray(origin, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    def world(local: np.ndarray) -> np.ndarray:
        return origin + rotation @ local

    left = world(config.Y_BASELINE_LEFT_HOLD_LOCAL)
    right = world(config.Y_BASELINE_RIGHT_HOLD_LOCAL)
    return YPropPlan(
        entry=world(config.Y_BASELINE_ENTRY_LOCAL),
        left_hold=left,
        right_hold=right,
        target_span=float(np.linalg.norm(right - left)),
    )


def plan_y_inspection_transport(
    left_point: np.ndarray,
    right_point: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> YInspectionPlan:
    """Translate a bimanual cable pose to Y while preserving its orientation.

    Both targets receive the same midpoint translation.  Only a small,
    symmetric extension is added along the original inter-gripper axis, so
    relative pose is fixed and cable length is constant plus light tension.
    """
    left = np.asarray(left_point, dtype=np.float64)
    right = np.asarray(right_point, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    relative = right - left
    span = float(np.linalg.norm(relative))
    if span <= 1e-9:
        raise ValueError("inspection grasp points must be distinct")
    axis = relative / span
    target_span = span + config.Y_BASELINE_INSPECTION_TENSION
    midpoint = origin + rotation @ config.Y_BASELINE_INSPECTION_CENTER_LOCAL
    target_relative = target_span * axis
    return YInspectionPlan(
        left_target=midpoint - 0.5 * target_relative,
        right_target=midpoint + 0.5 * target_relative,
        midpoint=midpoint,
        relative=target_relative,
        target_span=target_span,
    )


def measure_y_inspection_hold(
    left_point: np.ndarray,
    right_point: np.ndarray,
    cable_velocities: np.ndarray,
    plan: YInspectionPlan,
) -> YInspectionMetrics:
    left = np.asarray(left_point, dtype=np.float64)
    right = np.asarray(right_point, dtype=np.float64)
    velocities = np.asarray(cable_velocities, dtype=np.float64)
    midpoint = 0.5 * (left + right)
    span = float(np.linalg.norm(right - left))
    center_error = float(np.linalg.norm(midpoint - plan.midpoint))
    span_error = abs(span - plan.target_span)
    peak_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    passed = (
        center_error <= config.Y_BASELINE_INSPECTION_CENTER_TOL
        and span_error <= config.Y_BASELINE_INSPECTION_SPAN_TOL
        and peak_speed <= config.Y_BASELINE_MAX_CABLE_SPEED
    )
    return YInspectionMetrics(center_error, span_error, peak_speed, passed)


def measure_y_prop_route(
    cable_points: np.ndarray,
    cable_velocities: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray,
) -> YPropMetrics:
    """Verify a continuous cable reaches the junction and spans both arms.

    Points must remain in cable material order.  The verifier intentionally
    requires one material point near the junction and material on both sides
    farther along the Y branches; merely passing close to the fixture cannot
    validate the checkpoint.
    """
    points = np.asarray(cable_points, dtype=np.float64)
    velocities = np.asarray(cable_velocities, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("cable_points must have shape (N, 3), N >= 3")
    if velocities.shape != points.shape:
        raise ValueError("cable_velocities must match cable_points")

    local = (points - origin) @ rotation
    junction = config.Y_BASELINE_JUNCTION_LOCAL
    junction_distance = float(np.min(np.linalg.norm(local - junction, axis=1)))
    z_ok = np.abs(local[:, 2] - junction[2]) <= config.Y_BASELINE_MAX_Z_ERROR
    forward = local[:, 1] >= config.Y_BASELINE_BRANCH_MIN_Y
    left_mask = (
        z_ok
        & forward
        & (local[:, 0] <= -config.Y_BASELINE_BRANCH_MIN_X)
    )
    right_mask = (
        z_ok
        & forward
        & (local[:, 0] >= config.Y_BASELINE_BRANCH_MIN_X)
    )
    has_left = bool(np.any(left_mask))
    has_right = bool(np.any(right_mask))
    branch_span = 0.0
    if has_left and has_right:
        left_x = float(np.min(local[left_mask, 0]))
        right_x = float(np.max(local[right_mask, 0]))
        branch_span = right_x - left_x
    peak_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    span_ok = (
        config.Y_BASELINE_MIN_SPAN
        <= branch_span
        <= config.Y_BASELINE_MAX_SPAN
    )
    passed = (
        junction_distance <= config.Y_BASELINE_CHECK_RADIUS
        and has_left
        and has_right
        and span_ok
        and peak_speed <= config.Y_BASELINE_MAX_CABLE_SPEED
    )
    return YPropMetrics(
        junction_distance,
        has_left,
        has_right,
        branch_span,
        peak_speed,
        passed,
    )
