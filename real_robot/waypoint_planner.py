"""Pure NumPy semantic O/C/Y waypoint planning; values are geometry inputs."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: np.ndarray
    arm: str
    note: str = ""


def plan_o(center: np.ndarray, start: np.ndarray, obstacle_radius_m: float,
           cable_radius_m: float, clearance_m: float, tangent_lead_m: float,
           arc_rad: float = 1.75 * math.pi, samples: int = 12, direction: int = 1) -> list[Waypoint]:
    center, start = np.asarray(center, float), np.asarray(start, float)
    radius = obstacle_radius_m + cable_radius_m + clearance_m
    angle0 = math.atan2(start[1] - center[1], start[0] - center[0])
    z = max(center[2], start[2]) + clearance_m
    def target(angle: float) -> np.ndarray:
        radial = np.array([math.cos(angle), math.sin(angle)])
        tangent = direction * np.array([-radial[1], radial[0]])
        xy = center[:2] + radius * radial + tangent_lead_m * tangent
        return np.r_[xy, z]
    out = [Waypoint("O_PREWRAP", target(angle0), "right", "raised tangent entry")]
    for i, angle in enumerate(angle0 + direction * np.linspace(arc_rad / samples, arc_rad, samples), 1):
        out.append(Waypoint(f"O_WRAP_{i:02d}", target(float(angle)), "right", "tangent-lead boundary sample"))
    out.append(Waypoint("O_SETTLE", out[-1].position.copy(), "right", "verify from perception before advance"))
    return out


def plan_c(center: np.ndarray, mouth_direction: np.ndarray, clearance_m: float, seat_depth_m: float) -> list[Waypoint]:
    center, direction = np.asarray(center, float), np.asarray(mouth_direction, float)
    direction /= np.linalg.norm(direction)
    return [Waypoint("C_APPROACH", center + direction * (seat_depth_m + clearance_m), "right"),
            Waypoint("C_MOUTH", center + direction * seat_depth_m, "right"),
            Waypoint("C_SEAT", center, "right", "seat along perceived mouth axis"),
            Waypoint("C_EXIT", center - direction * clearance_m, "right", "continue through; do not pull back through mouth")]


def plan_y(center: np.ndarray, span_axis: np.ndarray, span_m: float, clearance_m: float) -> list[Waypoint]:
    center, axis = np.asarray(center, float), np.asarray(span_axis, float)
    axis /= np.linalg.norm(axis)
    left, right = center - 0.5 * span_m * axis, center + 0.5 * span_m * axis
    up = np.array([0.0, 0.0, clearance_m])
    return [Waypoint("Y_APPROACH", left + up, "left"), Waypoint("Y_APPROACH", right + up, "right"),
            Waypoint("Y_CENTER", left, "left", "synchronized bimanual target"),
            Waypoint("Y_CENTER", right, "right", "synchronized bimanual target"),
            Waypoint("Y_HOLD", center, "both", "verify midpoint, span and temporal stability")]
