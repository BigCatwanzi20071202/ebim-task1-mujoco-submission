"""Independent guards. Numeric defaults are placeholders, never FR3 certification."""

from dataclasses import dataclass
import numpy as np


REQUIRES_ONSITE_VALIDATION = "REQUIRES_ONSITE_VALIDATION"


@dataclass(frozen=True)
class SafetyLimits:
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_waypoint_delta_m: float
    max_translation_speed_mps: float
    max_rotation_speed_rps: float
    max_wrench_norm_n: float
    validation_status: str = REQUIRES_ONSITE_VALIDATION


def require_motion_enabled(enabled: bool) -> None:
    if not enabled:
        raise PermissionError("motion_enabled=False")


def validate_waypoint(current: np.ndarray, target: np.ndarray, limits: SafetyLimits) -> None:
    current, target = np.asarray(current, float), np.asarray(target, float)
    if not np.all((target >= limits.workspace_min) & (target <= limits.workspace_max)):
        raise ValueError("workspace guard rejected waypoint")
    if np.linalg.norm(target - current) > limits.max_waypoint_delta_m:
        raise ValueError("max waypoint delta guard rejected waypoint")


def validate_rates(translation_mps: float, rotation_rps: float, limits: SafetyLimits) -> None:
    if translation_mps > limits.max_translation_speed_mps:
        raise ValueError("translation speed guard triggered")
    if rotation_rps > limits.max_rotation_speed_rps:
        raise ValueError("rotation speed guard triggered")


def validate_wrench(wrench: np.ndarray, limits: SafetyLimits) -> None:
    if np.linalg.norm(np.asarray(wrench, float)) > limits.max_wrench_norm_n:
        raise ValueError("wrench guard triggered")
