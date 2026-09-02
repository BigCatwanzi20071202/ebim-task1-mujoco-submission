"""Y checkpoint representation with a normalized stem direction."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class YDetection:
    center: np.ndarray
    stem_direction: np.ndarray
    arm_span_m: float
    frame_id: str
    confidence: float


class YDetector:
    def from_landmarks(self, center: np.ndarray, stem_point: np.ndarray, arm_span_m: float, frame_id: str) -> YDetection:
        center = np.asarray(center, dtype=float)
        direction = np.asarray(stem_point, dtype=float) - center
        norm = float(np.linalg.norm(direction))
        if center.shape != (3,) or direction.shape != (3,) or norm <= 1e-9 or arm_span_m <= 0:
            raise ValueError("valid Y landmarks and positive span are required")
        return YDetection(center, direction / norm, float(arm_span_m), frame_id, 0.0)
