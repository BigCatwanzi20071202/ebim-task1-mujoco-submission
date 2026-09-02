"""Ordered cable material-point contract and geometric sampling helper."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CableDetection:
    points: np.ndarray
    frame_id: str
    ordered_fixed_to_free: bool
    confidence: float


class CableDetector:
    def from_ordered_points(self, points: np.ndarray, frame_id: str) -> CableDetection:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3 or not np.all(np.isfinite(pts)):
            raise ValueError("an ordered finite Nx3 cable centerline is required")
        return CableDetection(pts, frame_id, True, 0.0)

    @staticmethod
    def select_material_points(points: np.ndarray, fractions: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        pts = np.asarray(points, dtype=float)
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if lengths[-1] <= 0:
            raise ValueError("cable centerline has zero length")
        selected = []
        for fraction in fractions:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("material fractions must be within [0, 1]")
            target = fraction * lengths[-1]
            i = min(int(np.searchsorted(lengths, target, side="right") - 1), len(pts) - 2)
            alpha = (target - lengths[i]) / max(lengths[i + 1] - lengths[i], 1e-12)
            selected.append((1.0 - alpha) * pts[i] + alpha * pts[i + 1])
        return tuple(selected)
