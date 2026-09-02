"""O landmark result and simple point-cloud circle estimator."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ODetection:
    center: np.ndarray
    radius_m: float
    frame_id: str
    confidence: float


class ODetector:
    def detect_points(self, points: np.ndarray, frame_id: str) -> ODetection:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 6 or pts.shape[1] != 3:
            raise ValueError("at least six 3D O-boundary points are required")
        center = pts.mean(axis=0)
        radius = float(np.median(np.linalg.norm(pts[:, :2] - center[:2], axis=1)))
        return ODetection(center, radius, frame_id, 0.0)  # confidence awaits real validation
