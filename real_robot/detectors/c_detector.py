"""C-slot landmark representation; segmentation remains a field integration item."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CDetection:
    center: np.ndarray
    mouth_direction: np.ndarray
    frame_id: str
    confidence: float


class CDetector:
    def from_landmarks(self, center: np.ndarray, mouth_point: np.ndarray, frame_id: str) -> CDetection:
        center = np.asarray(center, dtype=float)
        direction = np.asarray(mouth_point, dtype=float) - center
        norm = float(np.linalg.norm(direction))
        if center.shape != (3,) or direction.shape != (3,) or norm <= 1e-9:
            raise ValueError("valid 3D C center and mouth landmark are required")
        return CDetection(center, direction / norm, frame_id, 0.0)
