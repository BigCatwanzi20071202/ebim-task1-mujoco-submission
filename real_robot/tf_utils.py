"""Small ROS-independent rigid-transform helpers (xyzw quaternion convention)."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    matrix: np.ndarray
    source_frame: str
    target_frame: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("transform must be a finite 4x4 matrix")
        object.__setattr__(self, "matrix", matrix)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        flat = pts.reshape(-1, 3)
        out = (self.matrix[:3, :3] @ flat.T).T + self.matrix[:3, 3]
        return out.reshape(pts.shape)


def compose(a: RigidTransform, b: RigidTransform) -> RigidTransform:
    if b.target_frame != a.source_frame:
        raise ValueError("transform frame chain mismatch")
    return RigidTransform(a.matrix @ b.matrix, b.source_frame, a.target_frame)
