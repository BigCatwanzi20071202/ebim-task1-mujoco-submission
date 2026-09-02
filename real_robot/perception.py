"""RGB-D observations and metric back-projection without hard-coded intrinsics."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("valid CameraInfo-derived focal lengths are required")


@dataclass(frozen=True)
class RGBDFrame:
    color: np.ndarray
    depth_m: np.ndarray
    frame_id: str
    stamp_s: float
    intrinsics: CameraIntrinsics


def deproject_pixel(u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be finite and positive")
    return np.array([(u - intrinsics.cx) * depth_m / intrinsics.fx,
                     (v - intrinsics.cy) * depth_m / intrinsics.fy,
                     depth_m], dtype=float)


def check_perception_runtime() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        return False, f"NumPy unavailable: {exc}"
    return True, "offline perception primitives available; cameras/CameraInfo not probed"
