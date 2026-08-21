"""Runtime entry for the stage-1 single C-prop primitive."""

from __future__ import annotations

from . import config
from .c_prop_baseline import CPropBaseline
from .run_baseline import run_controller


def configure_c_baseline() -> None:
    """Apply the C primitive's contact and motion limits."""
    config.BASELINE_GRIPPER_BACKOFF = config.C_BASELINE_GRIPPER_BACKOFF
    config.BASELINE_GRASP_POSITION_TOL = config.C_BASELINE_GRASP_POSITION_TOL
    config.BASELINE_GRASP_ALIGN_TOL = config.C_BASELINE_GRASP_ALIGN_TOL
    config.GRASP_ASSIST_START_DELAY = config.C_BASELINE_ASSIST_START_DELAY
    config.GRASP_ASSIST_RAMP_TIME = config.C_BASELINE_ASSIST_RAMP_TIME
    config.GRASP_ASSIST_MAX_FORCE = config.C_BASELINE_ASSIST_MAX_FORCE
    config.GRASP_NOCONTACT_RELEASE_TIME = config.C_BASELINE_NOCONTACT_RELEASE_TIME
    config.GRASP_ASSIST_RELEASE_DIST = config.C_BASELINE_ASSIST_RELEASE_DISTANCE
    config.BASELINE_MAX_CABLE_GRASP_FORCE = config.C_BASELINE_MAX_CABLE_GRASP_FORCE
    config.BASELINE_STATE_TIMEOUT = config.C_BASELINE_STATE_TIMEOUT


def main(args) -> None:
    configure_c_baseline()
    run_controller(args, CPropBaseline)
