# FR3 real-robot O/C/Y migration (v0.1)

This package is a fail-safe, importable first migration layer. It performs no robot or
gripper movement. `motion_enabled` defaults to `False`; the FR3 PTP goal is deliberately
unimplemented until the onsite output of `ros2 interface show franka_msgs/action/PTPMotion`
is recorded. Camera intrinsics must come from confirmed CameraInfo and are never hard-coded.

Run offline with `python -m real_robot.run_real_ocy --stage ocy --dry-run`. Robot and
perception dependency checks are available through `--check-robot` and
`--check-perception`; ROS2 imports are lazy so dry-run works without ROS2.

## Baseline audit and migration boundary

The source audit covered the actual `teleop/` matches, principally
`o_prop_baseline.py`, `c_prop_baseline.py`, `y_prop_baseline.py`,
`y_prop_controller.py`, `run_oc_baseline.py`, and `run_ocy_baseline.py`.

Migrated concepts are: explicit staged state machines with settle/verify gates; the O
planner's obstacle boundary radius, sampled arc and tangent lead, including raised entry;
the C mouth-to-seat-to-through-exit route; Y's symmetric dual material-point targets and
midpoint/span/temporal-stability inspection; and ordered fixed-to-free cable centerline
sampling by normalized arc length. Verification is observation-based: O winding/proximity,
C occupancy on both sides of its seat, and Y midpoint/span/hold stability are intended
interfaces, not claims of achieved perception.

Not migrated: direct MuJoCo qpos/qvel writes, privileged body/geom state, `xfrc_applied`,
simulation contact force, virtual/bounded grasp assist, teleport/base teleport, C retention
seed or force field, cable velocity decay, direct cable-state mutation, and any simulator
mechanism that manufactures retention or success. MuJoCo control gains and collision/contact
thresholds are not real safety limits.

## Onsite work still required

- Confirm ROS2 message/action schemas, QoS and live connectivity for both pose topics and
  both PTP action servers; then design a separately reviewed motion implementation.
- Confirm common task frame, camera optical frames, TF tree, timestamps and calibration.
- Confirm each camera's CameraInfo topics, distortion model, depth units/alignment and sync.
- Confirm Robotiq driver topics/actions/services, feedback, units, limits and safe stop.
- Validate workspace, waypoint-delta, translation/rotation-speed and wrench limits. Every
  real threshold is `REQUIRES_ONSITE_VALIDATION` and must pass the facility safety process.
