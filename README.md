# GG boy — EBiM Task 1 Phase II minimal safe-hold policy

This pinned branch contains a deliberately simple, runnable Phase II baseline for
Task 1 (Cable Routing & Plugging). It observes both FR3 measured-joint-state topics
and sends **no robot, gripper, base, or spine commands**. Its expected Task 1 score
is zero. It is submitted to preserve an honest runnable baseline; it does not claim
cable routing, plugging, simulation success, or real-robot success.

The Phase I MuJoCo submission remains under `task1_mujoco/`. The Phase II technical
report remains under `phase2_technical_report/`.

## Exact build and launch commands

From a clean checkout of the pinned commit:

```bash
docker build --pull -t ggboy-task1-phase2-safe-hold:20260912 .
docker run --rm ggboy-task1-phase2-safe-hold:20260912 self-test
docker run --rm --init --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ggboy-task1-phase2-safe-hold:20260912 run
```

The operator may replace `RMW_IMPLEMENTATION` with the testbed's installed ROS 2
middleware. If the host uses CycloneDDS, pass its normal middleware setting and
mounted DDS configuration. Runtime internet access is not required.

Successful startup requires one valid seven-joint `sensor_msgs/msg/JointState`
message from each of:

- `/left/franka_robot_state_broadcaster/measured_joint_states`
- `/right/franka_robot_state_broadcaster/measured_joint_states`

The process reports `ACTIVE_SAFE_HOLD_NO_COMMAND` and remains alive after both streams
are observed. It exits nonzero if either stream is missing, malformed, stale, or stops.
It never creates a publisher, action client, service client, or robot network socket.

## Environment and dependencies

- Linux x86_64 with Docker Engine
- ROS 2 Jazzy DDS network access to the robot-side ROS graph
- Base image: `ros:jazzy-ros-base-noble`
- No CUDA, GPU, model weights, dataset, credentials, or runtime network download
- Expected monitoring rate: callbacks at the source topic rate; health check at 10 Hz

## Hardware assumptions

- Mobile FR3 Duo controllers and state broadcasters are already started by the testbed operator.
- Both measured-joint-state topics above use seven finite positions with unique joint names.
- `ROS_DOMAIN_ID`, middleware, DDS discovery and host networking match the testbed.
- The controller is already in the facility-approved stationary/holding state.
- Arms, grippers, base and spine start in a safe pose chosen by the operator.
- No object pose, camera pose, fiducial, hard-coded task pose or operator input is consumed.
- Between rounds, the operator performs the normal physical reset; this policy changes nothing.

## Safety and scoring behavior

This baseline is autonomous in the narrow sense that it starts and monitors state without
operator input. It intentionally attempts none of the Task 1 objectives and should receive
zero task points. No externally supplied object poses are used. The organizer-released
trajectory dataset informed earlier diagnostics but is not loaded by this baseline.

The operator has full authority to stop the container or robot at any time. Missing or stale
state causes a fail-closed exit. The absence of command paths is intentional and testable.

## Local source test

```bash
python3 -m unittest discover -s phase2_minimal_policy/tests -v
python3 phase2_minimal_policy/policy.py self-test
```

GitHub Actions repeats the source tests, builds the image from a clean checkout, runs
the in-container self-test, and launches the policy without a ROS graph to verify the
documented timeout and fail-closed exit. A physical robot is not available in CI; live
topic discovery remains an organizer-site requirement.
