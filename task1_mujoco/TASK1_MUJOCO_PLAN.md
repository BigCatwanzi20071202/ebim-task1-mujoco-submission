# Task 1 MuJoCo implementation plan

Last reviewed: 2026-07-30. The competition site labels the current release a
developer preview, so dates and submission details must be rechecked before a
real submission.

## Official task contract

The official Task 1 objective is to manipulate a flexible cable through an
ordered fixture sequence: route around O-props in the required direction,
thread through C-props, and reach the Y checkpoints. The Y steps require cable
tension control and bimanual coordination. A run is fully autonomous and has a
30-minute limit.

The primary outcome is the fraction of the fixture sequence completed in the
correct order, validated at the Y checkpoints. Only the uninterrupted correct
prefix from the start counts; an incorrect routing invalidates the remaining
sequence. Completion time, regrasps and handovers are secondary measures.
SPARC, torque, simultaneous manipulation and tension-related process measures
are reported separately. Excessive force or damage is a hard safety failure.

Official submission is one GitHub issue per team and task in the public
[`EBiM-Benchmark/submissions`](https://github.com/EBiM-Benchmark/submissions)
repository. The linked team repository must be public and contain a Dockerfile
and run instructions. Source publication is not required. Weights or other
supplementary assets are allowed when their integration is documented. A newer
issue supersedes an older submission.

The competition page currently says evaluation uses Docker jobs scheduled in a
Google Cloud/AMD GPU pool with static analysis. It does **not** publish exact
CPU, RAM, storage, image-size, startup-time or network quotas. Those values
must not be assumed. The local ManipulationNet client is a development and
evidence-capture path; it is not a substitute for the competition's GitHub
submission and organizer-side scoring.

The schedule shown on 2026-07-29 gives the Phase-I simulation deadline as
August 10, 2026 AoE and results on August 15. Advancing teams get local
hands-on bench testing August 17--25; organizer-run physical testing and code
updates continue August 26--September 10, with the Phase-II deadline on
September 10 AoE. The developer-preview APIs and schedule remain subject to
change, so recheck the competition page immediately before submission.

Primary references:

- [Competition rules and schedule](https://ebim-benchmark.github.io/competition.html)
- [Official benchmark repository](https://github.com/EBiM-Benchmark/benchmark)
- [Official submissions repository](https://github.com/EBiM-Benchmark/submissions)
- [ManipulationNet ROS 2 client guide](https://mnet-client.readthedocs.io/ros_2/general.html)

For the ROS 2 evidence interface, `team_config.json` supplies the image and
camera-info topics, writable result directory, team code and autonomy level.
The local default image topic is `/mujoco/camera/image_raw` at 30 FPS; the
client requires a `sensor_msgs/Image` stream at at least 25 FPS plus
CameraInfo. Runtime task and board configuration arrive on
`/mnet_client/ongoing_task` and
`/mnet_client/board_configuration`; completion and skip are Trigger services.
Task 1 competition runs require full autonomy. The one-time code must remain
visible in the evidence camera. The submission client is rate-limited, though
the public guide does not state an exact weekly count.

## Local execution path

The MuJoCo implementation is a single process:

1. `robotiq_duo_full_scene_minimal_core/main.py` dispatches the selected mode.
2. `teleop/session.py` owns MuJoCo model/data, both arms, the mobile base,
   cable state and each physics step.
3. `teleop/robot_arm.py`, `grasping.py` and `base_drive.py` provide the shared
   DLS IK, contact-aware twist clamp, force-servo gripper, capped grasp assist
   and base control.
4. `teleop/mnet_bridge.py` and `mnet_board.py` implement the ROS 2 evidence,
   task/configuration and Tier-2 fixture-randomization path.
5. `release/Dockerfile.eval`, `docker-run.sh` and `release/ebim` are the
   packaging and container entry path.

The root Task 1 README currently marks its ManipulationNet eval section as
temporarily on hold pending a project decision, while the bridge and Docker
code remain present. Therefore this stage does not report a partial primitive
as an official MNet task completion.

The vendored local Tier-2 board contains two adapters, one C-clip and four
round pegs. A clearly labelled `yclip_0` proxy has now been added only for
fixed-layout stage-1 Y-controller development because the official F3 mesh is
not present in the developer-preview assets. It must not be treated as
submission geometry. `mnet_board.py` applies the incoming fixture coordinates and stores
the local ordered routing configuration `[0, -1, +2, +4, -3, -5, 6]`; this is
the simulator's current task contract, not a claim that all of those primitives
are implemented by the stage-1 controller.

## Architecture and implementation order

The intended architecture keeps a deterministic closed-loop controller as the
main path. A learned vision module may later replace the privileged MuJoCo
locator, and a learned local manipulation policy may be used only where model
based motion is consistently weak. A VLA may eventually select high-level
fixture primitives, but it must not directly replace the safety controller or
claim completion without geometric verification.

Work is split into independently testable increments:

1. **Single O-prop primitive (implemented):** locate one peg and a safe
   cable segment, align the base, grasp, lift, follow a directional arc,
   release, retreat, settle and verify winding.
2. **Single C-prop primitive (implemented, requalification failing):** capture near the free end,
   approach the open mouth, insert to the retained channel, release and verify
   a continuous settled centerline crossing with cable on both sides.
3. **In progress:** repeat the O primitive across all four pegs, both
   directions and randomized Tier-2 coordinates; replace fixed tuning with
   reachability/collision costs.
4. Add a primitive sequencer that consumes the announced ordered fixture
   configuration and preserves the correct-prefix invariant.
5. **Partially implemented:** the Y checkpoint geometry boundary,
   ordered-cable verifier, outside-table mobile-base stance planner and guarded
   two-arm handover/tension state machine are present; safe bimanual
   lift/routing qualification remains.
6. Replace MuJoCo truth localization with calibrated overhead-camera
   segmentation/keypoints and uncertainty-aware retries.
7. Exercise the full ROS 2 evidence path, Docker cold start, one-time-code
   visibility, 25+ FPS camera rate and organizer-provided cloud limits.

## Stage-1 O-prop controller

Run the deterministic default (`round_peg_3`, counter-clockwise):

```bash
cd task1_mujoco/robotiq_duo_full_scene_minimal_core
python3 main.py --input baseline
python3 main.py --input baseline --no-viewer
```

The stage-1 baseline defaults to `noslip_iterations=0`, its currently verified
contact configuration. Other simulator modes retain the XML value of 20. To
reproduce the baseline default explicitly:

```bash
python3 main.py --input baseline --no-viewer --noslip-iterations 0
```

In the distribution container the equivalent entry point is:

```bash
ebim baseline --no-viewer
```

Useful options are `--baseline-prop round_peg_0..3`,
`--baseline-direction ccw|cw`, `--baseline-timeout SECONDS`,
`--randomize-board` and `--randomize-seed`.

The state sequence is:

```text
LOCATE -> BASE_ALIGN -> APPROACH -> DESCEND -> GRASP_ALIGN -> CLOSE
       -> LIFT -> PREWRAP -> WRAP -> LOWER -> RELEASE -> RETREAT
       -> SETTLE -> SUCCEEDED
```

Any timeout, non-finite simulation state, lost grasp, excessive obstacle or
cable-grasp force, closed viewer, failed winding check or unsettled cable goes
to `FAILED`, zeros external forces, stops the base and holds both arms. The
controller uses MuJoCo state for localization in this stage, but reuses the
existing IK, contact clamp, base driver, gripper force servo and capped grasp
assist. It intentionally does not call the MNet finished service.

The result detector computes signed local winding from ordered cable segments,
requires a direction-adjusted increase of at least 80 degrees, checks that at
least two segments remain near the prop and now rejects a nearest cable centre
farther than 28 mm. The path uses a 17.5 mm cable-centre boundary and a 25 mm
tangent lead instead of the former 90 mm clearance circle. In combined mode
the O grasp remains closed and the dynamic prefix is passed directly to C.

Tests:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q main.py teleop tests
python3 main.py --input baseline --help
```

Confirmed headless run (`--noslip-iterations 0`) on 2026-07-29: success at
14.774 simulated seconds; signed winding changed from -111.6 degrees to +98.0
degrees, giving +209.6 degrees in the requested CCW direction; nearest cable
distance was 0.089 m.

The first multi-peg increment adds a board-clear transfer planner. A target
within 0.25 m keeps the original direct PREWRAP; a farther target raises the
held cable to 0.20 m above the prop, translates at that height, and descends
only at the route radius. Unit tests cover both transfer forms and route
direction. Fixed-layout qualification currently gives these results:

- `round_peg_3 ccw` still succeeds at 14.786 simulated seconds.
- `round_peg_2 ccw` safely completes all three transfer waypoints, then reaches
  the right-arm workspace boundary at WRAP waypoint 3/9 and safely stops at
  22.004 simulated seconds.
- `round_peg_0 ccw` no longer collides with the board during transfer, but the
  arm reaches its workspace limit before the distant prewrap target and safely
  times out.
- `round_peg_3 cw` completes the motion safely but the released cable takes the
  shorter, unwound topology; the retained 270-degree planner fails the winding
  check. An experimental 180-degree shortcut also failed and was removed.

The O controller grasps the last board-safe segment rather than the physical
cable tip: eight free tail segments remain beyond that grasp in the fixed
layout. That tail is the leading explanation for the CW topology relaxing
after release. A first target-aware pre-grasp increment is now implemented:
for distant props the empty arm is biased away from the target before the base
aligns the gripper with the cable, preserving joint workspace in the direction
of the subsequent route. Nearby targets retain the previously verified
posture. Grasp selection now explicitly takes the last fixed-to-free ordered
board-safe segment and reports how many uncontrolled tail segments remain,
rather than relying on proximity to select that segment accidentally.

The short Docker qualification for this increment passes all 10 unit tests,
`compileall` and baseline CLI help. Per the recorded-result policy, no already
passed long simulation was repeated. Peg 0/2 and CW still require new targeted
simulation qualification; the remaining CW step is active tail control (a
tip regrasp or second-arm constraint), not a longer open-loop arc.

## Stage-1 C-prop controller

Run with a viewer or headlessly:

```bash
cd task1_mujoco
./docker-run.sh --input baseline_c
./docker-run.sh --input baseline_c --no-viewer
```

The C controller reuses the O controller's state machine and safety checks,
but selects a free-end cable segment and a base orientation whose arm workspace
contains both the grasp point and clip. It parks the base, uses the shared DLS
IK solution through a deterministic bounded joint-step servo, captures the
cable with the existing capped-force grasp assist while preserving the
capture-time cable/gripper offset, and plans waypoints in the C-clip body
frame. The desired cable centerline—not the gripper center—is placed first in
front of the mouth and then at `CLIP_SEAT_LOCAL`.

Verification requires at least one cable body or continuous cable edge through
the channel, material on both local-X sides, and peak cable speed below the
settled limit. The continuous-edge test avoids false negatives caused by the
composite cable's finite segment spacing.

An earlier deterministic Docker headless run (`timestep=0.002`,
`noslip_iterations=0`) succeeded at 47.700 simulated seconds with one segment
inside the channel, one crossing edge, cable on both sides and peak cable speed
0.002 m/s. A 2026-07-29 requalification did not reproduce that result: it
safely stopped at 5.988 simulated seconds after the gripper formed no two-pad
cable contact and the `CLOSE` state timed out. The C primitive is therefore
kept as an experimental next increment; the currently qualified stage-1 MVP
is the fixed-layout `round_peg_3` CCW O-prop route.

The `baseline_oc` entry point runs the O primitive and then the C primitive in
one uninterrupted simulation. It preserves the O result, resets the active arm
to a known posture, realigns the mobile base to the post-O cable location,
raises the captured cable, transfers toward the clip, traverses the mouth,
seat and exit, then returns to the retention center before release.

A deterministic Docker headless run on 2026-07-29 completed both checks at
60.642 simulated seconds. O succeeded at 14.774 seconds with +209.6 degrees of
directed winding gain. C verification found one continuous crossing edge,
material on both sides and settled cable speed of 0.038 m/s.

This combined result is a simulator baseline, not a hardware-ready policy. The
post-O cable can lag the gripper because the capped grasp assist acts on
discrete composite bodies. At final C release, `baseline_oc` therefore projects
the currently held cable body to the clip retention center before returning
control to MuJoCo for retreat, settling and independent geometric
verification. This privileged retention seed must be replaced by a physical
regrasp or two-arm handover before hardware transfer.

## Current limitations

- Localization is privileged MuJoCo truth, not camera perception.
- Only the fixed-layout `round_peg_3` CCW path has an end-to-end confirmed run;
  other pegs, CW routing and randomized boards are exposed for testing but are
  not yet qualified.
- The C-prop controller currently fails its requalification at cable capture;
  its older successful run is not treated as a current guarantee.
- The new raised transfer prevents the observed low PREWRAP board collision,
  but it does not extend the arm workspace. Pegs 0 and 2 require a target-aware
  mobile-base stance, regrasp or handover before their routes can be qualified.
- The cable-grasp contact cap is intentionally separate from the lower
  obstacle-contact cap because the current mesh/capsule contact model produces
  large positive-margin forces during a valid grasp. This requires calibration
  against the organizer's force metric before submission.
- A qualification run with the XML default `--noslip-iterations 20` safely
  stopped at PREWRAP because the mesh/capsule grasp force reached about 681 N.
  Contact calibration under the full no-slip solver remains required; the
  baseline mode's default of 0 is a stage-1 constraint, not a hidden success.
- The C primitive uses privileged geometry and a capped-force capture assist
  with a preserved relative offset because the present fingertip meshes cannot
  make a clean two-pad free-end contact without touching the board. It is a
  simulation baseline, not yet a hardware-ready grasp policy.
- No general ordered multi-fixture sequence, qualified Y completion, retry policy or
  learned perception is implemented yet. The experimental Y state machine
  selects ordered material points, jointly scores arm reach and TCP travel,
  follows an outside-table base path and synchronously prepositions both arms.
  After dual capture it preserves the inter-gripper vector, adds 10 mm of
  symmetric tension, translates the cable midpoint to the Y region, stops and
  holds both closed grippers for five seconds before checking centre, span and
  settled speed. The
  fixed-layout O-C-Y run captures both selected material points and enters
  bimanual lift. A 2026-08-01 qualification safely stopped there when an
  adjacent cable segment produced 5273.1 N of right-gripper contact force
  against the unchanged 2500 N limit. Material-pair planning now also requires
  0.18 m of world-space gripper clearance; this removed the unsafe pair in the
  next run, which then stopped at the conservative reach gate because the base
  had used the former 25 mm alignment tolerance. The Y-only tolerance is now
  5 mm and its geometry/planning tests pass, but that last adjustment has not
  yet completed a full headless qualification. Increasing or filtering the
  force limit remains explicitly out of scope.
- The revised held O route is qualified through the O-to-C handoff boundary:
  +140.8 degrees of directed winding and 24 mm nearest cable-centre distance.
  The first direct C transfer exceeded the cable-grasp force gate; a 35 mm/s
  base stance change with a 0.20 gripper target is implemented but not yet
  fully qualified. If it still exceeds the unchanged 2500 N limit, a physical
  two-arm handover is required. The final C retention seed also remains a
  privileged MuJoCo operation.
- The partial primitive does not constitute a complete Task 1 submission.
