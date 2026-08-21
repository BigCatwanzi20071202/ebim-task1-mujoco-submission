# EBiM Task 1 — Cable Management (MuJoCo)

A runnable EBiM 2026 Phase I Task 1 submission for cable routing with the Mobile FR3 Duo in MuJoCo.

## Submission

| Field | Value |
|---|---|
| Competition | EBiM Competition 2026, Phase I — Simulation |
| Track | Track 1 / Task 1 |
| Simulator | MuJoCo 3.9.0 |
| Robot | Mobile FR3 Duo with dual Robotiq 2F-85 grippers |
| Task | Cable Management / cable routing |
| Main entry | `/ws/sim/main.py` via the `ebim` launcher |
| Container | `ebim-task1-mujoco:latest` |

## Requirements

- Linux x86_64.
- Docker Engine.
- Docker Compose v2 only when using the included `docker-run.sh` helper.
- For graphical or official ManipulationNet evaluation: an NVIDIA GPU, NVIDIA driver, NVIDIA Container Toolkit, and X11 display access.

The retained Task 1 MuJoCo files are ordinary Git objects; this submission does not require Git LFS.

## Build

Clone the final public repository and build from its root:

```bash
git clone https://github.com/BigCatwanzi20071202/ebim-task1-mujoco-submission.git
cd ebim-task1-mujoco-submission

docker build --pull -t ebim-task1-mujoco:latest .
```

## Run

Show the available in-container entry points:

```bash
docker run --rm ebim-task1-mujoco:latest
```

Run the simulator with a viewer on a native Linux X11 desktop:

```bash
xhost +local:docker
docker run --rm -it --init \
  --network host --ipc host --gpus all \
  -e DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ebim-task1-mujoco:latest \
  ebim sim
```

`ebim sim` starts `main.py --input keyboard --mnet`. Additional simulator arguments may be appended after `ebim sim`.

### Local Baseline

The deterministic local O → C → Y validation is separate from the official ManipulationNet run:

```bash
docker run --rm --init --network host --ipc host \
  ebim-task1-mujoco:latest \
  ebim baseline-ocy --no-viewer
```

Remove `--no-viewer` and use the X11/GPU options from the viewer command above to inspect the motion graphically.

## ManipulationNet Evaluation

ManipulationNet evaluation requires a registered team's configuration. Before running it, update:

```text
task1_mujoco/mnet_client-ros_2/config/team_config.json
```

Set the camera topic to `/mujoco/camera/image_raw`, set `file_dir` to `/ws/out`, and keep credentials outside published commits. The repository contains only the public `TEST0000` placeholder.

Create the evidence-output directory:

```bash
mkdir -p mnet_out
```

Check registration and connectivity without consuming a scored attempt:

```bash
docker run --rm -it --network host --ipc host \
  -v "$PWD/task1_mujoco/mnet_client-ros_2/config/team_config.json:/ws/install/mnet_client/share/mnet_client/config/team_config.json:ro" \
  -v "$PWD/mnet_out:/ws/out" \
  ebim-task1-mujoco:latest \
  ebim ros2 run mnet_client connection_test
```

Start the simulator in one terminal using the viewer command from the previous section. In a second terminal, start the official, rate-limited ManipulationNet submission client:

```bash
docker run --rm -it --network host --ipc host \
  -v "$PWD/task1_mujoco/mnet_client-ros_2/config/team_config.json:/ws/install/mnet_client/share/mnet_client/config/team_config.json:ro" \
  -v "$PWD/mnet_out:/ws/out" \
  ebim-task1-mujoco:latest \
  ebim submit
```

The project helper exposes the equivalent client actions as:

```bash
cd task1_mujoco
./docker-run.sh connection-test
./docker-run.sh submit
```

ManipulationNet performance submission is distinct from the EBiM repository submission form.

## Repository Layout

```text
.
├── Dockerfile
├── README.md
├── LICENSE
├── LICENSES/
├── NOTICE
├── CONTRIBUTORS.md
├── .dockerignore
└── task1_mujoco/
    ├── docker-run.sh
    ├── robotiq_duo_full_scene_minimal_core/
    ├── mnet_client-ros_2/
    └── teleop_ros2/
```

The original Task 1 compose stack and `release/Dockerfile.eval` remain under the simulator directory for compatibility. The root `Dockerfile` is the standard competition build entry and uses the Git repository root as its build context.

## Reproducibility

- Upstream Task 1 benchmark base: `12bb48d1c1554c581c7abc2d9ee44df13c76b1df`.
- Frozen O/C/Y stabilization base: `91d50d0e3bf3ea3949de7cad59ab96858e46cf9c`.
- The submitted revision is the `main` branch of this repository.

## License

The repository is distributed under the Apache License 2.0. See `LICENSE`, `NOTICE`, `CONTRIBUTORS.md`, `LICENSES/`, `task1_mujoco/LICENSE`, and `task1_mujoco/mnet_client-ros_2/LICENSE` for upstream and third-party attribution.
