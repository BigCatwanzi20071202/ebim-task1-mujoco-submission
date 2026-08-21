# EBiM Task 1 MuJoCo runtime and ManipulationNet evaluation stack.
# Build from the Git repository root:
#
#   docker build --pull -t ebim-task1-mujoco:latest .
FROM ros:humble-ros-base-jammy

ARG SIM_DIR=task1_mujoco/robotiq_duo_full_scene_minimal_core
ARG MNET_DIR=task1_mujoco/mnet_client-ros_2
ARG TELEOP_PKGS_DIR=task1_mujoco/teleop_ros2
ARG SOURCE_REPO=https://github.com/BigCatwanzi20071202/ebim-task1-mujoco

LABEL org.opencontainers.image.source=${SOURCE_REPO}
LABEL org.opencontainers.image.description="EBiM Task 1 cable management submission: MuJoCo, ROS 2 Humble, teleoperation publishers, and ManipulationNet client"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ffmpeg \
    libglfw3 \
    libgl1 \
    libglu1-mesa \
    libegl1 \
    libvulkan1 \
    ros-humble-cv-bridge \
    python3-opencv \
 && rm -rf /var/lib/apt/lists/*

# NumPy remains below 2 for compatibility with ROS 2 Humble cv_bridge.
# Ubuntu's OpenCV package uses the system FFmpeg/H.264 stack required for
# ManipulationNet evidence-video recording.
RUN pip3 install --no-cache-dir \
    mujoco==3.9.0 "numpy>=1.24,<2" glfw==2.10.0 pygame==2.6.1 pillow \
    "pydantic>=2,<3" "requests>=2.32" "tqdm>=4.67" \
    pupil-apriltags pybullet python-xlib \
    pyopenxr==1.1.5301 PyOpenGL==3.1.10

WORKDIR /ws
COPY ${MNET_DIR} /ws/src/mnet_client
COPY ${TELEOP_PKGS_DIR}/keyboard_teleop_publisher /ws/src/teleop_ros2/keyboard_teleop_publisher
COPY ${TELEOP_PKGS_DIR}/gamepad_teleop_publisher /ws/src/teleop_ros2/gamepad_teleop_publisher
COPY ${TELEOP_PKGS_DIR}/vr_teleop_publisher /ws/src/teleop_ros2/vr_teleop_publisher
RUN . /opt/ros/humble/setup.sh && colcon build

COPY ${SIM_DIR} /ws/sim
RUN echo 'source /opt/ros/humble/setup.bash && source /ws/install/setup.bash' >> /root/.bashrc \
 && install -m 755 /ws/sim/release/ebim /usr/local/bin/ebim

WORKDIR /ws/sim
CMD ["ebim", "help"]
