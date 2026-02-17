# TG2 Inspirehand Deployment

## Setup (Docker, ROS1)

This deployment uses a ROS1 Noetic container (see `Dockerfile` + `docker-compose.yml`).

### Prerequisites

- Docker + Docker Compose v2 (`docker compose`)
- (Optional) NVIDIA Container Toolkit for GPU acceleration
- X11 enabled on the host for GUI tools (RViz, etc.)

## Set up the container environment

1. Robot side (192.168.41.1)

```bash
ssh ubuntu@192.168.41.1 
sudo su
# both passwords are tg2TUM2025
source /home/ubuntu/rosws/install_isolated/setup.bash

export ROS_MASTER_URI=http://192.168.41.1:11311
export ROS_IP=192.168.41.1
roscore
```

Make sure you launch the body control node:

```bash
roslaunch body_control body.launch
```

Reminder: every robot-side terminal must export the same `ROS_MASTER_URI` and `ROS_IP` before launching ROS tools.

2. Host workstation (192.168.41.108) — build the Docker image once:

```bash
cd ~/projects/tg2_dexman_isaac/dextrah_lab/deployment_tg2_inspirehand
./scripts/build.sh
```

3. Start the workspace container pointing at the robot master:

```bash
./scripts/run_with_robot.sh --master 192.168.41.1:11311 --advertise 192.168.41.108
```

Inside the shell it opens, verify the environment and connectivity:

```bash
env | grep -E 'ROS_MASTER_URI|ROS_HOSTNAME|ROS_IP'   # expect 192.168.41.* values
ping -c 1 192.168.41.1
rostopic list   # should display the robot's topics
```

If `ROS_IP` is missing, export it manually:

```bash
export ROS_IP=192.168.41.108
```

Reminder: each new shell in the container must export `ROS_MASTER_URI` and `ROS_IP` (or reuse the values above) before running ROS tools.

4. Build and source the catkin workspace:

```bash
cd /tiangong_infra_ws/ws
catkin_make
source devel/setup.bash
```

5. Sanity checks before commanding the real arm:

```bash
rostopic echo /arm/status -n 3   # robot publishes status
rostopic echo /arm/cmd_pos -n 3   # verify velocities look safe
```

You should now see the robot’s topics from the container.

6. Launch the control bridge(s) for execution

Single-arm feedback control bridge (fake controller → real arm).

To publish `sensor_msgs/JointState` commands to the robot, launch the bridge:

```bash
roslaunch feedback_control_bridge feedback_control_bridge.launch \
  arm_side:=right \
  input_joint_state_topic:=/arm/command_joint_states \
  command_topic:=/arm/cmd_pos \
  status_topic:=/arm/status \
  default_velocity_rpm:=10.0 \
  velocity_rpm:="[15,15,15,15,15,15,15]"
```

## Camera node launch

Launch the stereo camera ROS1 publisher from inside the running container:

```bash
cd /tiangong_infra_ws/ws
catkin_make
source devel/setup.bash
roslaunch stereo_camera stereo_ros_publisher.launch
```

Default launch resolution is `320x240` (matching the distillation pipeline input size).

Default topics:
- `/stereo/left/image_raw`
- `/stereo/right/image_raw`

Quick checks:

```bash
rostopic list | grep stereo
rostopic hz /stereo/left/image_raw
rostopic hz /stereo/right/image_raw
```

Visualization test (side-by-side viewer for left/right streams):

```bash
python3 /tiangong_infra_ws/ws/src/calibration_tests/tests/test_stereo_camera_node.py \
  --left-topic /stereo/left/image_raw \
  --right-topic /stereo/right/image_raw \
  --no-stereo
```

If the stereo publisher is not already running, remove `--no-stereo`.

Optional launch overrides (example):

```bash
roslaunch stereo_camera stereo_ros_publisher.launch \
  width:=320 \
  height:=240 \
  left_config:=ov9732_L \
  right_config:=ov9732_R \
  left_topic:=/stereo/left/image_raw \
  right_topic:=/stereo/right/image_raw \
  flip:=vertical \
  rate:=30
```
## Camera Intrinsic Calibration

Run intrinsic/stereo calibration when:
- You use a new camera pair.
- You re-mount cameras or change baseline/orientation.
- You change capture resolution (for distillation, target is `320x240`).
- Depth/disparity quality degrades or left/right rectification looks wrong.

If hardware and resolution are unchanged and depth is stable, you usually do not need to recalibrate every run.

Detailed calibration guide (recommended):
- `/tiangong_infra_ws/ws/src/stereo_camera/README.md`

Quick command (inside the container):

```bash
cd /tiangong_infra_ws/ws/src/stereo_camera
python3 tests/camera_calibration.py \
  --left-config ov9732_L --right-config ov9732_R \
  --square-size-mm 25 --board-cols 8 --board-rows 6 \
  --frames 50 --flip vertical \
  --save-dir tests/calibration --camera-name jetson_stereo
```

Capture controls:
- Press `c` to capture a synchronized checkerboard pair.
- Press `x` to abort.

Expected outputs:
- `tests/calibration/jetson_stereo.npz`
- `tests/calibration/jetson_stereoc1.npz`
- `tests/calibration/jetson_stereoc2.npz`

## Camera Calibration (Hand-Eye)

This folder provides a TG2 Inspirehand calibration script that estimates the
camera-to-robot transform using an AR tag and robot joint states.

### Prerequisites

- ROS 2 is running and the TG2 controller publishes joint states.
- The TG2 pose command topic is active and accepts `sensor_msgs/JointState` pose commands.
- An AR tag detector publishes `tag36h11:0` (or your configured tag) on `/tf`.
- The URDF path matches the TG2 model that your joint state names correspond to.

### Usage

From repo root:

```bash
python tg2_dexman_isaac/dextrah_lab/deployment_tg2_inspirehand/calibration/camera_calibration.py \
  --camera left \
  --home-pose x y z yaw pitch roll \
  --target-pose x y z yaw pitch roll \
  --joint-state-topic /tg2/joint_states \
  --pose-command-topic /tg2_inspirehand_fabric/pose_commands \
  --tag-frame-id tag36h11:0
```

Notes:
- Euler angles are **ZYX (yaw, pitch, roll)** in radians.
- The script interpolates between `home_pose` and `target_pose`, logs (camera→tag, robot joints), runs Gauss‑Newton, and saves
  `robot_cam_<camera>_calibration.txt` in the current working directory.

### Common Options

```text
--camera                right | left | center (required)
--urdf                  TG2 URDF path (default: tg2_with_hands_no_legs.urdf)
--palm-link             palm link name in URDF (default: palm)
--device                cuda | cpu (default: cuda)
--joint-state-topic     JointState feedback topic
--pose-command-topic    JointState pose command topic
--tf-topic              TF topic (default: /tf)
--tag-frame-id          tag36h11:0 (default)
--home-pose             x y z yaw pitch roll
--target-pose           x y z yaw pitch roll
--num-steps             number of interpolated setpoints (default: 30)
--fabric-vel-threshold  velocity threshold for settling (default: 0.05)
--publish-dt            pose command publish period (default: 1/30)
```

### Outputs

- `robot_cam_<camera>_calibration.txt` (robot→camera 4x4 transform).
