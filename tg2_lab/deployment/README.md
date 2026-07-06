# Tiangong Deployment ROS 2 Workspace

This directory is the ROS 2 Humble deployment workspace for the TG2 Inspirehand stack in this repository.

Active packages under `src/`:

- `bodyctrl_msgs`
- `calibration`
- `feedback_control_bridge`
- `inference_offline`
- `policy_inference_stereo_transformer`
- `stereo_camera`

## Table Of Contents

1. [Prerequisites](#prerequisites)
2. [Workspace Setup](#workspace-setup)
3. [Hardware Connection](#hardware-connection)
4. [Robot Bring-Up](#robot-bring-up)
5. [Feedback Control Loop](#feedback-control-loop)
6. [Offline Motion Tools](#offline-motion-tools)
7. [Stereo Camera And Calibration](#stereo-camera-and-calibration)

## Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- `python3-colcon-common-extensions`
- `python3-rosdep`
- `git`
- `ros-humble-xacro`
- `ros-humble-rviz2`

Recommended host setup:

```bash
sudo apt update
sudo apt install -y \
  git \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-xacro \
  ros-humble-rviz2

sudo rosdep init
rosdep update
```

All commands below assume:

```bash
source /opt/ros/humble/setup.bash
```

## Workspace Setup

Build this workspace from `deployment`:

```bash
source /opt/ros/humble/setup.bash
cd /home/chi/tg2_dexman_isaac/tg2_lab/deployment
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## Hardware Connection

Use this section for native host-based ROS 2 work against the real robot.

### Step 1: Power Up The Robot

Use the button panel on the back of the robot waist:

1. Press the large leftmost button once. It should turn green.
2. Release the emergency stop on the right.
3. Press the small rightmost button. After roughly 6 seconds the fans should start and the button should turn green.

Wait until the onboard computers and network interfaces finish booting before trying to connect from the workstation.

### Step 2: Host Network And SSH

For wired Ethernet:

- Use a Linux host with an Ethernet cable.
- Assign a static IP in the `192.168.41.xxx` range.
- Do not use `192.168.41.1`, `192.168.41.2`, or `192.168.41.3`.
- Use subnet mask `255.255.255.0`.
- Example host IP: `192.168.41.108/24`.

For wireless access:

- Connect to Wi-Fi `AIRhumanoid`.
- The Wi-Fi password matches the SSID.

Verify the motion-control board:

```bash
ping -c 1 192.168.41.1
ssh ubuntu@192.168.41.1
```

The motion-control board password is `123`.

If you need root on the board:

```bash
sudo su
```

The `sudo` password matches the SSH password.

If you also use the camera hosts:

```bash
ping -c 1 192.168.41.2
ssh nvidia@192.168.41.2
```

Known robot-side hosts:

- `192.168.41.1`: motion-control board
- `192.168.41.2`: head camera board
- `192.168.41.3`: second camera/compute board

## Robot Bring-Up

The low-level body-control node runs on the robot-side workspace, not in this repository. This workspace assumes that stack is already alive before you launch the bridge or the policy node here.

### Step 3: Disable Auto-Start If Needed

On the robot-side board:

```bash
sudo systemctl status proc_manager.service
```

If the service is still enabled:

```bash
sudo systemctl disable proc_manager.service
sudo reboot
```

After reboot, reconnect and confirm the target state is no longer active.

### Step 4: Launch The Robot-Side Body Control Node

From the robot-side ROS 2 workspace:

```bash
cd ros2ws
sudo su
source install/setup.bash
ros2 launch body_control body.launch.py
```

Wait for `All devices ready` before continuing.

From the workstation, verify that the expected robot-facing topics and services now exist:

```bash
ros2 topic list
ros2 service list
```

The bring-up below expects interfaces such as:

- `/arm/status`
- `/arm/cmd_pos`
- `/head/status`
- `/leg/status`
- `/waist/status`
- `/inspire_hand/state/right_hand`
- `/inspire_hand/ctrl/right_hand`
- `/inspire_hand/set_angle_flexible/right_hand`

## Feedback Control Loop

This workspace uses a simple pattern:

1. A control source publishes high-level joint targets to `/arm/command_joint_states`.
2. `feedback_control_bridge` converts those targets into robot-facing commands.
3. The bridge reads robot feedback and republishes merged `/joint_states`.
4. Test and execution scripts consume `/joint_states` as their start state.

### Step 1: Source The Workspace

```bash
source /opt/ros/humble/setup.bash
cd /home/chi/tg2_dexman_isaac/tg2_lab/deployment
source install/setup.bash
```

### Step 2: Launch The Stereo Camera Publisher

If your control path needs stereo images:

```bash
ros2 launch stereo_camera stereo_ros_publisher.launch.py \
  width:=320 \
  height:=240 \
  flip:=both
```

Default image topics are:

- `/stereo/left/image_raw`
- `/stereo/right/image_raw`

### Step 3: Launch The Policy Control Node

The stereo transformer node consumes stereo images and proprio input, then publishes arm targets to `/arm/command_joint_states`.

Example:

```bash
ros2 launch policy_inference_stereo_transformer policy_inference_stereo_transformer.launch.py \
  repo_root:=/home/chi/tg2_dexman_isaac \
  checkpoint_path:=/absolute/path/to/checkpoint.pth \
  left_topic:=/stereo/left/image_raw \
  right_topic:=/stereo/right/image_raw \
  proprio_topic:=/policy/proprio \
  joint_command_topic:=/arm/command_joint_states \
  rate:=20.0
```

Notes:

- `repo_root` must point at the repository root, not `deployment`.
- If `checkpoint_path` is empty, the node starts with random weights.
- By default this node publishes only the right-arm joint targets.

### Step 4: Launch The Feedback Bridge

Launch the bridge when you want `/arm/command_joint_states` translated to the robot command path and mirrored back into `/joint_states`.

```bash
ros2 launch feedback_control_bridge feedback_control_bridge.launch.py \
  control_domain:=right_full \
  command_topic:=/arm/cmd_pos \
  status_topic:=/arm/status \
  head_status_topic:=/head/status \
  leg_status_topic:=/leg/status \
  waist_status_topic:=/waist/status \
  input_joint_state_topic:=/arm/command_joint_states \
  input_hand_joint_state_topic:=/dummy_control/right_hand_joint_states \
  hand_command_interface:=service \
  hand_service_name:=/inspire_hand/set_angle_flexible/right_hand \
  publish_joint_states:=true \
  joint_state_topic:=/joint_states
```

Supported control domains:

- `right_arm`
- `left_arm`
- `right_full`
- `left_full`
- `upper_body`
- `full_body`

Use `right_full` if you need both right-arm and right-hand joints in `/joint_states`. If you launch `right_arm`, hand joints are not included, and scripts that expect 13 right-arm-plus-hand joints will reject that feedback.

### Step 5: Verify The Feedback Path

Before running motion scripts, verify the loop is complete:

```bash
ros2 topic list | grep -E '/arm/status|/arm/command_joint_states|/joint_states|/inspire_hand/state/right_hand'
ros2 topic info /arm/command_joint_states
ros2 topic echo /joint_states --once
```

`/joint_states` should include:

- `shoulder_pitch_r_joint`
- `shoulder_roll_r_joint`
- `shoulder_yaw_r_joint`
- `elbow_pitch_r_joint`
- `elbow_yaw_r_joint`
- `wrist_pitch_r_joint`
- `wrist_roll_r_joint`
- `little_joint_0` or `right_little_1_joint`
- `ring_joint_0` or `right_ring_1_joint`
- `middle_joint_0` or `right_middle_1_joint`
- `index_joint_0` or `right_index_1_joint`
- `thumb_joint_0` or `right_thumb_1_joint`
- `thumb_joint_1` or `right_thumb_2_joint`

## Offline Motion Tools

The scripts under `src/inference_offline/tests/` publish `sensor_msgs/JointState` commands to `/arm/command_joint_states` and rely on `/joint_states` feedback for initialization and safety checks.

Run them from `deployment` after sourcing the workspace.

### Move To Homing Or Init Pose

```bash
python3 src/inference_offline/tests/test_init_and_homing.py \
  --mode homing \
  --steps 120 \
  --rate 30 \
  --hold-sec 3
```

Or:

```bash
python3 src/inference_offline/tests/test_init_and_homing.py \
  --mode init \
  --steps 120 \
  --rate 30 \
  --hold-sec 3
```

Important:

- If `--start` is omitted, this script waits for `/joint_states`.
- It expects a complete right arm + right hand state.
- If `/joint_states` does not include the hand joints, startup aborts.

### Execute An Offline Trajectory

```bash
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30
```

Dry-run:

```bash
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30 \
  --dry-run
```

## Stereo Camera And Calibration

### Stereo Camera Publisher

```bash
ros2 launch stereo_camera stereo_ros_publisher.launch.py \
  width:=320 \
  height:=240 \
  flip:=both
```

You can also run it directly:

```bash
ros2 run stereo_camera stereo_ros_publisher -- --width 320 --height 240 --flip both
```

### AprilTag Calibration Flow

Quick path:

1. Start the stereo publisher.
2. Launch the AprilTag detector.
3. Run the calibration node.

Example detector launch:

```bash
ros2 run calibration april_tag_detector --ros-args \
  -p image_topic:=/stereo/left/image_raw \
  -p tag_family:=tag25h9 \
  -p tag_id:=-1 \
  -p tag_size:=0.10 \
  -p camera_frame:=stereo_left \
  -p intrinsics_npz:=/absolute/path/to/jetson_stereo_320_both.npz \
  -p intrinsics_camera:=left \
  -p input_reflip:=none \
  -p debug_view:=true
```

For the full camera calibration workflow and test utilities, see [src/calibration/README.md](/home/chi/tg2_dexman_isaac/tg2_lab/deployment/src/calibration/README.md).
