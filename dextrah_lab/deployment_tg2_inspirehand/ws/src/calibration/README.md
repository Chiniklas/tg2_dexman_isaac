# Calibration Package (ROS1)

This package contains camera-to-robot calibration scripts and test utilities for TG2 Inspirehand deployment.

## Main scripts

- `camera_calibration.py`
  - Runs hand-eye style calibration by moving through pose targets, logging `/tf` tag poses + robot joint states, and solving for `robot_T_camera`.
- `april_tag_detector.py`
  - Detects AprilTags from an image topic and publishes tag transforms on `/tf`.
  - Uses intrinsics loaded strictly from `.npz` (no `/camera_info` fallback).
- `tg2_inspirehand_random_targets.py`
  - Pose interpolation helpers used by calibration.

## Test utilities

- `tests/test_stereo_camera_node.py`
  - Visual side-by-side stereo stream check.
- `tests/test_april_tag_detection.py`
  - End-to-end AprilTag detection check (`Image` + `CameraInfo` + `/tf`).
- `tests/test_execute_targets.py`
  - Placeholder (currently empty).

## Prerequisites

- ROS1 Noetic sourced
- `roscore` running
- Stereo camera topics available (or launch via test script)
- `opencv-contrib-python` for AprilTag support (`cv2.aruco`)
- Calibration tag visible to camera (for detector/calibration)

## Quick calibration flow

1. Start stereo camera publisher:

```bash
roslaunch stereo_camera stereo_ros_publisher.launch \
  width:=320 \
  height:=240 \
  flip:=both
```

2. Start AprilTag detector (example):

```bash
rosrun calibration april_tag_detector.py \
  _image_topic:=/stereo/left/image_raw \
  _tag_family:=tag25h9 \
  _tag_id:=-1 \
  _tag_size:=0.10 \
  _camera_frame:=stereo_left \
  _intrinsics_npz:=/tiangong_infra_ws/ws/src/stereo_camera/tests/calibration_320_both/jetson_stereo_320_both.npz \
  _intrinsics_camera:=left \
  _input_reflip:=none \
  _debug_view:=true \
  _debug_display_flip:=none
```

Notes:
- `_intrinsics_npz` is required.
- If loading `_intrinsics_npz` fails, the detector raises an error and exits (no `/camera_info` fallback).
- Detector default family is `tag25h9`.
- Detector default `tag_id` is `-1` (detect all tag IDs).
- `input_reflip` is applied before detection/pose. Use it only to undo upstream image flips.
- `debug_display_flip` only affects the debug window rendering; detection/pose uses the original image.
- Use stream settings matching calibration intrinsics. For `jetson_stereo_320_both.npz`, validate first with `flip:=both` and `320x240`.
- Resolution must match the calibration file. Example: if `jetson_stereo_320_both.npz` was calibrated at `320x240` but the stream runs at `1280x720`, pose scale can be wrong.
- Quick check:
  - `rostopic echo -n 1 /stereo/left/image_raw | grep -E "width|height"`
- If mismatched:
  - relaunch stereo publisher with matching size (`width:=320 height:=240 flip:=both`), or
  - recalibrate and use an intrinsics file generated for the current stream resolution.

3. Run calibration:

```bash
rosrun calibration camera_calibration.py \
  --camera left \
  --joint-state-topic /tg2/joint_states \
  --pose-command-topic /tg2_inspirehand_fabric/pose_commands \
  --tag-frame-id tag25h9:0 \
  --home-pose x y z yaw pitch roll \
  --target-pose x y z yaw pitch roll
```

Output:

- `robot_cam_<camera>_calibration.txt` in the current working directory.

## Running tests directly

```bash
cd /tiangong_infra_ws/ws/src/calibration/tests
python3 test_stereo_camera_node.py --left-topic /stereo/left/image_raw --right-topic /stereo/right/image_raw

python3 test_april_tag_detection.py --tag-family tag25h9 --tag-id -1 --tag-size 0.10

python3 test_pure_apriltag_detection.py --tag-family tag25h9 --tag-id -1 --tag-size 0.10 \
  --intrinsics-npz /tiangong_infra_ws/ws/src/stereo_camera/tests/calibration_320_both/jetson_stereo_320_both.npz \
  --intrinsics-camera left
```

Use `--no-stereo` if camera publisher is already running. Use `--no-camera-info` if real `CameraInfo` is already available.

## Pure Test (Direct Camera Access)

`test_pure_apriltag_detection.py` opens the camera devices directly (`/dev/video*`), so it cannot run at the same time as `stereo_ros_publisher`.

Stop ROS stereo publisher first:

```bash
pkill -f stereo_ros_publisher.py
```

Then run:

```bash
cd /tiangong_infra_ws/ws/src/calibration/tests
python3 test_pure_apriltag_detection.py \
  --left-config ov9732_L_320 \
  --width 320 \
  --height 240 \
  --tag-family tag25h9 \
  --tag-id -1 \
  --tag-size 0.10 \
  --flip both \
  --intrinsics-npz /tiangong_infra_ws/ws/src/stereo_camera/tests/calibration_320_both/jetson_stereo_320_both.npz \
  --intrinsics-camera left
```

If camera open fails, check which process is holding the devices:

```bash
lsof /dev/video0 /dev/video2
```
