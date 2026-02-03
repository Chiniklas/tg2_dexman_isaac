# Calibration Tests (ROS1)

This package contains lightweight test utilities for the TG2 Inspirehand calibration pipeline. These scripts are intended to be run directly with `python3` after sourcing your ROS1 environment and launching `roscore`.

## Prerequisites

- ROS1 Noetic environment sourced
- `roscore` running in a separate terminal
- Cameras available (for stereo tests)
- `opencv-contrib-python` installed if you need AprilTag detection (`cv2.aruco`)

## Scripts

- `tests/test_stereo_camera_node.py`
  - Launches the stereo publisher (optional) and displays a side-by-side preview.
- `tests/test_april_tag_detection.py`
  - Launches the stereo publisher (optional), the AprilTag detector, publishes a synthetic `CameraInfo`, and waits for a tag TF.
- `tests/test_execute_targets.py`
  - Placeholder for future execution tests (currently empty).

## Usage

From inside the container or a sourced ROS1 shell:

```bash
cd /tiangong_infra_ws/ws/src/calibration_tests/tests
python3 test_stereo_camera_node.py \
  --left-topic /stereo/left/image_raw \
  --right-topic /stereo/right/image_raw
```

For AprilTag detection:

```bash
cd /tiangong_infra_ws/ws/src/calibration_tests/tests
python3 test_april_tag_detection.py \
  --tag-family tag25h9 --tag-id 0 --tag-size 0.04
```

If the stereo publisher is already running, add `--no-stereo`. If you already publish real `CameraInfo`, add `--no-camera-info`.
