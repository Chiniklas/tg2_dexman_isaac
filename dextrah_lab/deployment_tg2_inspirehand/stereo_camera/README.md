DIY stereo camera test environment (dual OV9732 modules on a 3D-printed holder). More build notes and assets coming soon.

## Setup
- Install Miniconda/Anaconda if you don't already have it.
- Create and activate an isolated environment:
  ```bash
  conda create -n stereo_camera python=3.10 -y
  conda activate stereo_camera
  ```
- Install Python requirements:
  ```bash
  pip install -r requirements.txt
  ```

## Single-camera capture
- Config lives in `stereo_camera/cameras/config/ov9732_L.yaml` (or `_R.yaml`).
- Quick run with defaults (prefer the test wrapper):
  ```bash
  python tests/capture_single_stream.py --show
  ```
- Override device/resolution/fps on the fly:
  ```bash
  python tests/capture_single_stream.py \
    --config ov9732_L \
    --device /dev/video4 \
    --width 1280 --height 720 --fps 30 \
    --out left.avi --show --save
  ```
- Behavior: preview is on by default; add `--headless` to disable. Recording happens only when `--save` is set; `--segment N` rotates files; press `q` in the preview to stop.

## Two-camera capture
- Configs: `ov9732_L.yaml` and `ov9732_R.yaml`.
- Quick run with defaults (prefer the test wrapper):
  ```bash
  python tests/capture_two_stream.py --show
  ```
- Override devices/resolution/fps:
  ```bash
  python tests/capture_two_stream.py \
    --config-left ov9732_L --config-right ov9732_R \
    --device-left /dev/video4 --device-right /dev/video2 \
    --width 1280 --height 720 --fps 30 \
    --out-dir recordings --basename stereo --show --save
  ```
- Behavior: preview is on by default; add `--headless` to disable. Recording happens only when `--save` is set; if saving, writes `basename_cam0.avi` and `basename_cam1.avi`; `--segment N` rotates files; `--show` opens a side-by-side preview.

## Stereo calibration
- Start the interactive calibration capture (uses the same resilient reader as two-stream):
  ```bash
  python tests/camera_calibration.py \
    --left-config ov9732_L --right-config ov9732_R \
    --square-size-mm 25 --board-cols 8 --board-rows 6 \
    --frames 50 --flip vertical
  ```
- Press `c` to capture synchronized pairs; aim for 40–50 diverse poses; press `x` to abort.
- Outputs land in `tests/calibration/` by default: `jetson_stereo.npz` (stereo), `jetson_stereoc1.npz`, `jetson_stereoc2.npz`.
- Use inner-corner counts for `--board-cols/rows` (e.g., an 8×6 inner-corner A4 board with 25 mm squares → `--board-cols 8 --board-rows 6 --square-size-mm 25`).

## Depth sanity check
- Place a flat target at a known distance (e.g., 0.50 m) centered in view.
- Run the plausibility test:
  ```bash
  python tests/depth_plausibility.py \
    --calib tests/calibration/jetson_stereo.npz \
    --left-config ov9732_L --right-config ov9732_R \
    --expected 0.50 --tolerance 0.25 --frames 30 --show --show-rgb
  ```
- Shows disparity (and optional RGB on top). Reports median depth in the center ROI; exits non-zero if relative error exceeds tolerance.
- If depth is far off: recheck printed square size, baseline rigidity, or recalibrate.

## Live stereo depth preview
- Visualize rectified RGB (top) and real-time disparity (bottom) using your saved calibration:
  ```bash
  python tests/capture_stereo_stream.py \
    --calib tests/calibration/jetson_stereo.npz \
    --left-config ov9732_L --right-config ov9732_R \
    --flip vertical --preview-scale 0.7 \
    --num-disp 160 --block-size 7 --uniqueness 15 \
    --speckle-window 200 --speckle-range 4 --median --clahe
  ```
- Tune SGBM with `--num-disp`, `--block-size`, `--uniqueness`, `--speckle-window`, `--speckle-range`; add `--median` and `--clahe` to denoise/boost contrast. Press `q` to quit.

## Notes
- Defaults target MJPG @ 1280x720 @ 30 fps.
- If a camera only supports YUYV/lower FPS, change the `fourcc` to `YUYV` and reduce `--fps`.
