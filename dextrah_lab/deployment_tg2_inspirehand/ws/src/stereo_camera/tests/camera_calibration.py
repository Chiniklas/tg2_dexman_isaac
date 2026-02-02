"""
Runnable helper to capture a chessboard sequence and run stereo calibration.

Usage (from repo root):
    python -m tests.camera_calibration \
        --left-config ov9732_L --right-config ov9732_R \
        --square-size-mm 25 --board-cols 8 --board-rows 6 \
        --frames 50 --save-dir calibration --camera-name jetson_stereo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stereo_camera.StereoCameraCalibration.camera_calibration import stereo_calibrate_camera
from stereo_camera.cameras.ov9732_camera import Ov9732Camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture chessboard pairs and calibrate stereo cameras.")
    parser.add_argument("--left-config", default="ov9732_L", help="YAML name or path for the left camera.")
    parser.add_argument("--right-config", default="ov9732_R", help="YAML name or path for the right camera.")
    parser.add_argument("--camera-name", default="jetson_stereo", help="Prefix used for saved .npz files.")
    parser.add_argument("--save-dir", default="calibration", help="Directory to store calibration outputs.")
    parser.add_argument("--square-size-mm", type=float, default=25.0, help="Checkerboard square edge length in millimetres.")
    parser.add_argument("--board-cols", type=int, default=8, help="Number of inner corners along the width.")
    parser.add_argument("--board-rows", type=int, default=6, help="Number of inner corners along the height.")
    parser.add_argument("--frames", type=int, default=50, help="Number of valid chessboard pairs to capture.")
    parser.add_argument("--display-scale", type=float, default=0.6, help="Preview downscale factor for side-by-side view.")
    parser.add_argument(
        "--free-intrinsic",
        action="store_true",
        help="Do not fix intrinsics during stereoCalibrate (CALIB_FIX_INTRINSIC off).",
    )
    parser.add_argument("--flip", choices=["none", "vertical", "horizontal", "both"], default="vertical", help="Optional flip applied to both streams.")
    parser.add_argument("--max-fails", type=int, default=5, help="Consecutive read failures before attempting reconnect.")
    parser.add_argument("--reconnect-wait", type=float, default=1.0, help="Initial reconnect backoff (seconds).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def build_camera(cfg: str) -> Ov9732Camera:
    # The wrapper loads YAML from stereo_camera/cameras/config by default.
    return Ov9732Camera.from_config(cfg)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    board_size = (args.board_cols, args.board_rows)
    save_dir = Path(args.save_dir)

    logging.info("Using board_size=%s (inner corners), square_size=%.3f mm", board_size, args.square_size_mm)
    logging.info("Capturing %d pairs. Press 'c' to capture, 'x' to abort.", args.frames)

    left = build_camera(args.left_config)
    right = build_camera(args.right_config)

    try:
        stereo_path = stereo_calibrate_camera(
            camera_left=left,
            camera_right=right,
            camera_name=args.camera_name,
            square_size_mm=args.square_size_mm,
            board_size=board_size,
            frames_to_capture=args.frames,
            display_scale=args.display_scale,
            save_dir=save_dir,
            fix_intrinsic=not args.free_intrinsic,
            max_fails=args.max_fails,
            reconnect_wait=args.reconnect_wait,
            flip=args.flip,
        )
        logging.info("Calibration complete. Saved stereo params to %s", stereo_path)
        logging.info("Individual camera files: %s / %s", save_dir / f"{args.camera_name}c1.npz", save_dir / f"{args.camera_name}c2.npz")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.exception("Calibration failed: %s", exc)
        return 1
    finally:
        for cam in (left, right):
            try:
                cam.stop()
                cam.release()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
