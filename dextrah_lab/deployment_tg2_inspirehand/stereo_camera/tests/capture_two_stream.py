#!/usr/bin/env python3
"""
Thin test wrapper that exercises the dual-camera util.
"""

from __future__ import annotations

import argparse
import sys

from stereo_camera.utils.capture_two_stream import capture_two_stream
from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config


def main():
    parser = argparse.ArgumentParser(description="Test dual-camera capture via util")
    parser.add_argument("--config-left", default="ov9732_L", help="Left camera config name (YAML)")
    parser.add_argument("--config-right", default="ov9732_R", help="Right camera config name (YAML)")
    parser.add_argument("--device-left", default=None, help="Override left device")
    parser.add_argument("--device-right", default=None, help="Override right device")
    parser.add_argument("--width", type=int, default=None, help="Override width (both cams)")
    parser.add_argument("--height", type=int, default=None, help="Override height (both cams)")
    parser.add_argument("--fps", type=int, default=None, help="Override fps (both cams)")
    parser.add_argument("--out-dir", default=".", help="Directory for output files")
    parser.add_argument("--basename", default="capture", help="Base name for output files")
    parser.add_argument("--segment", type=int, default=0, help="If >0, rotate output file every N seconds")
    parser.add_argument("--show", action="store_true", help="Show preview windows")
    parser.add_argument("--log", default="capture_two.log", help="Log file path")
    parser.add_argument("--max-fails", type=int, default=5, help="Max consecutive frame failures before reconnect")
    parser.add_argument("--reconnect-wait", type=int, default=1, help="Initial reconnect wait (seconds) for exponential backoff")
    parser.add_argument("--stats-interval", type=int, default=10, help="Seconds between status logs")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Flip frames if cameras are inverted",
    )
    args = parser.parse_args()

    cfgL = load_camera_config(args.config_left)
    cfgR = load_camera_config(args.config_right)

    overridesL = {
        "device": args.device_left or cfgL.get("device"),
        "width": args.width or cfgL.get("width"),
        "height": args.height or cfgL.get("height"),
        "fps": args.fps or cfgL.get("fps"),
    }
    overridesR = {
        "device": args.device_right or cfgR.get("device"),
        "width": args.width or cfgR.get("width"),
        "height": args.height or cfgR.get("height"),
        "fps": args.fps or cfgR.get("fps"),
    }

    cam_left = Ov9732Camera.from_config(args.config_left, overrides=overridesL)
    cam_right = Ov9732Camera.from_config(args.config_right, overrides=overridesR)

    return capture_two_stream(
        [cam_left, cam_right],
        out_dir=args.out_dir,
        basename=args.basename,
        segment=args.segment,
        show=args.show,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
    )


if __name__ == "__main__":
    sys.exit(main())
