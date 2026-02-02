#!/usr/bin/env python3
"""
Thin test wrapper that exercises the capture_single_stream util with the
configured camera profiles.
"""

from __future__ import annotations

import argparse
import sys

from stereo_camera.utils.capture_single_stream import capture_single_stream
from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config


def main():
    parser = argparse.ArgumentParser(description="Test single-camera capture via util")
    parser.add_argument("--config", default="ov9732_L", help="Camera config name (YAML in cameras/config)")
    parser.add_argument("--device", default=None, help="Override device path")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--out", default="capture.avi", help="Output file path or prefix for segments")
    parser.add_argument("--segment", type=int, default=0, help="If >0, rotate output file every N seconds")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--log", default="capture.log", help="Log file path")
    parser.add_argument("--max-fails", type=int, default=5, help="Max consecutive frame failures before reconnect")
    parser.add_argument("--reconnect-wait", type=int, default=1, help="Initial reconnect wait (seconds) for exponential backoff")
    parser.add_argument("--stats-interval", type=int, default=10, help="Seconds between status logs")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Flip the frame if your camera is inverted",
    )
    parser.add_argument("--label", default=None, help="Label for logs/preview")
    args = parser.parse_args()

    cfg = load_camera_config(args.config)
    overrides = {
        "device": args.device or cfg.get("device"),
        "width": args.width or cfg.get("width"),
        "height": args.height or cfg.get("height"),
        "fps": args.fps or cfg.get("fps"),
    }
    cam = Ov9732Camera.from_config(args.config, overrides=overrides)

    return capture_single_stream(
        cam,
        out=args.out,
        segment=args.segment,
        show=args.show,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
        label=args.label or args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
