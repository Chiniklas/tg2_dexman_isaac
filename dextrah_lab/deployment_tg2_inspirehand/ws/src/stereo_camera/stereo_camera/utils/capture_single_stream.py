"""
Single-camera capture helper.
Provide an already-configured Ov9732Camera instance, and this will record,
optionally preview, and rotate video segments.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Camera wrapper and shared helpers
from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config, resolve_log_path


def capture_single_stream(
    cam: Ov9732Camera,
    *,
    out: Path | str = "capture.avi",
    segment: int = 0,
    save: bool = False,
    show: bool = True,
    log: str = "capture.log",
    max_fails: int = 5,
    reconnect_wait: int = 1,
    stats_interval: int = 10,
    flip: str = "vertical",
    stop_event: Optional[threading.Event] = None,
    label: str = "cam",
):
    """
    Capture a single camera stream. Returns exit code (0 on success).
    The camera must be pre-configured.
    """
    out_path = Path(out)

    log_path = resolve_log_path(Path(log).name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )

    stop_event = stop_event or threading.Event()

    def sig_handler(signum, frame):
        logging.info("Signal %s received, stopping...", signum)
        stop_event.set()

    # Only the main thread can set signal handlers; guard for threaded use.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
    writer = None
    seq = 0

    consecutive_failures = 0
    max_failures_before_reopen = max_fails
    reconnect_wait_base = reconnect_wait

    def open_writer_for_seq(seq_idx):
        if segment and segment > 0:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_path.with_name(f"{out_path.stem}_{suffix}.avi")
        else:
            path = out_path
        writer = cam.save_video(path, codec="MJPG", fps=cam.fps)
        if not writer.isOpened():
            logging.error("VideoWriter failed to open %s", path)
            return path, None
        return path, writer

    backoff = reconnect_wait_base
    try:
        cam.start()
    except Exception as exc:
        logging.warning("Initial camera open failed (%s); will retry in %ds", exc, backoff)
        time.sleep(backoff)
        try:
            cam.start()
        except Exception as exc2:
            logging.error("Could not open camera; exiting (%s)", exc2)
            return 1

    path, writer = (None, None)
    if save:
        path, writer = open_writer_for_seq(seq)
        if writer is None:
            logging.error("Failed to open writer; exiting.")
            return 1

    segment_start = time.monotonic()

    try:
        while not stop_event.is_set():
            ret, frame = cam.read()
            now = time.monotonic()

            if not ret or frame is None:
                consecutive_failures += 1
                logging.warning("Frame read failed (consecutive fails=%d)", consecutive_failures)
                if consecutive_failures >= max_failures_before_reopen:
                    logging.warning("Too many consecutive frame failures, will reopen camera")
                    cam.release()
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    try:
                        cam.start()
                        consecutive_failures = 0
                        backoff = reconnect_wait_base
                    except Exception as exc:
                        logging.error("Reopen failed: %s", exc)
                    continue
                time.sleep(0.01)
                continue

            consecutive_failures = 0

            if flip != "none":
                flip_map = {"vertical": 0, "horizontal": 1, "both": -1}
                frame = cv2.flip(frame, flip_map[flip])

            if save and writer:
                writer.write(frame)

            if save and segment and (now - segment_start) >= segment:
                logging.info("Rotating segment file after %.1fs", now - segment_start)
                if writer:
                    writer.release()
                seq += 1
                path, writer = open_writer_for_seq(seq)
                if writer is None:
                    logging.error("Failed to open new segment writer; stopping")
                    break
                segment_start = now

            if show:
                cv2.imshow(f"capture {label}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("'q' pressed, exiting preview")
                    stop_event.set()
                    break

            cam.log_stats(interval=stats_interval, label=label)
    finally:
        logging.info("Stopping capture")
        if writer:
            writer.release()
        cam.release()
        if show:
            cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Single camera capture")
    parser.add_argument("--config", default="ov9732_L", help="Camera config name (YAML in cameras/config)")
    parser.add_argument("--device", default=None, help="Override device path")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--out", default="capture.avi", help="Output file path or prefix for segments")
    parser.add_argument("--segment", type=int, default=0, help="If >0, rotate output file every N seconds")
    parser.add_argument("--save", action="store_true", help="Enable saving video files (off by default)")
    parser.add_argument("--headless", action="store_true", help="Disable preview window")
    parser.add_argument("--log", default="capture.log", help="Log file path")
    parser.add_argument("--max-fails", type=int, default=5, help="Max consecutive frame failures before reconnect")
    parser.add_argument("--reconnect-wait", type=int, default=1, help="Initial reconnect wait (seconds) for exponential backoff")
    parser.add_argument("--stats-interval", type=int, default=10, help="Seconds between status logs")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="Flip the frame if your camera is inverted (vertical=upside down, horizontal=mirror, both=180° rotate)",
    )
    parser.add_argument("--label", default=None, help="Label for logs and preview title")
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
        save=args.save,
        show=not args.headless,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
        label=args.label or args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
