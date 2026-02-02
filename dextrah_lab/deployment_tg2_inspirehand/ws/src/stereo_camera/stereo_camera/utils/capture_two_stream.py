"""
Dual-camera capture pipeline using Ov9732Camera directly (no single-stream wrapper).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config


FLIP_MAP = {"none": None, "vertical": 0, "horizontal": 1, "both": -1}


class DualCameraReader:
    """
    Shared resilient reader for two cameras with reopen/backoff logic.
    This is the baseline used by both capture_two_stream and calibration flows.
    """

    def __init__(self, cams: Iterable[Ov9732Camera], *, flip: str = "vertical", max_fails: int = 5, reconnect_wait: float = 1.0):
        cams = list(cams)
        if len(cams) != 2:
            raise ValueError("Exactly two camera instances are required")
        if flip not in FLIP_MAP:
            raise ValueError("flip must be one of: none, vertical, horizontal, both")

        self.cams = cams
        self.flip = flip
        self.flip_code = FLIP_MAP[flip]
        self.max_fails = max_fails
        self.reconnect_wait = reconnect_wait

        self.consecutive_failures = [0, 0]
        self.backoff = [reconnect_wait, reconnect_wait]
        self.labels = ["L", "R"]

    def start(self):
        for cam in self.cams:
            cam.start()
        return self

    def stop(self):
        for cam in self.cams:
            try:
                cam.stop()
                cam.release()
            except Exception:
                pass

    def read_pair(self):
        frames = []
        for idx, cam in enumerate(self.cams):
            ok, frame = cam.read()
            if not ok or frame is None:
                self.consecutive_failures[idx] += 1
                if self.consecutive_failures[idx] >= self.max_fails:
                    logging.warning("[%s] Reopening after %d failures", self.labels[idx], self.consecutive_failures[idx])
                    cam.release()
                    time.sleep(self.backoff[idx])
                    self.backoff[idx] = min(self.backoff[idx] * 2, 30)
                    try:
                        cam.start()
                        self.consecutive_failures[idx] = 0
                        self.backoff[idx] = self.reconnect_wait
                    except Exception as exc:  # noqa: BLE001
                        logging.error("[%s] Reopen failed: %s", self.labels[idx], exc)
                return None

            self.consecutive_failures[idx] = 0
            if self.flip_code is not None:
                frame = cv2.flip(frame, self.flip_code)
            frames.append((idx, frame))

        frames_sorted = [f for _, f in sorted(frames, key=lambda x: x[0])]
        return frames_sorted

    def log_stats(self):
        for idx, cam in enumerate(self.cams):
            try:
                cam.log_stats(interval=0, label=self.labels[idx])
            except Exception:
                pass


def capture_two_stream(
    cams: Iterable[Ov9732Camera],
    *,
    out_dir: Path | str = ".",
    basename: str = "capture",
    segment: int = 0,
    save: bool = False,
    show: bool = True,
    log: str = "capture_two.log",
    max_fails: int = 5,
    reconnect_wait: int = 1,
    stats_interval: int = 10,
    flip: str = "vertical",
    preview_scale: float = 0.5,
):
    cams = list(cams)
    reader = DualCameraReader(cams, flip=flip, max_fails=max_fails, reconnect_wait=reconnect_wait)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging per call (overrides prior basicConfig only once)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # writers and segment timers
    writers: List[cv2.VideoWriter | None] = [None, None]
    segment_start = [time.monotonic(), time.monotonic()]
    labels = ["L", "R"]
    reader.labels = labels  # preserve existing labels

    def open_writer(idx: int, seq: int) -> cv2.VideoWriter | None:
        cam = cams[idx]
        path = out_dir / f"{basename}_cam{idx}.avi"
        if segment and segment > 0 and seq > 0:
            suffix = time.strftime("%Y%m%d_%H%M%S")
            path = path.with_name(f"{path.stem}_{suffix}.avi")
        writer = cam.save_video(path, codec="MJPG", fps=cam.fps)
        if not writer.isOpened():
            logging.error("[%s] VideoWriter failed to open %s", labels[idx], path)
            return None
        logging.info("[%s] Writing to %s", labels[idx], path)
        return writer

    # Start cameras
    try:
        reader.start()
    except Exception as exc:
        logging.error("Could not open cameras: %s", exc)
        return 1
    if save:
        for idx in range(len(cams)):
            writers[idx] = open_writer(idx, 0)
            if writers[idx] is None:
                return 1

    seq = [0, 0]
    last_stats = time.monotonic()
    stop = False

    while not stop:
        now = time.monotonic()
        frames_sorted = reader.read_pair()
        if frames_sorted is None:
            continue

        frame_l, frame_r = frames_sorted

        for idx, frame in enumerate((frame_l, frame_r)):
            if save and writers[idx]:
                writers[idx].write(frame)

            if save and segment and (now - segment_start[idx]) >= segment:
                logging.info("[%s] Rotating segment after %.1fs", labels[idx], now - segment_start[idx])
                if writers[idx]:
                    writers[idx].release()
                seq[idx] += 1
                writers[idx] = open_writer(idx, seq[idx])
                segment_start[idx] = now

        # preview in main thread
        if show:
            frames_for_preview = (frame_l, frame_r)
            if preview_scale != 1.0:
                frames_scaled = [
                    cv2.resize(f, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
                    for f in frames_for_preview
                ]
            else:
                frames_scaled = frames_for_preview
            try:
                composite = cv2.hconcat(frames_scaled)
                cv2.imshow("preview L | R", composite)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
            except Exception as exc:
                logging.warning("Preview failed: %s", exc)

        if now - last_stats > stats_interval:
            reader.log_stats()
            last_stats = now

    # cleanup
    for idx in range(len(cams)):
        if writers[idx]:
            try:
                writers[idx].release()
            except Exception:
                pass
    reader.stop()
    if show:
        cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Dual camera capture (direct pipeline)")
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
    parser.add_argument("--save", action="store_true", help="Enable saving video files (off by default)")
    parser.add_argument("--headless", action="store_true", help="Disable preview window")
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
    parser.add_argument("--preview-scale", type=float, default=0.5, help="Scale preview window")
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
        save=args.save,
        show=not args.headless,
        log=args.log,
        max_fails=args.max_fails,
        reconnect_wait=args.reconnect_wait,
        stats_interval=args.stats_interval,
        flip=args.flip,
        preview_scale=args.preview_scale,
    )


if __name__ == "__main__":
    sys.exit(main())
