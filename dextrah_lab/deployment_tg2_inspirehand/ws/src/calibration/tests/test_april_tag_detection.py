#!/usr/bin/env python3
"""Integration test: launch stereo camera + AprilTag detector and wait for detections.

This script intentionally launches both components:
1. `stereo_ros_publisher.py`
2. `april_tag_detector.py`

Then it monitors detection topics and exits success when the requested tag appears.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray


class StreamInfoWatcher:
    """Track incoming image dimensions for debug/status output."""

    def __init__(self, image_topic: str) -> None:
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.msg_count = 0
        self._sub = rospy.Subscriber(image_topic, Image, self._cb, queue_size=1)

    def _cb(self, msg: Image) -> None:
        self.width = int(msg.width)
        self.height = int(msg.height)
        self.msg_count += 1


class TagDetectionWatcher:
    """Watch /tag_detection_ids and report when desired tag appears."""

    def __init__(self, target_tag_id: int) -> None:
        self.target_tag_id = target_tag_id
        self.detected = False
        self.ever_detected = False
        self.last_ids: list[int] = []
        self._sub = rospy.Subscriber("/tag_detection_ids", Int32MultiArray, self._cb, queue_size=10)

    def _cb(self, msg: Int32MultiArray) -> None:
        ids = [int(v) for v in msg.data]
        self.last_ids = ids
        if self.target_tag_id < 0:
            self.detected = len(ids) > 0
        else:
            self.detected = self.target_tag_id in ids
        if self.detected:
            self.ever_detected = True


def _build_env(ws_src: Path) -> dict:
    env = os.environ.copy()
    extra = str(ws_src)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra
    return env


def _start_process(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env)


def _find_ws_src(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        direct = candidate / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
        if direct.exists():
            return candidate
        via_src = candidate / "src" / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
        if via_src.exists():
            return candidate / "src"
    raise FileNotFoundError(
        f"Could not locate stereo_ros_publisher.py by searching from {start}. "
        "Expected path: <ws>/src/stereo_camera/scripts/stereo_ros_publisher.py"
    )


def _find_calibration_dir(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        direct = candidate / "calibration"
        if (direct / "april_tag_detector.py").exists():
            return direct
        via_src = candidate / "src" / "calibration"
        if (via_src / "april_tag_detector.py").exists():
            return via_src
    raise FileNotFoundError(
        f"Could not locate april_tag_detector.py by searching from {start}. "
        "Expected path: <ws>/src/calibration/april_tag_detector.py"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch stereo camera + AprilTag detector and wait for detection."
    )

    # Stereo camera launch settings
    parser.add_argument("--left-config", default="ov9732_L")
    parser.add_argument("--right-config", default="ov9732_R")
    parser.add_argument("--left-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", default="/stereo/right/image_raw")
    parser.add_argument("--left-frame", default="stereo_left")
    parser.add_argument("--right-frame", default="stereo_right")
    parser.add_argument("--flip", default="both", choices=["none", "vertical", "horizontal", "both"])
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rate", type=float, default=0.0, help="Optional publish rate limit (Hz). 0 disables.")

    # Detector settings
    parser.add_argument("--tag-family", default="tag25h9")
    parser.add_argument("--tag-id", type=int, default=-1)
    parser.add_argument("--tag-size", type=float, default=0.10)
    parser.add_argument("--camera-frame", default="stereo_left")
    parser.add_argument("--intrinsics-npz", default="")
    parser.add_argument("--intrinsics-camera", default="left", choices=["left", "right", "auto"])
    parser.add_argument(
        "--input-reflip",
        default="none",
        choices=["none", "vertical", "horizontal", "both"],
        help="Undo upstream image flip before detection.",
    )
    parser.add_argument(
        "--debug-display-flip",
        default="auto",
        choices=["auto", "none", "vertical", "horizontal", "both"],
        help=(
            "Flip detector debug window display only. "
            "'auto' matches --input-reflip so view is usually upright."
        ),
    )
    parser.add_argument("--debug-view", action="store_true", help="Enable detector OpenCV debug window.")

    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Optional max runtime in seconds. <=0 means run until Ctrl+C.",
    )
    parser.add_argument("--startup-wait", type=float, default=2.0, help="Seconds to wait after launching nodes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tests_dir = Path(__file__).resolve().parent
    ws_src = _find_ws_src(tests_dir)
    calibration_dir = _find_calibration_dir(tests_dir)
    stereo_script = ws_src / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
    detector_script = calibration_dir / "april_tag_detector.py"
    default_intrinsics_path = (
        ws_src
        / "stereo_camera"
        / "tests"
        / "calibration_320_both"
        / "jetson_stereo_320_both.npz"
    )

    if not stereo_script.exists():
        raise FileNotFoundError(f"Stereo publisher script not found: {stereo_script}")
    if not detector_script.exists():
        raise FileNotFoundError(f"AprilTag detector script not found: {detector_script}")

    intrinsics_path = Path(args.intrinsics_npz) if args.intrinsics_npz else default_intrinsics_path
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {intrinsics_path}")

    env = _build_env(ws_src)
    procs: list[subprocess.Popen] = []

    try:
        rospy.init_node("test_april_tag_detection", anonymous=True)

        stereo_cmd = [
            sys.executable,
            str(stereo_script),
            "--left-config",
            args.left_config,
            "--right-config",
            args.right_config,
            "--left-topic",
            args.left_topic,
            "--right-topic",
            args.right_topic,
            "--left-frame",
            args.left_frame,
            "--right-frame",
            args.right_frame,
            "--flip",
            args.flip,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
        ]
        if args.rate > 0.0:
            stereo_cmd.extend(["--rate", str(args.rate)])
        rospy.loginfo("Launching stereo publisher: %s", " ".join(stereo_cmd))
        procs.append(_start_process(stereo_cmd, env))

        debug_display_flip = args.debug_display_flip
        if debug_display_flip == "auto":
            debug_display_flip = args.input_reflip

        detector_cmd = [
            sys.executable,
            str(detector_script),
            f"_image_topic:={args.left_topic}",
            f"_tag_family:={args.tag_family}",
            f"_tag_id:={args.tag_id}",
            f"_tag_size:={args.tag_size}",
            f"_camera_frame:={args.camera_frame}",
            f"_intrinsics_npz:={intrinsics_path}",
            f"_intrinsics_camera:={args.intrinsics_camera}",
            f"_input_reflip:={args.input_reflip}",
            f"_debug_view:={str(bool(args.debug_view)).lower()}",
            f"_debug_display_flip:={debug_display_flip}",
        ]
        rospy.loginfo("Launching detector: %s", " ".join(detector_cmd))
        procs.append(_start_process(detector_cmd, env))

        time.sleep(max(args.startup_wait, 0.0))

        stream = StreamInfoWatcher(args.left_topic)
        watcher = TagDetectionWatcher(args.tag_id)

        start = time.time()
        last_log = 0.0
        last_detect_log = 0.0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            now = time.time()
            if watcher.detected and (now - last_detect_log >= 1.0):
                last_detect_log = now
                rospy.loginfo("Detected tag IDs on /tag_detection_ids: %s", watcher.last_ids)

            if now - last_log >= 2.0:
                last_log = now
                dim = "unknown"
                if stream.width is not None and stream.height is not None:
                    dim = f"{stream.width}x{stream.height}"
                rospy.loginfo(
                    "Running... image=%s msgs=%d ids=%s detected_once=%s",
                    dim,
                    stream.msg_count,
                    watcher.last_ids,
                    watcher.ever_detected,
                )

            if args.timeout > 0.0 and (now - start) > args.timeout:
                if watcher.ever_detected:
                    rospy.loginfo("Reached timeout after successful detections.")
                    return 0
                rospy.logerr("Reached timeout without detections on /tag_detection_ids.")
                return 1

            rate.sleep()

        return 0 if watcher.ever_detected else 1
    finally:
        for proc in procs:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        for proc in procs:
            try:
                proc.wait(timeout=3.0)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
