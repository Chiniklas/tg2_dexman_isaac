#!/usr/bin/env python3
"""
Launch the stereo camera stream and verify AprilTag detection via TF.

This helper script:
1. Optionally launches the stereo ROS publisher (left/right image streams).
2. Launches the AprilTag detector node.
3. Publishes a basic CameraInfo (derived from the image size + HFOV).
4. Listens on /tf for the requested tag frame and exits when detected.

Usage (from a sourced ROS1 environment with `roscore` running):
    python test_april_tag_detection.py \
      --image-topic /stereo/left/image_raw \
      --camera-info-topic /stereo/left/camera_info \
      --tag-family tag25h9 --tag-id 0 --tag-size 0.04
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import rospy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage


class CameraInfoPublisher:
    def __init__(self, image_topic: str, camera_info_topic: str, camera_frame: str, hfov_deg: float) -> None:
        self._camera_info_topic = camera_info_topic
        self._camera_frame = camera_frame
        self._hfov_deg = hfov_deg
        self._pub = rospy.Publisher(camera_info_topic, CameraInfo, queue_size=1, latch=True)
        self._sub = rospy.Subscriber(image_topic, Image, self._image_cb, queue_size=1)
        self._published = False

    def _image_cb(self, msg: Image) -> None:
        if self._published:
            return
        width = int(msg.width)
        height = int(msg.height)
        if width <= 0 or height <= 0:
            return

        hfov_rad = math.radians(self._hfov_deg)
        fx = width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        cam_info = CameraInfo()
        cam_info.header.stamp = msg.header.stamp
        cam_info.header.frame_id = self._camera_frame or msg.header.frame_id
        cam_info.width = width
        cam_info.height = height
        cam_info.distortion_model = "plumb_bob"
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        cam_info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._pub.publish(cam_info)
        self._published = True
        rospy.loginfo("Published synthetic CameraInfo to %s (%.0fx%.0f, hfov=%.1f deg)", self._camera_info_topic, width, height, self._hfov_deg)


class TagDetectionWatcher:
    def __init__(self, tag_family: str, tag_id: int) -> None:
        self._tag_family = tag_family
        self._tag_id = tag_id
        self.detected = False
        self.last_child_frame: Optional[str] = None
        self._sub = rospy.Subscriber("/tf", TFMessage, self._tf_cb, queue_size=10)

    def _tf_cb(self, msg: TFMessage) -> None:
        if self.detected:
            return
        for transform in msg.transforms:
            child = transform.child_frame_id
            if self._tag_id >= 0:
                expected = f"{self._tag_family}:{self._tag_id}"
                if child != expected:
                    continue
            else:
                if not child.startswith(f"{self._tag_family}:"):
                    continue
            self.last_child_frame = child
            self.detected = True
            rospy.loginfo("Detected tag frame on /tf: %s", child)
            return


def _build_env(ws_src: Path) -> dict:
    env = os.environ.copy()
    extra = str(ws_src)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = extra + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = extra
    return env


def _start_process(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _find_ws_src(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        stereo_script = candidate / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
        if stereo_script.exists():
            return candidate
        stereo_script = candidate / "src" / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
        if stereo_script.exists():
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
        src = candidate / "src" / "calibration"
        if (src / "april_tag_detector.py").exists():
            return src
    raise FileNotFoundError(
        f"Could not locate april_tag_detector.py by searching from {start}. "
        "Expected path: <ws>/src/calibration/april_tag_detector.py"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch stereo stream + AprilTag detector and wait for /tf tag.")
    parser.add_argument("--image-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-image-topic", default="/stereo/right/image_raw")
    parser.add_argument("--camera-info-topic", default="/stereo/left/camera_info")
    parser.add_argument("--camera-frame", default="")
    parser.add_argument("--tag-family", default="tag25h9")
    parser.add_argument("--tag-id", type=int, default=-1)
    parser.add_argument("--tag-size", type=float, default=0.04)
    parser.add_argument("--hfov-deg", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-camera-info", action="store_true", help="Do not publish synthetic CameraInfo.")
    parser.add_argument("--no-stereo", action="store_true", help="Do not launch the stereo ROS publisher.")
    parser.add_argument("--left-config", default="ov9732_L")
    parser.add_argument("--right-config", default="ov9732_R")
    parser.add_argument("--flip", default="vertical", choices=["none", "vertical", "horizontal", "both"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tests_dir = Path(__file__).resolve().parent
    ws_src = _find_ws_src(tests_dir)
    calibration_dir = _find_calibration_dir(tests_dir)
    stereo_script = ws_src / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"
    detector_script = calibration_dir / "april_tag_detector.py"

    if not detector_script.exists():
        raise FileNotFoundError(f"AprilTag detector not found: {detector_script}")

    env = _build_env(ws_src)

    procs: list[subprocess.Popen] = []
    try:
        if not args.no_stereo:
            if not stereo_script.exists():
                raise FileNotFoundError(f"Stereo publisher not found: {stereo_script}")
            stereo_cmd = [
                sys.executable,
                str(stereo_script),
                "--left-topic",
                args.image_topic,
                "--right-topic",
                args.right_image_topic,
                "--left-config",
                args.left_config,
                "--right-config",
                args.right_config,
                "--flip",
                args.flip,
            ]
            procs.append(_start_process(stereo_cmd, env))
            time.sleep(1.0)

        detector_cmd = [
            sys.executable,
            str(detector_script),
            f"_image_topic:={args.image_topic}",
            f"_camera_info_topic:={args.camera_info_topic}",
            f"_tag_family:={args.tag_family}",
            f"_tag_id:={args.tag_id}",
            f"_tag_size:={args.tag_size}",
        ]
        if args.camera_frame:
            detector_cmd.append(f"_camera_frame:={args.camera_frame}")
        procs.append(_start_process(detector_cmd, env))

        rospy.init_node("test_april_tag_detection", anonymous=True)
        if not args.no_camera_info:
            CameraInfoPublisher(
                image_topic=args.image_topic,
                camera_info_topic=args.camera_info_topic,
                camera_frame=args.camera_frame,
                hfov_deg=args.hfov_deg,
            )
        watcher = TagDetectionWatcher(args.tag_family, args.tag_id)

        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if watcher.detected:
                return 0
            if (time.time() - start) > args.timeout:
                rospy.logerr("Timed out waiting for tag detection on /tf.")
                return 1
            rate.sleep()
        return 1
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
