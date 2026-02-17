#!/usr/bin/env python3
"""
Launch the stereo camera node and visualize the stream.

Usage (from a sourced ROS1 environment with `roscore` running):
    python test_stereo_camera_node.py \
      --left-topic /stereo/left/image_raw \
      --right-topic /stereo/right/image_raw
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

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


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
    raise FileNotFoundError(
        f"Could not locate stereo_ros_publisher.py by searching from {start}. "
        "Expected path: <ws>/src/stereo_camera/scripts/stereo_ros_publisher.py"
    )


class StereoViewer:
    def __init__(self, left_topic: str, right_topic: str, scale: float) -> None:
        self.bridge = CvBridge()
        self.left_img: Optional[object] = None
        self.right_img: Optional[object] = None
        self.scale = scale

        self._left_sub = rospy.Subscriber(left_topic, Image, self._left_cb, queue_size=1)
        self._right_sub = rospy.Subscriber(right_topic, Image, self._right_cb, queue_size=1)

    def _left_cb(self, msg: Image) -> None:
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _right_cb(self, msg: Image) -> None:
        try:
            self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def _prepare_pair(self):
        if self.left_img is None and self.right_img is None:
            return None

        if self.left_img is None:
            frame = self.right_img
        elif self.right_img is None:
            frame = self.left_img
        else:
            left = self.left_img
            right = self.right_img
            if left.shape[0] != right.shape[0]:
                target_h = min(left.shape[0], right.shape[0])
                left = cv2.resize(left, (int(left.shape[1] * target_h / left.shape[0]), target_h))
                right = cv2.resize(right, (int(right.shape[1] * target_h / right.shape[0]), target_h))
            frame = cv2.hconcat([left, right])

        if frame is None:
            return None

        if self.scale and abs(self.scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        return frame

    def spin(self, window_name: str) -> int:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._prepare_pair()
            if frame is not None:
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            rate.sleep()
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch stereo camera node and visualize stream.")
    parser.add_argument("--left-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", default="/stereo/right/image_raw")
    parser.add_argument("--left-config", default="ov9732_L")
    parser.add_argument("--right-config", default="ov9732_R")
    parser.add_argument("--flip", default="none", choices=["none", "vertical", "horizontal", "both"])
    parser.add_argument("--scale", type=float, default=0.6, help="Preview downscale factor.")
    parser.add_argument("--no-stereo", action="store_true", help="Do not launch the stereo ROS publisher.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    calibration_dir = Path(__file__).resolve().parent
    ws_src = _find_ws_src(calibration_dir)
    stereo_script = ws_src / "stereo_camera" / "scripts" / "stereo_ros_publisher.py"

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
                args.left_topic,
                "--right-topic",
                args.right_topic,
                "--left-config",
                args.left_config,
                "--right-config",
                args.right_config,
                "--flip",
                args.flip,
            ]
            procs.append(_start_process(stereo_cmd, env))
            time.sleep(1.0)

        rospy.init_node("test_stereo_camera_viewer", anonymous=True)
        viewer = StereoViewer(args.left_topic, args.right_topic, args.scale)
        return viewer.spin("Stereo Stream")
    finally:
        cv2.destroyAllWindows()
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
