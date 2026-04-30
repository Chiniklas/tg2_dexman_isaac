#!/usr/bin/env python3
"""ROS 2 publisher for stereo camera streams."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image

from stereo_camera.cameras.ov9732_camera import Ov9732Camera
from stereo_camera.utils.capture_two_stream import DualCameraReader, FLIP_MAP


class StereoRosPublisherNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("stereo_ros_publisher")

        self.declare_parameter("left_config", args.left_config)
        self.declare_parameter("right_config", args.right_config)
        self.declare_parameter("device_left", args.device_left or "")
        self.declare_parameter("device_right", args.device_right or "")
        self.declare_parameter("width", args.width or 0)
        self.declare_parameter("height", args.height or 0)
        self.declare_parameter("fps", args.fps or 0)
        self.declare_parameter("left_topic", args.left_topic)
        self.declare_parameter("right_topic", args.right_topic)
        self.declare_parameter("left_frame", args.left_frame)
        self.declare_parameter("right_frame", args.right_frame)
        self.declare_parameter("flip", args.flip)
        self.declare_parameter("max_fails", args.max_fails)
        self.declare_parameter("reconnect_wait", args.reconnect_wait)
        self.declare_parameter("rate", args.rate or 0.0)

        left_config = self._string_param("left_config")
        right_config = self._string_param("right_config")
        left_topic = self._string_param("left_topic")
        right_topic = self._string_param("right_topic")
        left_frame = self._string_param("left_frame")
        right_frame = self._string_param("right_frame")
        flip = self._string_param("flip")
        max_fails = int(self.get_parameter("max_fails").value)
        reconnect_wait = float(self.get_parameter("reconnect_wait").value)
        rate_hz = self._optional_float_param("rate")
        device_left = self._optional_string_param("device_left")
        device_right = self._optional_string_param("device_right")
        width = self._optional_int_param("width")
        height = self._optional_int_param("height")
        fps = self._optional_int_param("fps")

        if flip not in FLIP_MAP:
            raise ValueError("flip must be one of: none, vertical, horizontal, both")

        self.cam_left = Ov9732Camera.from_config(
            left_config,
            overrides={
                "device": device_left,
                "width": width,
                "height": height,
                "fps": fps,
            },
        )
        self.cam_right = Ov9732Camera.from_config(
            right_config,
            overrides={
                "device": device_right,
                "width": width,
                "height": height,
                "fps": fps,
            },
        )

        self.reader = DualCameraReader(
            [self.cam_left, self.cam_right],
            flip=flip,
            max_fails=max_fails,
            reconnect_wait=reconnect_wait,
        )
        self.reader.labels = ["L", "R"]

        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(Image, left_topic, 2)
        self.right_pub = self.create_publisher(Image, right_topic, 2)

        self.left_frame = left_frame
        self.right_frame = right_frame
        self.sleep_sec = (1.0 / rate_hz) if rate_hz and rate_hz > 0.0 else 0.0

    def _string_param(self, name: str) -> str:
        value = self.get_parameter(name).value
        return str(value).strip()

    def _optional_string_param(self, name: str) -> str | None:
        value = self._string_param(name)
        return value or None

    def _optional_int_param(self, name: str) -> int | None:
        value = int(self.get_parameter(name).value)
        return value if value > 0 else None

    def _optional_float_param(self, name: str) -> float | None:
        value = float(self.get_parameter(name).value)
        return value if value > 0.0 else None

    def start(self) -> None:
        self.reader.start()

    def stop(self) -> None:
        self.reader.stop()

    def run(self) -> None:
        while rclpy.ok():
            frames = self.reader.read_pair()
            if frames is None:
                if self.sleep_sec > 0.0:
                    time.sleep(self.sleep_sec)
                continue

            left_frame, right_frame = frames
            now = self.get_clock().now().to_msg()

            left_msg = self.bridge.cv2_to_imgmsg(left_frame, encoding="bgr8")
            left_msg.header.stamp = now
            left_msg.header.frame_id = self.left_frame
            self.left_pub.publish(left_msg)

            right_msg = self.bridge.cv2_to_imgmsg(right_frame, encoding="bgr8")
            right_msg.header.stamp = now
            right_msg.header.frame_id = self.right_frame
            self.right_pub.publish(right_msg)

            if self.sleep_sec > 0.0:
                time.sleep(self.sleep_sec)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 stereo camera publisher")
    parser.add_argument("--left-config", default="ov9732_L", help="Left camera config name (YAML)")
    parser.add_argument("--right-config", default="ov9732_R", help="Right camera config name (YAML)")
    parser.add_argument("--device-left", default=None, help="Override left device")
    parser.add_argument("--device-right", default=None, help="Override right device")
    parser.add_argument("--width", type=int, default=None, help="Override width (both cams)")
    parser.add_argument("--height", type=int, default=None, help="Override height (both cams)")
    parser.add_argument("--fps", type=int, default=None, help="Override fps (both cams)")
    parser.add_argument("--left-topic", default="/stereo/left/image_raw")
    parser.add_argument("--right-topic", default="/stereo/right/image_raw")
    parser.add_argument("--left-frame", default="stereo_left")
    parser.add_argument("--right-frame", default="stereo_right")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="both",
        help="Flip frames if cameras are inverted",
    )
    parser.add_argument("--max-fails", type=int, default=5)
    parser.add_argument("--reconnect-wait", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=None, help="Publish rate limit (Hz)")
    return parser.parse_args(argv)


def main(args: list[str] | None = None) -> int:
    cli_args = remove_ros_args(args=args or sys.argv)[1:]
    parsed_args = parse_args(cli_args)
    rclpy.init(args=args)

    node = StereoRosPublisherNode(parsed_args)
    try:
        node.start()
        node.run()
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
