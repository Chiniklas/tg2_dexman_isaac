#!/usr/bin/env python3
"""ROS1 publisher for stereo camera streams.

Publishes left/right Image messages from OV9732 cameras.
"""

from __future__ import annotations

import argparse
import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from stereo_camera.cameras.ov9732_camera import Ov9732Camera, load_camera_config
from stereo_camera.utils.capture_two_stream import DualCameraReader, FLIP_MAP


class StereoRosPublisher:
    def __init__(
        self,
        left_config: str,
        right_config: str,
        left_topic: str,
        right_topic: str,
        left_frame: str,
        right_frame: str,
        flip: str,
        max_fails: int,
        reconnect_wait: float,
        rate_hz: float | None,
        device_left: str | None,
        device_right: str | None,
        width: int | None,
        height: int | None,
        fps: int | None,
    ) -> None:
        if flip not in FLIP_MAP:
            raise ValueError("flip must be one of: none, vertical, horizontal, both")

        cfg_l = load_camera_config(left_config)
        cfg_r = load_camera_config(right_config)

        overrides_l = {
            "device": device_left or cfg_l.get("device"),
            "width": width or cfg_l.get("width"),
            "height": height or cfg_l.get("height"),
            "fps": fps or cfg_l.get("fps"),
        }
        overrides_r = {
            "device": device_right or cfg_r.get("device"),
            "width": width or cfg_r.get("width"),
            "height": height or cfg_r.get("height"),
            "fps": fps or cfg_r.get("fps"),
        }

        self.cam_left = Ov9732Camera.from_config(left_config, overrides=overrides_l)
        self.cam_right = Ov9732Camera.from_config(right_config, overrides=overrides_r)

        self.reader = DualCameraReader(
            [self.cam_left, self.cam_right],
            flip=flip,
            max_fails=max_fails,
            reconnect_wait=reconnect_wait,
        )
        self.reader.labels = ["L", "R"]

        self.bridge = CvBridge()
        self.left_pub = rospy.Publisher(left_topic, Image, queue_size=2)
        self.right_pub = rospy.Publisher(right_topic, Image, queue_size=2)

        self.left_frame = left_frame
        self.right_frame = right_frame
        self.rate_hz = rate_hz

    def start(self) -> None:
        self.reader.start()

    def stop(self) -> None:
        self.reader.stop()

    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz) if self.rate_hz else None
        while not rospy.is_shutdown():
            frames = self.reader.read_pair()
            if frames is None:
                if rate:
                    rate.sleep()
                continue

            left_frame, right_frame = frames
            now = rospy.Time.now()

            left_msg = self.bridge.cv2_to_imgmsg(left_frame, encoding="bgr8")
            left_msg.header.stamp = now
            left_msg.header.frame_id = self.left_frame
            self.left_pub.publish(left_msg)

            right_msg = self.bridge.cv2_to_imgmsg(right_frame, encoding="bgr8")
            right_msg.header.stamp = now
            right_msg.header.frame_id = self.right_frame
            self.right_pub.publish(right_msg)

            if rate:
                rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS1 stereo camera publisher")
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
        default="vertical",
        help="Flip frames if cameras are inverted",
    )
    parser.add_argument("--max-fails", type=int, default=5)
    parser.add_argument("--reconnect-wait", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=None, help="Publish rate limit (Hz)")
    return parser.parse_args(rospy.myargv(sys.argv)[1:])


def _param(name: str, default):
    return rospy.get_param(f"~{name}", default)


def _maybe_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    rospy.init_node("stereo_ros_publisher")

    left_config = _param("left_config", args.left_config)
    right_config = _param("right_config", args.right_config)
    left_topic = _param("left_topic", args.left_topic)
    right_topic = _param("right_topic", args.right_topic)
    left_frame = _param("left_frame", args.left_frame)
    right_frame = _param("right_frame", args.right_frame)
    flip = _param("flip", args.flip)
    max_fails = int(_param("max_fails", args.max_fails))
    reconnect_wait = float(_param("reconnect_wait", args.reconnect_wait))
    rate_hz = _maybe_float(_param("rate", args.rate))
    device_left = _param("device_left", args.device_left)
    device_right = _param("device_right", args.device_right)
    width = _param("width", args.width)
    height = _param("height", args.height)
    fps = _param("fps", args.fps)

    pub = StereoRosPublisher(
        left_config=left_config,
        right_config=right_config,
        left_topic=left_topic,
        right_topic=right_topic,
        left_frame=left_frame,
        right_frame=right_frame,
        flip=flip,
        max_fails=max_fails,
        reconnect_wait=reconnect_wait,
        rate_hz=rate_hz,
        device_left=device_left,
        device_right=device_right,
        width=width,
        height=height,
        fps=fps,
    )

    try:
        pub.start()
        pub.spin()
    finally:
        pub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
