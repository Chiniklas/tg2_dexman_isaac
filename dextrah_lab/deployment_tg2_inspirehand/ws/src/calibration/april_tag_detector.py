#!/usr/bin/env python3
"""
AprilTag detector for ROS1 that publishes tag poses to TF.

Uses OpenCV's aruco module (AprilTag dictionaries) to detect tags in a camera
image stream, estimates pose using camera intrinsics from CameraInfo, and
broadcasts a transform from the camera optical frame to the tag frame.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
import tf
import tf2_ros


DICT_MAP = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


class AprilTagDetectorNode:
    def __init__(self) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module not available. Install opencv-contrib-python.")

        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.camera_frame_param = rospy.get_param("~camera_frame", "")
        self.tag_family = rospy.get_param("~tag_family", "tag25h9")
        self.tag_id_filter = rospy.get_param("~tag_id", -1)
        self.tag_size = float(rospy.get_param("~tag_size", 0.04))
        self.tag_frame_prefix = rospy.get_param("~tag_frame_prefix", "")
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))

        if self.tag_family not in DICT_MAP:
            rospy.logwarn(
                "Unknown tag_family '%s'. Supported: %s. Falling back to tag25h9.",
                self.tag_family,
                ", ".join(sorted(DICT_MAP.keys())),
            )
            self.tag_family = "tag25h9"

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_frame: Optional[str] = None

        aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[self.tag_family])
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            self._detect = self._detector.detectMarkers
        else:
            self._detector = None
            self._detect = lambda img: cv2.aruco.detectMarkers(  # type: ignore[return-value]
                img, aruco_dict, parameters=cv2.aruco.DetectorParameters_create()
            )

        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        self._camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        self._image_sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo("AprilTag detector ready. Subscribed to %s and %s", self.image_topic, self.camera_info_topic)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is not None:
            return
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        if msg.D:
            self.dist_coeffs = np.array(msg.D, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)
        if self.camera_frame_param:
            self.camera_frame = self.camera_frame_param
        else:
            self.camera_frame = msg.header.frame_id
        rospy.loginfo("Camera intrinsics received. frame_id=%s", self.camera_frame)

    def _to_gray_u8(self, msg: Image) -> np.ndarray:
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if cv_img.ndim == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def _make_child_frame_id(self, tag_id: int) -> str:
        if self.tag_frame_prefix:
            return f"{self.tag_frame_prefix}{tag_id}"
        return f"{self.tag_family}:{tag_id}"

    def _image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn_throttle(5.0, "No CameraInfo yet; skipping tag detection.")
            return
        if not self.publish_tf:
            return

        gray = self._to_gray_u8(msg)
        corners, ids, _ = self._detect(gray)
        if ids is None or len(ids) == 0:
            return

        try:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.tag_size, self.camera_matrix, self.dist_coeffs
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            rospy.logwarn_throttle(5.0, "Pose estimation failed: %s", exc)
            return

        for idx, tag_id_arr in enumerate(ids.flatten()):
            tag_id = int(tag_id_arr)
            if self.tag_id_filter >= 0 and tag_id != self.tag_id_filter:
                continue

            rvec = rvecs[idx].reshape(3)
            tvec = tvecs[idx].reshape(3)
            rot_mtx, _ = cv2.Rodrigues(rvec)
            rot4 = np.eye(4, dtype=np.float64)
            rot4[:3, :3] = rot_mtx
            quat = tf.transformations.quaternion_from_matrix(rot4)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = msg.header.stamp
            tf_msg.header.frame_id = self.camera_frame or msg.header.frame_id
            tf_msg.child_frame_id = self._make_child_frame_id(tag_id)
            tf_msg.transform.translation.x = float(tvec[0])
            tf_msg.transform.translation.y = float(tvec[1])
            tf_msg.transform.translation.z = float(tvec[2])
            tf_msg.transform.rotation.x = float(quat[0])
            tf_msg.transform.rotation.y = float(quat[1])
            tf_msg.transform.rotation.z = float(quat[2])
            tf_msg.transform.rotation.w = float(quat[3])

            self._tf_broadcaster.sendTransform(tf_msg)


def main() -> None:
    rospy.init_node("april_tag_detector")
    AprilTagDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
