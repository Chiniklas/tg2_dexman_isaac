#!/usr/bin/env python3
"""ROS1 AprilTag detector aligned with the reference detection behavior.

Core behavior mirrors the reference:
- detect tags from RGB frames with cv2.aruco
- solve pose with solvePnP
- publish detections (PoseArray + IDs)
- draw marker + axes overlays for debug view

Deployment-specific difference:
- camera intrinsics are loaded strictly from a calibration .npz file
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import tf2_ros


# Backward-compatible mapping for existing launch/readme usage
FAMILY_TO_DICT = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}

DICT_TO_FAMILY = {v: k for k, v in FAMILY_TO_DICT.items()}


class AprilTagDetectorNode:
    """Subscribe image stream, detect AprilTags, and publish poses/TF."""

    def __init__(self) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module not available. Install opencv-contrib-python.")

        self.image_topic = rospy.get_param("~image_topic", "/stereo/left/image_raw")
        self.camera_frame_param = rospy.get_param("~camera_frame", "")

        # Strict intrinsic loading: required, no fallback to CameraInfo.
        self.intrinsics_npz = rospy.get_param("~intrinsics_npz", "").strip()
        self.intrinsics_camera = rospy.get_param("~intrinsics_camera", "auto").strip().lower()
        if not self.intrinsics_npz:
            raise ValueError(
                "Parameter '~intrinsics_npz' is required. "
                "CameraInfo fallback is disabled by design."
            )

        # Tag settings
        tag_family = rospy.get_param("~tag_family", "").strip().lower()
        tag_dictionary = rospy.get_param("~tag_dictionary", "").strip()
        if tag_dictionary:
            self.tag_dictionary_name = tag_dictionary
        elif tag_family:
            self.tag_dictionary_name = FAMILY_TO_DICT.get(tag_family, "DICT_APRILTAG_25h9")
        else:
            self.tag_dictionary_name = "DICT_APRILTAG_25h9"
            tag_family = DICT_TO_FAMILY.get(self.tag_dictionary_name, "tag25h9")

        self.family_label = DICT_TO_FAMILY.get(self.tag_dictionary_name, tag_family or "tag25h9")
        self.tag_id_filter = int(rospy.get_param("~tag_id", -1))
        self.tag_size = float(rospy.get_param("~tag_size", 0.10))
        self.tag_frame_prefix = rospy.get_param("~tag_frame_prefix", "").strip()

        # Output controls
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.publish_pose_array = bool(rospy.get_param("~publish_pose_array", True))
        self.publish_transform_matrix = bool(rospy.get_param("~publish_transform_matrix", True))
        self.log_camera_pose = bool(rospy.get_param("~log_camera_pose", True))
        self.debug_view = bool(rospy.get_param("~debug_view", False))
        self.debug_window_name = rospy.get_param("~debug_window_name", "AprilTag Detection")
        self.debug_display_flip = rospy.get_param("~debug_display_flip", "none").strip().lower()

        self.bridge = CvBridge()

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_frame: Optional[str] = None
        self._load_intrinsics_from_npz(Path(self.intrinsics_npz))
        if self.camera_frame is None:
            self.camera_frame = self.camera_frame_param if self.camera_frame_param else "camera"

        half = self.tag_size / 2.0
        self.obj_points = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float32,
        )

        dictionary = self._resolve_dictionary(self.tag_dictionary_name)
        self.detector_parameters = self._create_detector_parameters()
        self._detect: Callable[[np.ndarray], Tuple] = self._build_detect_fn(dictionary, self.detector_parameters)

        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.pose_array_pub = rospy.Publisher("tag_detections", PoseArray, queue_size=10)
        self.tag_ids_pub = rospy.Publisher("tag_detection_ids", Int32MultiArray, queue_size=10)
        self.matrix_pub = rospy.Publisher("apriltag2_transform_matrix", Float32MultiArray, queue_size=10)

        self._image_sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo(
            "AprilTag detector ready. image=%s dictionary=%s tag_size=%.3fm intrinsics=%s",
            self.image_topic,
            self.tag_dictionary_name,
            self.tag_size,
            self.intrinsics_npz,
        )

    @staticmethod
    def _create_detector_parameters():
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        raise AttributeError("OpenCV aruco module has no DetectorParameters factory")

    @staticmethod
    def _build_detect_fn(dictionary, parameters):
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            return detector.detectMarkers

        return lambda img: cv2.aruco.detectMarkers(  # type: ignore[return-value]
            img, dictionary, parameters=parameters
        )

    def _resolve_dictionary(self, name: str):
        if hasattr(cv2.aruco, name):
            return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))

        rospy.logwarn(
            "Unknown tag_dictionary '%s', falling back to DICT_APRILTAG_25h9",
            name,
        )
        self.tag_dictionary_name = "DICT_APRILTAG_25h9"
        self.family_label = DICT_TO_FAMILY.get(self.tag_dictionary_name, "tag25h9")
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)

    def _load_intrinsics_from_npz(self, npz_path: Path) -> None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {npz_path}")

        calib = np.load(npz_path)
        use_right = self.intrinsics_camera == "right" or (
            self.intrinsics_camera == "auto" and "/right/" in self.image_topic
        )
        if use_right:
            key_k_candidates = ("k2", "k")
            key_d_candidates = ("d2", "d")
            cam_label = "right"
        else:
            key_k_candidates = ("k1", "k")
            key_d_candidates = ("d1", "d")
            cam_label = "left"

        key_k = next((k for k in key_k_candidates if k in calib), None)
        key_d = next((k for k in key_d_candidates if k in calib), None)
        if key_k is None or key_d is None:
            raise KeyError(
                f"Could not find intrinsic keys in {npz_path}. "
                f"Expected one of {key_k_candidates} and {key_d_candidates}."
            )

        self.camera_matrix = np.array(calib[key_k], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(calib[key_d], dtype=np.float32).reshape(-1)
        self.camera_frame = self.camera_frame_param if self.camera_frame_param else self.camera_frame

        rospy.loginfo(
            "Loaded %s-camera intrinsics from %s (key %s/%s)",
            cam_label,
            npz_path,
            key_k,
            key_d,
        )

    def _to_gray(self, msg: Image) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray

    def _make_child_frame_id(self, tag_id: int) -> str:
        if self.tag_frame_prefix:
            return f"{self.tag_frame_prefix}{tag_id}"
        return f"{self.family_label}:{tag_id}"

    @staticmethod
    def _rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation_matrix))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

        q = np.array([w, x, y, z], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q / n

    @staticmethod
    def _rotation_matrix_to_rpy_deg(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
        sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0.0
        return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))

    def _apply_debug_display_flip(self, frame: np.ndarray) -> np.ndarray:
        if self.debug_display_flip == "vertical":
            return cv2.flip(frame, 0)
        if self.debug_display_flip == "horizontal":
            return cv2.flip(frame, 1)
        if self.debug_display_flip == "both":
            return cv2.flip(frame, -1)
        return frame

    def _image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.logwarn_throttle(5.0, "No intrinsics loaded; skipping tag detection.")
            return

        frame, gray = self._to_gray(msg)
        corners, ids, _ = self._detect(gray)

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.camera_frame or msg.header.frame_id
        ids_msg = Int32MultiArray()
        ids_msg.data = []

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for tag_id, marker_corners in zip(ids.flatten(), corners):
                tag_id = int(tag_id)
                if self.tag_id_filter >= 0 and tag_id != self.tag_id_filter:
                    continue

                img_pts = marker_corners.reshape(4, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    img_pts,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    rospy.logwarn_throttle(1.0, "Failed to solvePnP for tag %d", tag_id)
                    continue

                cv2.drawFrameAxes(
                    frame,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.tag_size * 0.5,
                )

                position = tvec.flatten()
                distance = math.sqrt(float(np.dot(position, position)))
                rot, _ = cv2.Rodrigues(rvec)
                roll_deg, pitch_deg, yaw_deg = self._rotation_matrix_to_rpy_deg(rot)
                quat_wxyz = self._rotation_matrix_to_quaternion_wxyz(rot)

                pose = Pose()
                pose.position.x = float(position[0])
                pose.position.y = float(position[1])
                pose.position.z = float(position[2])
                pose.orientation.w = float(quat_wxyz[0])
                pose.orientation.x = float(quat_wxyz[1])
                pose.orientation.y = float(quat_wxyz[2])
                pose.orientation.z = float(quat_wxyz[3])
                pose_array.poses.append(pose)
                ids_msg.data.append(tag_id)

                rospy.loginfo_throttle(
                    0.5,
                    "AprilTag %d position (x,y,z)=(%.3f, %.3f, %.3f)m distance=%.3fm",
                    tag_id,
                    position[0],
                    position[1],
                    position[2],
                    distance,
                )
                rospy.loginfo_throttle(
                    0.5,
                    "AprilTag %d orientation rpy_deg=(%.1f, %.1f, %.1f)",
                    tag_id,
                    roll_deg,
                    pitch_deg,
                    yaw_deg,
                )
                if self.log_camera_pose:
                    camera_position_tag = -rot.T @ tvec
                    camera_position_tag = camera_position_tag.flatten()
                    rospy.loginfo_throttle(
                        0.5,
                        "Camera pose wrt tag %d -> position (x,y,z)=(%.3f, %.3f, %.3f)m",
                        tag_id,
                        camera_position_tag[0],
                        camera_position_tag[1],
                        camera_position_tag[2],
                    )

                if self.publish_tf:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = msg.header.stamp
                    tf_msg.header.frame_id = self.camera_frame or msg.header.frame_id
                    tf_msg.child_frame_id = self._make_child_frame_id(tag_id)
                    tf_msg.transform.translation.x = float(position[0])
                    tf_msg.transform.translation.y = float(position[1])
                    tf_msg.transform.translation.z = float(position[2])
                    tf_msg.transform.rotation.w = float(quat_wxyz[0])
                    tf_msg.transform.rotation.x = float(quat_wxyz[1])
                    tf_msg.transform.rotation.y = float(quat_wxyz[2])
                    tf_msg.transform.rotation.z = float(quat_wxyz[3])
                    self._tf_broadcaster.sendTransform(tf_msg)

                if self.publish_transform_matrix:
                    mat = np.eye(4, dtype=np.float32)
                    mat[:3, :3] = rot.astype(np.float32)
                    mat[:3, 3] = position.astype(np.float32)
                    self.matrix_pub.publish(Float32MultiArray(data=mat.reshape(-1).tolist()))

        if self.publish_pose_array:
            self.pose_array_pub.publish(pose_array)
            self.tag_ids_pub.publish(ids_msg)

        if self.debug_view:
            cv2.imshow(self.debug_window_name, self._apply_debug_display_flip(frame))
            cv2.waitKey(1)


def _shutdown_cleanup() -> None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main() -> None:
    rospy.init_node("april_tag_detector")
    rospy.on_shutdown(_shutdown_cleanup)
    AprilTagDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
