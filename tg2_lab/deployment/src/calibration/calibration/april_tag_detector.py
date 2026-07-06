#!/usr/bin/env python3
"""ROS 2 AprilTag detector aligned with the reference detection behavior.

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
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from rclpy.node import Node
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


class AprilTagDetectorNode(Node):
    """Subscribe image stream, detect AprilTags, and publish poses/TF."""

    def __init__(self) -> None:
        super().__init__("april_tag_detector")

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco module not available. Install opencv-contrib-python.")

        self.declare_parameter("image_topic", "/stereo/left/image_raw")
        self.declare_parameter("camera_frame", "")
        self.image_topic = self._string_param("image_topic", "/stereo/left/image_raw")
        self.camera_frame_param = self._string_param("camera_frame", "")

        # Strict intrinsic loading: required, no fallback to CameraInfo.
        self.declare_parameter("intrinsics_npz", "")
        self.declare_parameter("intrinsics_camera", "auto")
        self.intrinsics_npz = self._string_param("intrinsics_npz", "")
        self.intrinsics_camera = self._string_param("intrinsics_camera", "auto").lower()
        if not self.intrinsics_npz:
            raise ValueError(
                "Parameter 'intrinsics_npz' is required. "
                "CameraInfo fallback is disabled by design."
            )

        # Tag settings
        self.declare_parameter("tag_family", "")
        self.declare_parameter("tag_dictionary", "")
        self.declare_parameter("tag_id", -1)
        self.declare_parameter("tag_size", 0.10)
        self.declare_parameter("tag_frame_prefix", "")
        tag_family = self._string_param("tag_family", "").lower()
        tag_dictionary = self._string_param("tag_dictionary", "")
        if tag_dictionary:
            self.tag_dictionary_name = tag_dictionary
        elif tag_family:
            self.tag_dictionary_name = FAMILY_TO_DICT.get(tag_family, "DICT_APRILTAG_25h9")
        else:
            self.tag_dictionary_name = "DICT_APRILTAG_25h9"
            tag_family = DICT_TO_FAMILY.get(self.tag_dictionary_name, "tag25h9")

        self.family_label = DICT_TO_FAMILY.get(self.tag_dictionary_name, tag_family or "tag25h9")
        self.tag_id_filter = int(self.get_parameter("tag_id").value)
        self.tag_size = float(self.get_parameter("tag_size").value)
        self.tag_frame_prefix = self._string_param("tag_frame_prefix", "")

        # Output controls
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_pose_array", True)
        self.declare_parameter("publish_transform_matrix", True)
        self.declare_parameter("log_camera_pose", True)
        self.declare_parameter("debug_view", False)
        self.declare_parameter("debug_window_name", "AprilTag Detection")
        self.declare_parameter("input_reflip", "none")
        self.declare_parameter("debug_display_flip", "none")
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.publish_pose_array = bool(self.get_parameter("publish_pose_array").value)
        self.publish_transform_matrix = bool(self.get_parameter("publish_transform_matrix").value)
        self.log_camera_pose = bool(self.get_parameter("log_camera_pose").value)
        self.debug_view = bool(self.get_parameter("debug_view").value)
        self.debug_window_name = self._string_param("debug_window_name", "AprilTag Detection")
        self.input_reflip = self._string_param("input_reflip", "none").lower()
        self.debug_display_flip = self._string_param("debug_display_flip", "none").lower()
        if self.input_reflip not in {"none", "vertical", "horizontal", "both"}:
            raise ValueError("input_reflip must be one of: none, vertical, horizontal, both")
        if self.debug_display_flip not in {"none", "vertical", "horizontal", "both"}:
            raise ValueError("debug_display_flip must be one of: none, vertical, horizontal, both")
        self._debug_frame_count = 0
        self._debug_fps = 0.0
        self._debug_t_prev = time.time()
        self._checked_intrinsic_vs_image = False
        self._throttle_log_times: dict[str, float] = {}

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

        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.pose_array_pub = self.create_publisher(PoseArray, "tag_detections", 10)
        self.tag_ids_pub = self.create_publisher(Int32MultiArray, "tag_detection_ids", 10)
        self.matrix_pub = self.create_publisher(Float32MultiArray, "apriltag2_transform_matrix", 10)

        self._image_sub = self.create_subscription(Image, self.image_topic, self._image_cb, 1)

        self.get_logger().info(
            f"AprilTag detector ready. image={self.image_topic} "
            f"dictionary={self.tag_dictionary_name} tag_size={self.tag_size:.3f}m "
            f"intrinsics={self.intrinsics_npz}"
        )
        self.get_logger().info(
            "Detection uses image after "
            f"input_reflip={self.input_reflip}. "
            f"debug_display_flip={self.debug_display_flip} is visualization-only."
        )
        self.get_logger().info(
            "Pipeline: raw image -> input_reflip -> detect/solvePnP -> publish TF/poses -> debug display flip."
        )

    def _string_param(self, name: str, default: str) -> str:
        value = self.get_parameter(name).value
        if value is None:
            return default
        return str(value).strip()

    def _log_throttled(self, key: str, period_sec: float, level: str, message: str) -> None:
        now = time.monotonic()
        last = self._throttle_log_times.get(key, -1e18)
        if (now - last) < period_sec:
            return
        self._throttle_log_times[key] = now
        logger = self.get_logger()
        if level == "warn":
            logger.warn(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

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

        self.get_logger().warn(f"Unknown tag_dictionary '{name}', falling back to DICT_APRILTAG_25h9")
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

        self.get_logger().info(f"Loaded {cam_label}-camera intrinsics from {npz_path} (key {key_k}/{key_d})")

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

    @staticmethod
    def _apply_flip(frame: np.ndarray, mode: str) -> np.ndarray:
        if mode == "vertical":
            return cv2.flip(frame, 0)
        if mode == "horizontal":
            return cv2.flip(frame, 1)
        if mode == "both":
            return cv2.flip(frame, -1)
        return frame

    def _apply_debug_display_flip(self, frame: np.ndarray) -> np.ndarray:
        return self._apply_flip(frame, self.debug_display_flip)

    def _flip_point_for_display(self, x: int, y: int, width: int, height: int) -> tuple[int, int]:
        if self.debug_display_flip == "vertical":
            return x, height - 1 - y
        if self.debug_display_flip == "horizontal":
            return width - 1 - x, y
        if self.debug_display_flip == "both":
            return width - 1 - x, height - 1 - y
        return x, y

    @staticmethod
    def _fit_overlay_text(
        text: str,
        width_px: int,
        *,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        base_scale: float = 0.55,
        thickness: int = 2,
        x_pad: int = 8,
    ) -> tuple[str, float]:
        """Keep overlay text readable and inside frame width."""
        max_w = max(20, int(width_px) - 2 * x_pad)
        scale = base_scale

        for _ in range(4):
            text_w = cv2.getTextSize(text, font, scale, thickness)[0][0]
            if text_w <= max_w:
                return text, scale
            scale *= 0.9

        if len(text) <= 3:
            return text, scale

        lo, hi = 1, len(text)
        best = "..."
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid] + "..."
            text_w = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
            if text_w <= max_w:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best, scale

    def _image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None or self.dist_coeffs is None:
            self._log_throttled("intrinsics_missing", 5.0, "warn", "No intrinsics loaded; skipping tag detection.")
            return

        if not self._checked_intrinsic_vs_image:
            cx = float(self.camera_matrix[0, 2])
            cy = float(self.camera_matrix[1, 2])
            w = float(msg.width)
            h = float(msg.height)
            if w > 0 and h > 0:
                if abs(cx - (w * 0.5)) > (0.2 * w) or abs(cy - (h * 0.5)) > (0.2 * h):
                    self.get_logger().warn(
                        "Possible intrinsic/image mismatch: "
                        f"image={msg.width}x{msg.height} but principal point=({cx:.1f}, {cy:.1f}). "
                        "Check stream resolution/flip against calibration file."
                    )
            self._checked_intrinsic_vs_image = True

        frame, gray = self._to_gray(msg)
        if self.input_reflip != "none":
            frame = self._apply_flip(frame, self.input_reflip)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.camera_frame or msg.header.frame_id
        ids_msg = Int32MultiArray()
        ids_msg.data = []
        pending_text: list[tuple[str, int, int, tuple[int, int, int]]] = []

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
                    self._log_throttled(
                        f"solvepnp_{tag_id}",
                        1.0,
                        "warn",
                        f"Failed to solvePnP for tag {tag_id}",
                    )
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
                cxy = marker_corners.reshape(-1, 2).mean(axis=0).astype(int)
                pending_text.append(
                    (
                        f"id={tag_id} d={distance:.2f}m z={position[2]:.2f}m",
                        int(cxy[0]) - 40,
                        int(cxy[1]) - 10,
                        (0, 255, 0),
                    )
                )

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

                self._log_throttled(
                    f"tag_pos_{tag_id}",
                    0.5,
                    "info",
                    "AprilTag "
                    f"{tag_id} position (x,y,z)=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})m "
                    f"distance={distance:.3f}m",
                )
                self._log_throttled(
                    f"tag_rpy_{tag_id}",
                    0.5,
                    "info",
                    f"AprilTag {tag_id} orientation rpy_deg=({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f})",
                )
                if self.log_camera_pose:
                    camera_position_tag = -rot.T @ tvec
                    camera_position_tag = camera_position_tag.flatten()
                    self._log_throttled(
                        f"camera_pose_{tag_id}",
                        0.5,
                        "info",
                        "Camera pose wrt tag "
                        f"{tag_id} -> position (x,y,z)=({camera_position_tag[0]:.3f}, "
                        f"{camera_position_tag[1]:.3f}, {camera_position_tag[2]:.3f})m",
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
            self._debug_frame_count += 1
            t_now = time.time()
            dt = t_now - self._debug_t_prev
            if dt >= 0.5:
                self._debug_fps = self._debug_frame_count / dt
                self._debug_frame_count = 0
                self._debug_t_prev = t_now

            status = (
                f"{self.family_label}"
                + (f":{self.tag_id_filter}" if self.tag_id_filter >= 0 else ":*")
                + f" | det={len(ids_msg.data)} | fps={self._debug_fps:.1f}"
            )
            frame_display = self._apply_debug_display_flip(frame)
            h, w = frame_display.shape[:2]
            for text, x, y, color in pending_text:
                tx, ty = self._flip_point_for_display(x, y, w, h)
                cv2.putText(
                    frame_display,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            status_text, status_scale = self._fit_overlay_text(status, w)
            cv2.putText(
                frame_display,
                status_text,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                status_scale,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.debug_window_name, frame_display)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        _shutdown_cleanup()
        return super().destroy_node()


def _shutdown_cleanup() -> None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
