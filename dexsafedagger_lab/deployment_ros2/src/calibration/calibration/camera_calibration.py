#!/usr/bin/env python3
# Copyright (c) 2024, Nvidia.  All rights reserved.

"""
Camera calibration pipeline for TG2 Inspirehand (ROS1).

Publishes direct joint-space trajectories (same execution style as
tests/test_execute_targets.py), logs (camera->tag pose, robot joints), runs a
Gauss-Newton optimizer to solve camera->robot and palm->tag transforms, and
saves the robot->camera transform to disk.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from typing import Iterable

import numpy as np
import torch
import warp as wp

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from calibration.repo_support import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dexsafedagger_lab.utils.kinematics import Kinematics
from dexsafedagger_lab.utils.rotation_utils import euler_to_matrix, quaternion_to_matrix
from dexsafedagger_lab.utils.utils import initialize_warp

RIGHT_ARM_HAND_JOINTS = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
    "little_joint_0",
    "ring_joint_0",
    "middle_joint_0",
    "index_joint_0",
    "thumb_joint_0",
    "thumb_joint_1",
]


def _transform_from_translation_euler(translation: np.ndarray, euler_angles: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from translation + Euler ZYX (radians)."""
    euler_angles = torch.tensor(euler_angles, dtype=torch.float32).unsqueeze(0)
    rotation_matrix = euler_to_matrix(euler_angles)[0].cpu().numpy()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def _warp_transform_to_matrix(transform: torch.Tensor) -> torch.Tensor:
    """Convert a warp transform (p, q_xyzw) to a 4x4 matrix tensor."""
    pos = transform[..., :3]
    quat_xyzw = transform[..., 3:7]
    quat_wxyz = torch.stack(
        (quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]),
        dim=-1,
    )
    rot = quaternion_to_matrix(quat_wxyz)
    batch = transform.shape[0]
    mat = torch.eye(4, device=transform.device, dtype=transform.dtype).unsqueeze(0).repeat(batch, 1, 1)
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = pos
    return mat


def _parse_vec(raw: str, label: str, size: int) -> np.ndarray:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(vals) != size:
        raise ValueError(f"{label} must have {size} comma-separated values, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64)


def _interpolate(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps < 2:
        return [a.copy(), b.copy()]
    out: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, num=steps):
        out.append(((1.0 - alpha) * a + alpha * b).astype(np.float64))
    return out


class OptimizeCameraCalibration:
    """Optimizer wrapper for camera calibration using AR tag poses and robot joint states."""

    def __init__(self, urdf_path: str, palm_link: str, device: str):
        self.device = device
        self.batch_size = 1
        self.kinematics = Kinematics(urdf_path, self.batch_size, device=device)
        self.palm_link_index = self.kinematics.get_link_index(palm_link)
        self.cspace_name2index = self.kinematics.urdf_info.cspace_name2index_map

    def build_q_from_joint_state(self, msg: JointState) -> np.ndarray:
        q = np.zeros(self.kinematics.cspace_dim, dtype=np.float32)
        seen = 0
        for name, position in zip(msg.name, msg.position):
            idx = self.cspace_name2index.get(name)
            if idx is not None:
                q[idx] = position
                seen += 1
        if seen == 0:
            print("Warning: no joint names matched the URDF cspace.")
        return q

    def _robot_T_palm(self, joint_q: np.ndarray) -> np.ndarray:
        q_torch = torch.tensor(joint_q, device=self.device, dtype=torch.float32).unsqueeze(0)
        q_wp = wp.torch.from_torch(q_torch)
        self.kinematics.eval(q_wp, jacobians=False)
        link_tf = self.kinematics.batch_link_transforms_torch[0, self.palm_link_index]
        link_tf = link_tf.unsqueeze(0)
        mat = _warp_transform_to_matrix(link_tf)[0].cpu().numpy()
        return mat

    def calibrate_camera(self, tag_poses: list[np.ndarray], robot_joints: list[np.ndarray]) -> np.ndarray:
        if len(tag_poses) == 0:
            raise ValueError("No tag poses collected; calibration aborted.")
        max_iters = 100
        min_iters = 30
        parameters = np.zeros(12, dtype=np.float64)
        cost = 1000.0
        cost_threshold = 1e-5
        for i in range(max_iters):
            if cost < cost_threshold and i > min_iters:
                break
            if (i % 10 == 0) and cost >= cost_threshold:
                parameters = parameters + 0.5 * (np.random.rand(12) - 0.5)
            cost, cost_vector = self.calc_pose_loss(parameters, tag_poses, robot_joints)
            jacobian = self.calc_jacobian(parameters, cost_vector, tag_poses, robot_joints)
            parameters = self.gauss_newton(parameters, jacobian, cost_vector)
        return parameters

    def save_calibration_matrices(self, parameters: np.ndarray, camera: str) -> None:
        robot_T_cam = np.linalg.inv(_transform_from_translation_euler(parameters[:3], parameters[3:6]))
        np.savetxt(f"robot_cam_{camera}_calibration.txt", robot_T_cam, delimiter=",")

    def calc_pose_loss(
        self, parameters: np.ndarray, tag_poses: list[np.ndarray], robot_joints: list[np.ndarray]
    ) -> tuple[float, np.ndarray]:
        cost_vector: list[float] = []
        for meas_cam_T_tag, joint_angles in zip(tag_poses, robot_joints):
            cam_T_robot = _transform_from_translation_euler(parameters[:3], parameters[3:6])
            robot_T_palm = self._robot_T_palm(joint_angles)
            palm_T_tag = _transform_from_translation_euler(parameters[6:9], parameters[9:12])
            cam_T_tag = cam_T_robot @ robot_T_palm @ palm_T_tag

            first_point = cam_T_tag[:3, 3].flatten()

            second_point_offset = np.zeros((4, 1))
            second_point_offset[0, 0] = 0.1
            second_point_offset[3, 0] = 1.0
            second_point = (cam_T_tag @ second_point_offset)[:3, 0]

            third_point_offset = np.zeros((4, 1))
            third_point_offset[1, 0] = 0.1
            third_point_offset[3, 0] = 1.0
            third_point = (cam_T_tag @ third_point_offset)[:3, 0]

            first_point_target = meas_cam_T_tag[:3, 3].flatten()
            second_point_target = (meas_cam_T_tag @ second_point_offset)[:3, 0]
            third_point_target = (meas_cam_T_tag @ third_point_offset)[:3, 0]

            cost_vector += list(first_point - first_point_target)
            cost_vector += list(second_point - second_point_target)
            cost_vector += list(third_point - third_point_target)

        cost_vector_np = np.array(cost_vector)
        cost = (0.5 / len(tag_poses)) * np.dot(cost_vector_np, cost_vector_np)
        return cost, cost_vector_np

    def calc_jacobian(
        self,
        parameters: np.ndarray,
        cost_vector: np.ndarray,
        tag_poses: list[np.ndarray],
        robot_joints: list[np.ndarray],
    ) -> np.ndarray:
        jacobian = np.zeros((len(cost_vector), len(parameters)))
        eps = 1e-6
        for j in range(len(parameters)):
            parameters[j] += eps
            _, cost_vector_new = self.calc_pose_loss(parameters, tag_poses, robot_joints)
            for i in range(len(cost_vector)):
                jacobian[i, j] = (cost_vector_new[i] - cost_vector[i]) / eps
            parameters[j] -= eps
        return jacobian

    def gauss_newton(self, parameters: np.ndarray, jacobian: np.ndarray, cost_vector: np.ndarray) -> np.ndarray:
        inertia = 1e-6 * np.eye(jacobian.shape[1])
        step = np.linalg.inv(jacobian.T @ jacobian + inertia) @ jacobian.T @ cost_vector
        return parameters - step


class CameraCalibrationNode(Node):
    def __init__(
        self,
        camera: str,
        urdf_path: str,
        palm_link: str,
        device: str,
        joint_state_topic: str,
        command_topic: str,
        command_joint_names: list[str],
        start_joints: np.ndarray | None,
        target_joints: np.ndarray,
        num_steps: int,
        command_rate_hz: float,
        max_command_step_rad: float,
        feedback_timeout_sec: float,
        wait_feedback_sec: float,
        tag_frame_id: str,
        tf_topic: str,
        required_valid_pairs: int,
        max_pose_commands: int,
        settle_sec: float,
    ):
        super().__init__("tg2_camera_calibration")

        self.camera = camera
        self.device = device
        initialize_warp(self.device)

        self.command_topic = command_topic
        self.command_joint_names = list(command_joint_names)
        self.command_dof = len(self.command_joint_names)
        self.start_joints = start_joints.copy() if start_joints is not None else None
        self.target_joints = target_joints.copy()
        self.num_steps = int(num_steps)
        self.command_rate_hz = float(command_rate_hz)
        self.max_command_step_rad = float(max_command_step_rad)
        self.feedback_timeout_sec = float(feedback_timeout_sec)
        self.wait_feedback_sec = float(wait_feedback_sec)
        self.settle_sec = float(settle_sec)

        self._joint_state_msg: JointState | None = None
        self._joint_feedback_time = time.time()

        self.tag_frame_id = tag_frame_id
        self.got_tag_info = False
        self.tag_transform = np.zeros((4, 4))
        self.tag_transform[3, 3] = 1.0

        self.optimizer = OptimizeCameraCalibration(urdf_path=urdf_path, palm_link=palm_link, device=device)

        self._cmd_pub = self.create_publisher(JointState, self.command_topic, 10)

        self._joint_sub = self.create_subscription(JointState, joint_state_topic, self._joint_state_callback, 1)
        self._tf_sub = self.create_subscription(TFMessage, tf_topic, self._tf_callback, 10)

        self.tag_transforms: list[np.ndarray] = []
        self.robot_joints: list[np.ndarray] = []

        self.required_valid_pairs = int(required_valid_pairs)
        self.max_pose_commands = int(max_pose_commands)

    def _joint_state_callback(self, msg: JointState) -> None:
        self._joint_feedback_time = time.time()
        self._joint_state_msg = msg

    def _extract_command_vector(self, msg: JointState) -> np.ndarray | None:
        idx = {name: i for i, name in enumerate(msg.name)}
        q = np.zeros(self.command_dof, dtype=np.float64)
        for j, name in enumerate(self.command_joint_names):
            i = idx.get(name)
            if i is None or i >= len(msg.position):
                return None
            q[j] = float(msg.position[i])
        return q

    def _tf_callback(self, msg: TFMessage) -> None:
        if len(msg.transforms) == 0:
            self.got_tag_info = False
            return
        for tag in msg.transforms:
            if tag.child_frame_id == self.tag_frame_id:
                trans = np.array(
                    [
                        tag.transform.translation.x,
                        tag.transform.translation.y,
                        tag.transform.translation.z,
                    ]
                )
                quat = torch.tensor(
                    [
                        [
                            tag.transform.rotation.w,
                            tag.transform.rotation.x,
                            tag.transform.rotation.y,
                            tag.transform.rotation.z,
                        ]
                    ],
                    device="cpu",
                )
                rot_matrix = quaternion_to_matrix(quat)[0].cpu().numpy()
                self.tag_transform[:3, 3] = trans
                self.tag_transform[:3, :3] = rot_matrix
                self.got_tag_info = True
                return
        self.got_tag_info = False

    def record_data(self) -> bool:
        if not self.got_tag_info or self._joint_state_msg is None:
            print("Missing tag or joint state data; skipping frame.")
            return False
        self.tag_transforms.append(copy.copy(self.tag_transform))
        q = self.optimizer.build_q_from_joint_state(self._joint_state_msg)
        self.robot_joints.append(q)
        self.got_tag_info = False
        return True

    def _publish_joint_target(self, q_cmd: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.command_joint_names)
        msg.position = q_cmd.astype(float).tolist()
        msg.velocity = []
        msg.effort = []
        self._cmd_pub.publish(msg)

    def _build_sweep_trajectory(self, q_start: np.ndarray, q_target: np.ndarray) -> list[np.ndarray]:
        up = _interpolate(q_start, q_target, self.num_steps)
        down = _interpolate(q_target, q_start, self.num_steps)
        return up + down[1:]

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.time() + max(0.0, duration_sec)
        while rclpy.ok() and time.time() < deadline:
            remaining = deadline - time.time()
            rclpy.spin_once(self, timeout_sec=max(0.0, min(0.05, remaining)))

    def run(self) -> None:
        if self.required_valid_pairs < 1:
            raise ValueError("--required-valid-pairs must be >= 1")
        if self.max_pose_commands < 1:
            raise ValueError("--max-pose-commands must be >= 1")
        if self.num_steps < 2:
            raise ValueError("--num-steps must be >= 2")
        if self.command_rate_hz <= 0.0:
            raise ValueError("--command-rate must be > 0")

        t0 = time.time()
        while rclpy.ok() and self._joint_state_msg is None and (time.time() - t0) < self.wait_feedback_sec:
            self._spin_for(0.05)
        if self._joint_state_msg is None:
            print(f"No joint feedback within {self.wait_feedback_sec:.2f}s; calibration aborted.")
            return

        q_feedback = self._extract_command_vector(self._joint_state_msg)
        if q_feedback is None:
            print("Current /joint_states do not include all commanded joints; calibration aborted.")
            return

        q_start = self.start_joints.copy() if self.start_joints is not None else q_feedback.copy()
        q_target = self.target_joints.copy()
        sweep = self._build_sweep_trajectory(q_start, q_target)

        max_step = 0.0
        for i in range(1, len(sweep)):
            step = float(np.max(np.abs(sweep[i] - sweep[i - 1])))
            max_step = max(max_step, step)
        if max_step > self.max_command_step_rad:
            raise ValueError(
                f"Per-step command delta {max_step:.4f} exceeds --max-command-step-rad {self.max_command_step_rad:.4f}"
            )

        print(f"Start={np.array2string(q_start, precision=3)}")
        print(f"Target={np.array2string(q_target, precision=3)}")
        print(
            f"Executing sweep trajectory with {len(sweep)} waypoints @ {self.command_rate_hz:.1f} Hz; "
            f"max_step={max_step:.4f} rad"
        )

        cmd_count = 0
        pose_index = 0
        sleep_sec = 1.0 / self.command_rate_hz
        while len(self.tag_transforms) < self.required_valid_pairs and rclpy.ok():
            q_cmd = sweep[pose_index % len(sweep)]
            print(
                f"Pose command {cmd_count + 1}/{self.max_pose_commands} | "
                f"valid pairs {len(self.tag_transforms)}/{self.required_valid_pairs}"
            )
            self._publish_joint_target(q_cmd)
            pose_index += 1
            cmd_count += 1
            if not rclpy.ok():
                return
            self._spin_for(sleep_sec)
            if (time.time() - self._joint_feedback_time) > self.feedback_timeout_sec:
                print(f"Stale joint feedback (> {self.feedback_timeout_sec:.2f}s); calibration aborted.")
                return
            if self.settle_sec > 0.0:
                self._spin_for(self.settle_sec)
            got_pair = self.record_data()
            if got_pair:
                print(
                    f"Collected valid pair {len(self.tag_transforms)}/{self.required_valid_pairs}"
                )
            if cmd_count >= self.max_pose_commands and len(self.tag_transforms) < self.required_valid_pairs:
                print(
                    f"Reached max pose commands ({self.max_pose_commands}) before collecting "
                    f"{self.required_valid_pairs} valid pairs."
                )
                break

        for _ in range(int(max(1, round(self.command_rate_hz * max(0.0, self.settle_sec))))):
            self._publish_joint_target(q_start)
            self._spin_for(sleep_sec)

        if len(self.tag_transforms) == 0:
            print("No calibration data collected; skipping optimization.")
            return
        if len(self.tag_transforms) < self.required_valid_pairs:
            print(
                f"Insufficient valid pairs ({len(self.tag_transforms)}/{self.required_valid_pairs}); "
                "skipping optimization."
            )
            return
        print("Running optimizer...")
        parameters = self.optimizer.calibrate_camera(self.tag_transforms, self.robot_joints)
        print("Saving calibration matrices")
        self.optimizer.save_calibration_matrices(parameters, self.camera)
        print("Done!")


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TG2 Inspirehand camera calibration (ROS 2).")
    parser.add_argument("--camera", type=str, required=True, help='Target camera: "right", "left", or "center".')
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/assets/tg2_inspirehand/urdf/tg2_with_hands_no_legs.urdf",
        help="Path to TG2 Inspirehand URDF.",
    )
    parser.add_argument("--palm-link", type=str, default="palm", help="Palm link name in the URDF.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--joint-state-topic", type=str, default="/joint_states")
    parser.add_argument("--command-topic", type=str, default="/arm/command_joint_states")
    parser.add_argument(
        "--command-joints",
        default=",".join(RIGHT_ARM_HAND_JOINTS),
        help="Comma-separated ordered joint names for direct command publishing.",
    )
    parser.add_argument(
        "--start-joints",
        default="",
        help="Optional start vector (same length/order as --command-joints). If omitted, uses current /joint_states.",
    )
    parser.add_argument(
        "--target-joints",
        required=True,
        help="Required target vector (same length/order as --command-joints).",
    )
    parser.add_argument("--num-steps", type=int, default=60, help="Interpolation points from start->target.")
    parser.add_argument("--command-rate", type=float, default=30.0, help="Command publish rate (Hz).")
    parser.add_argument(
        "--max-command-step-rad",
        type=float,
        default=0.08,
        help="Safety clamp: max per-step joint delta in radians.",
    )
    parser.add_argument(
        "--feedback-timeout-sec",
        type=float,
        default=1.0,
        help="Abort if joint feedback is older than this during execution.",
    )
    parser.add_argument(
        "--wait-feedback-sec",
        type=float,
        default=3.0,
        help="Max wait for initial /joint_states before aborting.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.0,
        help="Extra wait after each command before attempting to record a pair.",
    )
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--tag-frame-id", type=str, default="tag25h9:0")
    parser.add_argument(
        "--required-valid-pairs",
        type=int,
        default=50,
        help="Collect tag+joint pairs until this count is reached.",
    )
    parser.add_argument(
        "--max-pose-commands",
        type=int,
        default=1000,
        help="Safety cap on commanded poses while trying to gather valid pairs.",
    )

    parsed_args = parser.parse_args(remove_ros_args(args=args or sys.argv)[1:])
    if parsed_args.camera not in {"right", "left", "center"}:
        raise ValueError('Incorrect camera specification. Use "right", "left", or "center".')
    joint_names = [name.strip() for name in parsed_args.command_joints.split(",") if name.strip()]
    if not joint_names:
        raise ValueError("--command-joints resolved to an empty list")
    start_joints = (
        _parse_vec(parsed_args.start_joints, "--start-joints", len(joint_names))
        if parsed_args.start_joints
        else None
    )
    target_joints = _parse_vec(parsed_args.target_joints, "--target-joints", len(joint_names))

    rclpy.init(args=args)

    node = CameraCalibrationNode(
        camera=parsed_args.camera,
        urdf_path=parsed_args.urdf,
        palm_link=parsed_args.palm_link,
        device=parsed_args.device,
        joint_state_topic=parsed_args.joint_state_topic,
        command_topic=parsed_args.command_topic,
        command_joint_names=joint_names,
        start_joints=start_joints,
        target_joints=target_joints,
        num_steps=parsed_args.num_steps,
        command_rate_hz=parsed_args.command_rate,
        max_command_step_rad=parsed_args.max_command_step_rad,
        feedback_timeout_sec=parsed_args.feedback_timeout_sec,
        wait_feedback_sec=parsed_args.wait_feedback_sec,
        tag_frame_id=parsed_args.tag_frame_id,
        tf_topic=parsed_args.tf_topic,
        required_valid_pairs=parsed_args.required_valid_pairs,
        max_pose_commands=parsed_args.max_pose_commands,
        settle_sec=parsed_args.settle_sec,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
