#!/usr/bin/env python3
# Copyright (c) 2024, Nvidia.  All rights reserved.

"""
Camera calibration pipeline for TG2 Inspirehand (ROS1).

Moves the robot through a set of pose targets, logs (camera->tag pose, robot joints),
runs a Gauss-Newton optimizer to solve camera->robot and palm->tag transforms, and
saves the robot->camera transform to disk.
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import Iterable

import numpy as np
import torch
import warp as wp

import rospy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.utils.rotation_utils import euler_to_matrix, quaternion_to_matrix
from fabrics_sim.utils.utils import initialize_warp

from tg2_inspirehand_random_targets import build_calibration_trajectory


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


class CameraCalibrationNode:
    def __init__(
        self,
        camera: str,
        urdf_path: str,
        palm_link: str,
        device: str,
        joint_state_topic: str,
        pose_command_topic: str,
        tag_frame_id: str,
        tf_topic: str,
        home_pose: list[float],
        target_pose: list[float],
        num_steps: int,
        fabric_vel_threshold: float,
        publish_dt: float,
    ):
        self.camera = camera
        self.device = device
        initialize_warp(self.device)

        self.publish_dt = publish_dt
        self.fabric_vel_threshold = fabric_vel_threshold

        self._joint_state_msg: JointState | None = None
        self._joint_velocity: np.ndarray | None = None
        self._joint_feedback_time = time.time()

        self.tag_frame_id = tag_frame_id
        self.got_tag_info = False
        self.tag_transform = np.zeros((4, 4))
        self.tag_transform[3, 3] = 1.0

        self.optimizer = OptimizeCameraCalibration(urdf_path=urdf_path, palm_link=palm_link, device=device)

        self._pose_command: list[float] | None = None
        self._pose_pub = rospy.Publisher(pose_command_topic, JointState, queue_size=1)
        self._pose_timer = rospy.Timer(rospy.Duration(self.publish_dt), self._pose_pub_callback)

        self._joint_sub = rospy.Subscriber(joint_state_topic, JointState, self._joint_state_callback, queue_size=1)
        self._tf_sub = rospy.Subscriber(tf_topic, TFMessage, self._tf_callback, queue_size=10)

        self.tag_transforms: list[np.ndarray] = []
        self.robot_joints: list[np.ndarray] = []

        self.home_pose = torch.tensor([home_pose], device=self.device, dtype=torch.float32)
        self.target_pose = torch.tensor([target_pose], device=self.device, dtype=torch.float32)

        self.pose_targets = build_calibration_trajectory(self.home_pose, self.target_pose, num_steps)

    def _pose_pub_callback(self, _event: rospy.TimerEvent) -> None:
        if self._pose_command is None:
            return
        msg = JointState()
        msg.position = self._pose_command
        self._pose_pub.publish(msg)

    def _joint_state_callback(self, msg: JointState) -> None:
        self._joint_feedback_time = time.time()
        self._joint_state_msg = msg
        if msg.velocity:
            self._joint_velocity = np.array(msg.velocity, dtype=np.float32)

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

    def record_data(self) -> None:
        if not self.got_tag_info or self._joint_state_msg is None:
            print("Missing tag or joint state data; skipping frame.")
            return
        self.tag_transforms.append(copy.copy(self.tag_transform))
        q = self.optimizer.build_q_from_joint_state(self._joint_state_msg)
        self.robot_joints.append(q)
        self.got_tag_info = False

    def move_to_pose_target(self, pose_target: torch.Tensor, move_timeout: float) -> None:
        self._pose_command = list(pose_target[0, :].detach().cpu().numpy().astype("float"))
        start = time.time()
        while (time.time() - start) < move_timeout and not rospy.is_shutdown():
            if (time.time() - self._joint_feedback_time) > 0.1:
                print("No joint feedback; shutting down.")
                rospy.signal_shutdown("no feedback")
                return
            if self._joint_velocity is not None:
                if np.linalg.norm(self._joint_velocity) < self.fabric_vel_threshold and (time.time() - start) > 1.0:
                    break
            time.sleep(0.1)

    def run(self) -> None:
        move_timeout = 10.0
        print("Moving to home pose")
        self.move_to_pose_target(self.home_pose, move_timeout)
        if rospy.is_shutdown():
            return

        pose_index = 0
        start_index = 0
        if self.pose_targets:
            if torch.allclose(self.pose_targets[0], self.home_pose):
                start_index = 1
        for pose_target in self.pose_targets[start_index:]:
            print(f"Pose index {pose_index}")
            self.move_to_pose_target(pose_target, move_timeout)
            pose_index += 1
            if rospy.is_shutdown():
                return
            time.sleep(0.5)
            self.record_data()

        print("Moving to home pose")
        self.move_to_pose_target(self.home_pose, move_timeout)
        if rospy.is_shutdown():
            return

        if len(self.tag_transforms) == 0:
            print("No calibration data collected; skipping optimization.")
            return
        print("Running optimizer...")
        parameters = self.optimizer.calibrate_camera(self.tag_transforms, self.robot_joints)
        print("Saving calibration matrices")
        self.optimizer.save_calibration_matrices(parameters, self.camera)
        print("Done!")


def _parse_pose_arg(args: list[str], fallback: list[float]) -> list[float]:
    if args is None:
        return fallback
    if len(args) != 6:
        raise ValueError("Pose must be 6 floats: x y z yaw pitch roll (ZYX, radians).")
    return [float(v) for v in args]


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="TG2 Inspirehand camera calibration (ROS1).")
    parser.add_argument("--camera", type=str, required=True, help='Target camera: "right", "left", or "center".')
    parser.add_argument(
        "--urdf",
        type=str,
        default="/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/tg2_inspirehand/urdf/tg2_with_hands_no_legs.urdf",
        help="Path to TG2 Inspirehand URDF.",
    )
    parser.add_argument("--palm-link", type=str, default="palm", help="Palm link name in the URDF.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--joint-state-topic", type=str, default="/tg2/joint_states")
    parser.add_argument("--pose-command-topic", type=str, default="/tg2_inspirehand_fabric/pose_commands")
    parser.add_argument("--tf-topic", type=str, default="/tf")
    parser.add_argument("--tag-frame-id", type=str, default="tag25h9:0")
    parser.add_argument("--home-pose", nargs=6, help="Home pose: x y z yaw pitch roll (ZYX, radians).")
    parser.add_argument("--target-pose", nargs=6, help="Target pose: x y z yaw pitch roll (ZYX, radians).")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of interpolated setpoints.")
    parser.add_argument("--fabric-vel-threshold", type=float, default=0.05)
    parser.add_argument("--publish-dt", type=float, default=1.0 / 30.0)

    args = parser.parse_args(list(argv))
    if args.camera not in {"right", "left", "center"}:
        raise ValueError('Incorrect camera specification. Use "right", "left", or "center".')

    home_pose = _parse_pose_arg(args.home_pose, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target_pose = _parse_pose_arg(args.target_pose, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    rospy.init_node("tg2_camera_calibration")

    node = CameraCalibrationNode(
        camera=args.camera,
        urdf_path=args.urdf,
        palm_link=args.palm_link,
        device=args.device,
        joint_state_topic=args.joint_state_topic,
        pose_command_topic=args.pose_command_topic,
        tag_frame_id=args.tag_frame_id,
        tf_topic=args.tf_topic,
        home_pose=home_pose,
        target_pose=target_pose,
        num_steps=args.num_steps,
        fabric_vel_threshold=args.fabric_vel_threshold,
        publish_dt=args.publish_dt,
    )
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main([]))
