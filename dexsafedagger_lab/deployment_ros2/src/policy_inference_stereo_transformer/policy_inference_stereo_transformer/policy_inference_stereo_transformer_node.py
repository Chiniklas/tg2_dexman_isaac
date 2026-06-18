#!/usr/bin/env python3
"""ROS 2 node for stereo transformer policy inference.

Inputs:
- Stereo images (left/right)
- Proprio vector as std_msgs/Float32MultiArray

Outputs:
- Action vector as std_msgs/Float32MultiArray
- Optional sensor_msgs/JointState command for feedback_control_bridge
"""

from __future__ import annotations

import os
import threading
from typing import List

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from policy_inference_stereo_transformer.policy_runtime import StereoTransformerPolicy
from policy_inference_stereo_transformer.repo_support import ensure_repo_root_on_path


RIGHT_ARM_DEFAULT_NAMES = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]


class PolicyInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_inference_stereo_transformer")

        self._lock = threading.RLock()
        self._bridge = CvBridge()
        self._throttle_log_times: dict[str, float] = {}

        self.declare_parameter("repo_root", "")
        repo_root = str(self.get_parameter("repo_root").value).strip()
        resolved_repo_root = ensure_repo_root_on_path(repo_root or None)

        self.declare_parameter("num_proprio_obs", 159)
        self.declare_parameter("num_actions", 11)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)
        self.declare_parameter("deterministic", True)
        self.declare_parameter("action_scale", 1.0)
        self._num_proprio_obs = int(self.get_parameter("num_proprio_obs").value)
        self._num_actions = int(self.get_parameter("num_actions").value)
        self._img_width = int(self.get_parameter("image_width").value)
        self._img_height = int(self.get_parameter("image_height").value)
        self._deterministic = bool(self.get_parameter("deterministic").value)
        self._action_scale = float(self.get_parameter("action_scale").value)

        self.declare_parameter(
            "cfg_path",
            str(
                resolved_repo_root
                / "dexsafedagger_lab"
                / "tasks"
                / "tg2_inspirehand"
                / "agents"
                / "rl_games_ppo_stereo_transformer.yaml"
            ),
        )
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("device", "cuda")
        cfg_path = str(self.get_parameter("cfg_path").value).strip()
        ckpt_path = str(self.get_parameter("checkpoint_path").value).strip()
        device = str(self.get_parameter("device").value).strip()
        if not cfg_path:
            cfg_path = str(
                resolved_repo_root
                / "dexsafedagger_lab"
                / "tasks"
                / "tg2_inspirehand"
                / "agents"
                / "rl_games_ppo_stereo_transformer.yaml"
            )

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Policy cfg not found: {cfg_path}")
        if ckpt_path and not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self._policy = StereoTransformerPolicy(
            cfg_path=cfg_path,
            ckpt_path=ckpt_path or None,
            img_shape=(3, self._img_height, self._img_width),
            num_proprio_obs=self._num_proprio_obs,
            num_actions=self._num_actions,
            device=device,
            num_envs=2,
        )

        if not ckpt_path:
            self.get_logger().warn(
                "policy_inference_stereo_transformer: checkpoint_path is empty; "
                "running with uninitialized/random model weights"
            )

        self.declare_parameter("left_topic", "/stereo/left/image_raw")
        self.declare_parameter("right_topic", "/stereo/right/image_raw")
        self.declare_parameter("proprio_topic", "/policy/proprio")
        self.declare_parameter("action_topic", "/policy/action")
        self._left_topic = str(self.get_parameter("left_topic").value)
        self._right_topic = str(self.get_parameter("right_topic").value)
        self._proprio_topic = str(self.get_parameter("proprio_topic").value)
        self._action_topic = str(self.get_parameter("action_topic").value)

        self.declare_parameter("publish_joint_state", True)
        self.declare_parameter("joint_command_topic", "/arm/command_joint_states")
        self.declare_parameter("joint_names", RIGHT_ARM_DEFAULT_NAMES)
        self.declare_parameter("joint_action_indices", list(range(len(RIGHT_ARM_DEFAULT_NAMES))))
        self.declare_parameter("rate", 20.0)
        self._publish_joint_state = bool(self.get_parameter("publish_joint_state").value)
        self._joint_command_topic = str(self.get_parameter("joint_command_topic").value)
        self._joint_names = self.get_parameter("joint_names").value
        if not isinstance(self._joint_names, (list, tuple)):
            self._joint_names = RIGHT_ARM_DEFAULT_NAMES
        self._joint_names = [str(name) for name in self._joint_names]
        self._joint_action_indices = self.get_parameter("joint_action_indices").value
        if not isinstance(self._joint_action_indices, (list, tuple)):
            self._joint_action_indices = list(range(len(RIGHT_ARM_DEFAULT_NAMES)))
        self._joint_action_indices = [int(idx) for idx in self._joint_action_indices]

        self._left_tensor = None
        self._right_tensor = None
        self._proprio_tensor = None

        self._action_pub = self.create_publisher(Float32MultiArray, self._action_topic, 5)
        self._joint_pub = None
        if self._publish_joint_state:
            self._joint_pub = self.create_publisher(JointState, self._joint_command_topic, 5)

        self.create_subscription(Image, self._left_topic, self._left_cb, 1)
        self.create_subscription(Image, self._right_topic, self._right_cb, 1)
        self.create_subscription(Float32MultiArray, self._proprio_topic, self._proprio_cb, 5)

        rate_hz = float(self.get_parameter("rate").value)
        period = max(1e-3, 1.0 / rate_hz)
        self._timer = self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            "policy_inference_stereo_transformer: running on "
            f"{self._policy.device}, cfg={cfg_path}, ckpt={ckpt_path or '<none>'}"
        )

    def _log_throttled(self, key: str, period_sec: float, level: str, message: str) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        last = self._throttle_log_times.get(key, -1e18)
        if (now - last) < period_sec:
            return
        self._throttle_log_times[key] = now
        logger = self.get_logger()
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warn(message)
        else:
            logger.info(message)

    def _msg_to_tensor(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self._log_throttled("cv_bridge", 2.0, "error", f"cv_bridge conversion failed: {exc}")
            return None

        if frame is None:
            return None

        if frame.shape[1] != self._img_width or frame.shape[0] != self._img_height:
            frame = cv2.resize(frame, (self._img_width, self._img_height), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self._policy.device)

    def _left_cb(self, msg: Image) -> None:
        tensor = self._msg_to_tensor(msg)
        if tensor is None:
            return
        with self._lock:
            self._left_tensor = tensor

    def _right_cb(self, msg: Image) -> None:
        tensor = self._msg_to_tensor(msg)
        if tensor is None:
            return
        with self._lock:
            self._right_tensor = tensor

    def _proprio_cb(self, msg: Float32MultiArray) -> None:
        vec = np.asarray(msg.data, dtype=np.float32)
        if vec.size != self._num_proprio_obs:
            self._log_throttled(
                "bad_proprio",
                2.0,
                "warn",
                "policy_inference_stereo_transformer: expected proprio "
                f"len={self._num_proprio_obs}, got {vec.size}",
            )
            return

        tensor = torch.from_numpy(vec).reshape(1, self._num_proprio_obs).to(self._policy.device)
        with self._lock:
            self._proprio_tensor = tensor

    def _resolve_joint_names(self, n: int) -> List[str]:
        if len(self._joint_names) == n:
            return list(self._joint_names)
        if len(RIGHT_ARM_DEFAULT_NAMES) == n:
            return list(RIGHT_ARM_DEFAULT_NAMES)
        return [f"joint_{idx}" for idx in range(n)]

    def _select_joint_positions(self, action_list: List[float]) -> List[float]:
        if not self._joint_action_indices:
            return list(action_list)

        selected = []
        for idx in self._joint_action_indices:
            if isinstance(idx, int) and 0 <= idx < len(action_list):
                selected.append(action_list[idx])
            else:
                self._log_throttled(
                    "bad_joint_index",
                    2.0,
                    "warn",
                    "policy_inference_stereo_transformer: joint_action_indices "
                    f"contains invalid index {idx}",
                )
        return selected

    def _timer_cb(self) -> None:
        with self._lock:
            left = self._left_tensor
            right = self._right_tensor
            proprio = self._proprio_tensor

        if left is None or right is None or proprio is None:
            return

        try:
            out = self._policy.step(
                proprio=proprio,
                left_img=left,
                right_img=right,
                deterministic=self._deterministic,
            )
        except Exception as exc:
            self._log_throttled("inference_failed", 1.0, "error", f"policy inference failed: {exc}")
            return

        action = out["selected_action"].detach().reshape(-1).cpu().numpy()
        if self._action_scale != 1.0:
            action = action * self._action_scale
        action_list = action.astype(np.float32).tolist()

        self._action_pub.publish(Float32MultiArray(data=action_list))

        if self._joint_pub is not None:
            joint_positions = self._select_joint_positions(action_list)
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = self._resolve_joint_names(len(joint_positions))
            cmd.position = joint_positions
            self._joint_pub.publish(cmd)


def main() -> int:
    rclpy.init()
    node = PolicyInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
