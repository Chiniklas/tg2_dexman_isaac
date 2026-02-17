#!/usr/bin/env python3
"""ROS1 node for stereo transformer policy inference.

Inputs:
- Stereo images (left/right)
- Proprio vector as std_msgs/Float32MultiArray

Outputs:
- Action vector as std_msgs/Float32MultiArray
- Optional sensor_msgs/JointState command for feedback_control_bridge
"""

from __future__ import annotations

import os
import sys
import threading
from typing import List

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from policy_runtime import StereoTransformerPolicy


RIGHT_ARM_DEFAULT_NAMES = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]


class PolicyInferenceNode:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bridge = CvBridge()

        self._ensure_repo_root_on_path()

        self._num_proprio_obs = int(rospy.get_param("~num_proprio_obs", 159))
        self._num_actions = int(rospy.get_param("~num_actions", 11))
        self._img_width = int(rospy.get_param("~image_width", 320))
        self._img_height = int(rospy.get_param("~image_height", 240))
        self._deterministic = bool(rospy.get_param("~deterministic", True))
        self._action_scale = float(rospy.get_param("~action_scale", 1.0))

        cfg_path = rospy.get_param(
            "~cfg_path",
            "/tiangong_infra_ws/ws/src/tg2_dexman_isaac/"
            "dextrah_lab/tasks/dextrah_kuka_allegro/agents/rl_games_ppo_stereo_transformer.yaml",
        )
        ckpt_path = rospy.get_param("~checkpoint_path", "").strip()
        device = rospy.get_param("~device", "cuda").strip()

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
            rospy.logwarn(
                "policy_inference_stereo_transformer: checkpoint_path is empty; "
                "running with uninitialized/random model weights"
            )

        self._left_topic = rospy.get_param("~left_topic", "/stereo/left/image_raw")
        self._right_topic = rospy.get_param("~right_topic", "/stereo/right/image_raw")
        self._proprio_topic = rospy.get_param("~proprio_topic", "/policy/proprio")
        self._action_topic = rospy.get_param("~action_topic", "/policy/action")

        self._publish_joint_state = bool(rospy.get_param("~publish_joint_state", True))
        self._joint_command_topic = rospy.get_param(
            "~joint_command_topic", "/arm/command_joint_states"
        )
        self._joint_names = rospy.get_param("~joint_names", RIGHT_ARM_DEFAULT_NAMES)
        if not isinstance(self._joint_names, list):
            self._joint_names = RIGHT_ARM_DEFAULT_NAMES
        self._joint_action_indices = rospy.get_param(
            "~joint_action_indices", list(range(len(RIGHT_ARM_DEFAULT_NAMES)))
        )
        if not isinstance(self._joint_action_indices, list):
            self._joint_action_indices = list(range(len(RIGHT_ARM_DEFAULT_NAMES)))

        self._left_tensor = None
        self._right_tensor = None
        self._proprio_tensor = None

        self._action_pub = rospy.Publisher(self._action_topic, Float32MultiArray, queue_size=5)
        self._joint_pub = None
        if self._publish_joint_state:
            self._joint_pub = rospy.Publisher(self._joint_command_topic, JointState, queue_size=5)

        rospy.Subscriber(self._left_topic, Image, self._left_cb, queue_size=1)
        rospy.Subscriber(self._right_topic, Image, self._right_cb, queue_size=1)
        rospy.Subscriber(self._proprio_topic, Float32MultiArray, self._proprio_cb, queue_size=5)

        rate_hz = float(rospy.get_param("~rate", 20.0))
        period = max(1e-3, 1.0 / rate_hz)
        self._timer = rospy.Timer(rospy.Duration(period), self._timer_cb)

        rospy.loginfo(
            "policy_inference_stereo_transformer: running on %s, cfg=%s, ckpt=%s",
            self._policy.device,
            cfg_path,
            ckpt_path or "<none>",
        )

    def _ensure_repo_root_on_path(self) -> None:
        repo_root = rospy.get_param("~repo_root", "/tiangong_infra_ws/ws/src/tg2_dexman_isaac")
        if repo_root and os.path.isdir(repo_root) and repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _msg_to_tensor(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "cv_bridge conversion failed: %s", str(exc))
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
            rospy.logwarn_throttle(
                2.0,
                "policy_inference_stereo_transformer: expected proprio len=%d, got %d",
                self._num_proprio_obs,
                vec.size,
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
                rospy.logwarn_throttle(
                    2.0,
                    "policy_inference_stereo_transformer: joint_action_indices contains invalid index %s",
                    str(idx),
                )
        return selected

    def _timer_cb(self, _event) -> None:
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
            rospy.logerr_throttle(1.0, "policy inference failed: %s", str(exc))
            return

        action = out["selected_action"].detach().reshape(-1).cpu().numpy()
        if self._action_scale != 1.0:
            action = action * self._action_scale
        action_list = action.astype(np.float32).tolist()

        self._action_pub.publish(Float32MultiArray(data=action_list))

        if self._joint_pub is not None:
            joint_positions = self._select_joint_positions(action_list)
            cmd = JointState()
            cmd.header.stamp = rospy.Time.now()
            cmd.name = self._resolve_joint_names(len(joint_positions))
            cmd.position = joint_positions
            self._joint_pub.publish(cmd)


def main() -> int:
    rospy.init_node("policy_inference_stereo_transformer")
    PolicyInferenceNode()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
