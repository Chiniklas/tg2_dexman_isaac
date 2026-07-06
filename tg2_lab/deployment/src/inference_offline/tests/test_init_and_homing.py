#!/usr/bin/env python3
"""Move right arm+hand to either homing pose or init pose and hold.

Modes:
- homing: move to power-on/start pose reference.
- init: move to hardcoded init pose from task config.

Both modes work from arbitrary current state because start is read from /joint_states
unless --start is explicitly provided.

Hand translation rule:
- Real hand command semantics are ratio in [0, 1], where 0=full close and 1=full open.
- Sim hand joint semantics use limits [min, max], where min=open and max=close.
- For init defaults taken from sim joint positions, this script converts to real ratio with:
  ratio = 1 - (sim_pos - min) / (max - min)
- Before publishing to /arm/command_joint_states (so the bridge outputs that ratio), it maps back with:
  cmd_pos = max - ratio * (max - min)

Examples:
python3 src/inference_offline/tests/test_init_and_homing.py --mode homing --steps 120 --rate 30 --hold-sec 3
python3 src/inference_offline/tests/test_init_and_homing.py --mode init --steps 120 --rate 30 --hold-sec 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState


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

RIGHT_ARM_HAND_COMMAND_JOINTS = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
    "right_little_1_joint",
    "right_ring_1_joint",
    "right_middle_1_joint",
    "right_index_1_joint",
    "right_thumb_1_joint",
    "right_thumb_2_joint",
]

JOINT_NAME_ALIASES = {
    "little_joint_0": ("little_joint_0", "right_little_1_joint"),
    "ring_joint_0": ("ring_joint_0", "right_ring_1_joint"),
    "middle_joint_0": ("middle_joint_0", "right_middle_1_joint"),
    "index_joint_0": ("index_joint_0", "right_index_1_joint"),
    "thumb_joint_0": ("thumb_joint_0", "right_thumb_1_joint"),
    "thumb_joint_1": ("thumb_joint_1", "right_thumb_2_joint"),
}

RIGHT_ARM_INIT = np.asarray(
    [-1.570796, -0.523599, 1.108284, -1.275836, 0.089012, -0.027925, -0.048869],
    dtype=np.float64,
)
RIGHT_ARM_HOMING = np.asarray(
    [-0.07198, -0.04250, 0.04053, -0.15222, 0.0, 0.05065, 0.00471],
    dtype=np.float64,
)
HAND_JOINT_LIMITS = np.asarray(
    [
        [0.0, 1.1],  # little_joint_0
        [0.0, 1.1],  # ring_joint_0
        [0.0, 1.1],  # middle_joint_0
        [0.0, 1.1],  # index_joint_0
        [0.3, 1.2],  # thumb_joint_0
        [0.0, 0.5],  # thumb_joint_1
    ],
    dtype=np.float64,
)
RIGHT_HAND_INIT_SIM_POS = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.4, 0.1],
    dtype=np.float64,
)
RIGHT_HAND_HOMING_RATIO = np.asarray(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float64,
)


def _log_info(logger, msg: str, *args: object) -> None:
    logger.info(msg % args if args else msg)


def _log_error(logger, msg: str, *args: object) -> None:
    logger.error(msg % args if args else msg)


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


def _hand_pos_to_ratio(pos: np.ndarray) -> np.ndarray:
    lo = HAND_JOINT_LIMITS[:, 0]
    hi = HAND_JOINT_LIMITS[:, 1]
    span = np.maximum(1e-6, hi - lo)
    # Sim joint-space uses min=open, max=close; real ratio uses 0=close, 1=open.
    ratio = 1.0 - ((pos - lo) / span)
    return np.clip(ratio, 0.0, 1.0)


def _hand_ratio_to_pos(ratio: np.ndarray) -> np.ndarray:
    lo = HAND_JOINT_LIMITS[:, 0]
    hi = HAND_JOINT_LIMITS[:, 1]
    ratio = np.clip(ratio, 0.0, 1.0)
    # Bridge command semantics use joint-space min=open, max=close.
    return hi - ratio * (hi - lo)


@dataclass
class FeedbackState:
    stamp: float = 0.0
    q: Optional[np.ndarray] = None


class JointStateCommandNode(Node):
    def __init__(
        self,
        command_topic: str,
        joint_state_topic: str,
        joint_names: list[str],
        command_joint_names: list[str],
    ) -> None:
        super().__init__("test_init_and_homing")
        self._joint_names = joint_names
        self._command_joint_names = command_joint_names
        self.state = FeedbackState()
        self._pub = self.create_publisher(JointState, command_topic, 10)
        self._sub = self.create_subscription(JointState, joint_state_topic, self._cb, 10)

    def _cb(self, msg: JointState) -> None:
        idx = {name: i for i, name in enumerate(msg.name)}
        q = np.zeros(len(self._joint_names), dtype=np.float64)
        for j, name in enumerate(self._joint_names):
            aliases = JOINT_NAME_ALIASES.get(name, (name,))
            i = None
            for alias in aliases:
                candidate = idx.get(alias)
                if candidate is not None:
                    i = candidate
                    break
            if i is None or i >= len(msg.position):
                return
            q[j] = float(msg.position[i])
        self.state = FeedbackState(stamp=time.time(), q=q)

    def publish_target(self, q_cmd: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._command_joint_names)
        msg.position = q_cmd.astype(float).tolist()
        msg.velocity = []
        msg.effort = []
        self._pub.publish(msg)

    def wait_for_feedback(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while rclpy.ok() and self.state.q is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))
        return self.state.q is not None

    def sleep_with_spin(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            rclpy.spin_once(self, timeout_sec=min(0.05, remaining))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move right arm+hand to hardcoded homing/init pose and hold.")
    p.add_argument("--command-topic", default="/arm/command_joint_states")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument(
        "--mode",
        choices=["homing", "init"],
        default="homing",
        help="Target mode: homing uses power-on pose; init uses task init pose (arm+hand).",
    )
    p.add_argument(
        "--homing-pose",
        default="",
        help="Optional 7-value override for --mode homing. Use --homing-pose=<v1,...,v7>.",
    )
    p.add_argument(
        "--init-pose",
        default="",
        help="Optional 7-value override for --mode init. Use --init-pose=<v1,...,v7>.",
    )
    p.add_argument(
        "--homing-hand",
        default="",
        help="Optional 6-value override for --mode homing hand ratio target in [0,1] (little,ring,middle,index,thumb0,thumb1).",
    )
    p.add_argument(
        "--init-hand",
        default="",
        help="Optional 6-value override for --mode init hand ratio target in [0,1] (little,ring,middle,index,thumb0,thumb1).",
    )
    p.add_argument(
        "--start",
        default="",
        help="Optional start joint vector (13 values). If omitted, uses latest /joint_states.",
    )
    p.add_argument("--steps", type=int, default=120, help="Interpolation points from start->goal.")
    p.add_argument("--rate", type=float, default=30.0, help="Command publish rate (Hz).")
    p.add_argument("--hold-sec", type=float, default=3.0, help="Hold time at goal pose.")
    p.add_argument(
        "--wait-feedback-sec",
        type=float,
        default=3.0,
        help="Max wait for initial /joint_states before aborting.",
    )
    p.add_argument(
        "--feedback-timeout-sec",
        type=float,
        default=1.0,
        help="Abort if joint feedback is older than this during execution.",
    )
    p.add_argument(
        "--max-command-step-rad",
        type=float,
        default=0.08,
        help="Safety clamp: max per-step joint delta in radians.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print trajectory but do not publish.")
    cli_args = remove_ros_args(args=argv or sys.argv)[1:]
    return p.parse_args(cli_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    joint_names = RIGHT_ARM_HAND_JOINTS
    dof = len(joint_names)

    rclpy.init(args=argv)
    node = JointStateCommandNode(
        args.command_topic,
        args.joint_state_topic,
        joint_names,
        RIGHT_ARM_HAND_COMMAND_JOINTS,
    )
    logger = node.get_logger()

    try:
        if not args.start and not node.wait_for_feedback(args.wait_feedback_sec):
            _log_error(logger, "No /joint_states received within %.2fs, and --start not provided.", args.wait_feedback_sec)
            return 1

        q_start = _parse_vec(args.start, "--start", dof) if args.start else node.state.q.copy()
        if q_start is None:
            _log_error(logger, "Start vector is unavailable.")
            return 1

        if args.mode == "init":
            q_arm_goal = _parse_vec(args.init_pose, "--init-pose", 7) if args.init_pose else RIGHT_ARM_INIT.copy()
            q_hand_ratio = (
                _parse_vec(args.init_hand, "--init-hand", 6)
                if args.init_hand
                else _hand_pos_to_ratio(RIGHT_HAND_INIT_SIM_POS)
            )
            goal_label = "init"
        else:
            q_arm_goal = (
                _parse_vec(args.homing_pose, "--homing-pose", 7) if args.homing_pose else RIGHT_ARM_HOMING.copy()
            )
            q_hand_ratio = (
                _parse_vec(args.homing_hand, "--homing-hand", 6)
                if args.homing_hand
                else RIGHT_HAND_HOMING_RATIO.copy()
            )
            goal_label = "homing"

        q_hand_goal = _hand_ratio_to_pos(q_hand_ratio)
        q_goal = np.concatenate([q_arm_goal, q_hand_goal], axis=0)
        full = _interpolate(q_start, q_goal, args.steps)

        max_step = 0.0
        for i in range(1, len(full)):
            step = float(np.max(np.abs(full[i] - full[i - 1])))
            max_step = max(max_step, step)
        if max_step > args.max_command_step_rad:
            _log_error(
                logger,
                "Aborting: per-step delta %.4f rad exceeds --max-command-step-rad %.4f rad.",
                max_step,
                args.max_command_step_rad,
            )
            return 1

        _log_info(logger, "Moving right arm+hand (%d DoF) via %s", dof, args.command_topic)
        _log_info(logger, "Start=%s", np.array2string(q_start, precision=3))
        _log_info(logger, "Goal (%s)=%s", goal_label, np.array2string(q_goal, precision=3))
        _log_info(
            logger,
            "Hand ratio goal (%s)=%s",
            goal_label,
            np.array2string(np.clip(q_hand_ratio, 0.0, 1.0), precision=3),
        )
        _log_info(logger, "Waypoints=%d max_step=%.4f rad", len(full), max_step)

        if args.dry_run:
            return 0

        period_sec = 1.0 / max(1e-3, args.rate)
        hold_ticks = int(math.ceil(max(0.0, args.hold_sec) * args.rate))
        last_log = time.time()

        for idx, q_cmd in enumerate(full):
            if not rclpy.ok():
                return 1
            node.publish_target(q_cmd)
            node.sleep_with_spin(period_sec)

            if node.state.q is not None and (time.time() - node.state.stamp) > args.feedback_timeout_sec:
                _log_error(logger, "Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
                return 1

            if node.state.q is not None and (time.time() - last_log) > 0.5:
                err = float(np.max(np.abs(q_cmd - node.state.q)))
                _log_info(logger, "Waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(full), err)
                last_log = time.time()

        for _ in range(hold_ticks):
            if not rclpy.ok():
                return 1
            node.publish_target(q_goal)
            node.sleep_with_spin(period_sec)

        _log_info(logger, "Move complete (mode=%s, held for %.2fs).", goal_label, args.hold_sec)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
