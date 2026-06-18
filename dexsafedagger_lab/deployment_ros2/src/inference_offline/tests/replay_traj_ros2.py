#!/usr/bin/env python3
"""Replay a runtime-recorded H5 trajectory through feedback_control_bridge.

This script is the runtime counterpart to ``record_traj_ros2.py``.  It expects
an HDF5 trajectory whose ``obs`` or ``joint_position`` dataset is a [T, 13]
joint-position array in the recorder's canonical right arm + hand order, then
publishes those waypoints as ``sensor_msgs/msg/JointState`` commands.

Typical usage from a sourced ROS 2 workspace:

python3 src/inference_offline/tests/replay_traj_ros2.py \
  --traj-file src/inference_offline/tests/runtime_trajs/runtime_traj_YYYYMMDD_HHMMSS.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
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

JOINT_LOWER_LIMITS = np.asarray(
    [-2.96, -2.0, 0.0, -2.0, -2.9671, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = np.asarray(
    [0.0, -0.1, 2.5, 0.0, 2.9671, 0.6, 0.5, 1.1, 1.1, 1.1, 1.1, 1.2, 0.5],
    dtype=np.float64,
)

DEFAULT_TRAJ_FILE = str(Path(__file__).resolve().parent / "runtime_trajs" / "runtime_traj_latest.h5")


@dataclass
class FeedbackState:
    stamp: float = 0.0
    q: np.ndarray | None = None


def _log_info(logger, msg: str, *args: object) -> None:
    logger.info(msg % args if args else msg)


def _log_warn(logger, msg: str, *args: object) -> None:
    logger.warning(msg % args if args else msg)


def _log_error(logger, msg: str, *args: object) -> None:
    logger.error(msg % args if args else msg)


def _parse_joint_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--joint-names must contain at least one joint name.")
    return names


def _command_joint_names_for(joint_names: list[str]) -> list[str]:
    out: list[str] = []
    for name in joint_names:
        if name == "little_joint_0":
            out.append("right_little_1_joint")
        elif name == "ring_joint_0":
            out.append("right_ring_1_joint")
        elif name == "middle_joint_0":
            out.append("right_middle_1_joint")
        elif name == "index_joint_0":
            out.append("right_index_1_joint")
        elif name == "thumb_joint_0":
            out.append("right_thumb_1_joint")
        elif name == "thumb_joint_1":
            out.append("right_thumb_2_joint")
        else:
            out.append(name)
    return out


def _load_traj(path: str, dataset_key: str, batch_index: int, max_steps: int) -> np.ndarray:
    h5_path = os.path.expanduser(path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Trajectory file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        key = dataset_key
        if key not in f:
            fallback = "joint_position" if key == "obs" else "obs"
            if fallback in f:
                key = fallback
            else:
                raise KeyError(f"Dataset '{dataset_key}' not found in '{h5_path}'. Keys: {list(f.keys())}")
        arr = np.asarray(f[key], dtype=np.float64)

    if arr.ndim == 3:
        if batch_index < 0 or batch_index >= arr.shape[1]:
            raise ValueError(
                f"--batch-index {batch_index} out of range for dataset shape {arr.shape}"
            )
        arr = arr[:, batch_index, :]
    elif arr.ndim != 2:
        raise ValueError(f"Expected trajectory dataset rank 2 or 3, got shape {arr.shape}")

    if max_steps > 0:
        arr = arr[:max_steps]
    if arr.shape[0] == 0:
        raise ValueError("Trajectory has zero timesteps after slicing.")
    return arr


def _slice_joint_targets(traj: np.ndarray, joint_start: int, dof: int, clip: bool) -> tuple[np.ndarray, int]:
    if joint_start < 0:
        raise ValueError("--obs-joint-start must be >= 0.")
    if traj.shape[1] < joint_start + dof:
        raise ValueError(f"Dataset shape {traj.shape} is too small for slice [{joint_start}:{joint_start + dof}]")

    q = traj[:, joint_start : joint_start + dof].copy()
    if not clip:
        return q, 0

    lo = JOINT_LOWER_LIMITS[:dof].reshape(1, -1)
    hi = JOINT_UPPER_LIMITS[:dof].reshape(1, -1)
    q_clip = np.clip(q, lo, hi)
    clipped_count = int(np.count_nonzero(np.abs(q - q_clip) > 1e-9))
    return q_clip, clipped_count


def _interpolate(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        return []
    if steps == 1:
        return [b.astype(np.float64)]
    return [((1.0 - alpha) * a + alpha * b).astype(np.float64) for alpha in np.linspace(0.0, 1.0, steps)]


def _max_abs_step(path: list[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return max(float(np.max(np.abs(path[i] - path[i - 1]))) for i in range(1, len(path)))


class JointStateReplayNode(Node):
    def __init__(
        self,
        command_topic: str,
        joint_state_topic: str,
        joint_names: list[str],
        command_joint_names: list[str],
    ) -> None:
        super().__init__("runtime_traj_replay")
        self._joint_names = list(joint_names)
        self._command_joint_names = list(command_joint_names)
        self.state = FeedbackState()
        self._pub = self.create_publisher(JointState, command_topic, 10)
        self._sub = self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)

    def _joint_state_cb(self, msg: JointState) -> None:
        idx = {name: i for i, name in enumerate(msg.name)}
        q = np.zeros(len(self._joint_names), dtype=np.float64)
        for out_idx, joint_name in enumerate(self._joint_names):
            aliases = JOINT_NAME_ALIASES.get(joint_name, (joint_name,))
            src_idx = None
            for alias in aliases:
                candidate = idx.get(alias)
                if candidate is not None:
                    src_idx = candidate
                    break
            if src_idx is None or src_idx >= len(msg.position):
                return
            q[out_idx] = float(msg.position[src_idx])
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
            rclpy.spin_once(self, timeout_sec=min(0.02, remaining))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a runtime-recorded H5 trajectory through feedback_control_bridge.")
    p.add_argument("--traj-file", default=DEFAULT_TRAJ_FILE, help="Path to recorded .h5 trajectory.")
    p.add_argument("--dataset-key", default="obs", help="Dataset to replay. Defaults to obs.")
    p.add_argument("--batch-index", type=int, default=0, help="Batch index for rank-3 datasets.")
    p.add_argument("--max-steps", type=int, default=0, help="Replay at most this many waypoints. 0 = all.")
    p.add_argument("--obs-joint-start", type=int, default=0, help="Start index for joint targets in the dataset.")
    p.add_argument("--command-topic", default="/arm/command_joint_states")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--rate", type=float, default=30.0, help="Replay publish rate in Hz.")
    p.add_argument("--warmup-steps", type=int, default=60, help="Interpolate from live feedback to first waypoint.")
    p.add_argument("--hold-sec", type=float, default=0.5, help="Hold first waypoint after warmup before replay.")
    p.add_argument("--final-hold-sec", type=float, default=0.5, help="Hold final waypoint after replay.")
    p.add_argument("--wait-feedback-sec", type=float, default=3.0, help="Wait for initial /joint_states feedback.")
    p.add_argument("--feedback-timeout-sec", type=float, default=1.0, help="Abort if feedback becomes stale.")
    p.add_argument("--no-feedback-required", action="store_true", help="Replay without waiting for feedback.")
    p.add_argument("--no-clip", action="store_true", help="Do not clip targets to known joint limits.")
    p.add_argument("--dry-run", action="store_true", help="Load and validate trajectory without publishing.")
    p.add_argument(
        "--joint-names",
        default=",".join(RIGHT_ARM_HAND_JOINTS),
        help="Comma-separated canonical joint names matching the saved vector order.",
    )
    cli_args = remove_ros_args(args=argv or sys.argv)[1:]
    return p.parse_args(cli_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rate <= 0.0:
        raise ValueError("--rate must be > 0.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")
    if args.hold_sec < 0.0 or args.final_hold_sec < 0.0:
        raise ValueError("--hold-sec and --final-hold-sec must be >= 0.")
    if args.feedback_timeout_sec < 0.0:
        raise ValueError("--feedback-timeout-sec must be >= 0.")

    joint_names = _parse_joint_names(args.joint_names)
    command_joint_names = _command_joint_names_for(joint_names)
    traj_raw = _load_traj(args.traj_file, args.dataset_key, args.batch_index, args.max_steps)
    traj, clipped_count = _slice_joint_targets(
        traj=traj_raw,
        joint_start=args.obs_joint_start,
        dof=len(joint_names),
        clip=not args.no_clip,
    )

    period_sec = 1.0 / float(args.rate)
    replay_path = [q.astype(np.float64) for q in traj]

    if args.dry_run:
        print(f"Trajectory: {os.path.expanduser(args.traj_file)}")
        print(f"Dataset shape: {traj_raw.shape}")
        print(f"Replay shape: {traj.shape}")
        print(f"Joint order: {joint_names}")
        print(f"Command names: {command_joint_names}")
        print(f"Rate: {args.rate:.3f} Hz")
        print(f"Max abs replay step: {_max_abs_step(replay_path):.6f} rad")
        print(f"Clipped entries: {clipped_count}")
        return 0

    rclpy.init(args=argv)
    node = JointStateReplayNode(
        command_topic=args.command_topic,
        joint_state_topic=args.joint_state_topic,
        joint_names=joint_names,
        command_joint_names=command_joint_names,
    )
    logger = node.get_logger()

    try:
        _log_info(logger, "Loaded %d waypoints from %s", len(replay_path), os.path.expanduser(args.traj_file))
        _log_info(logger, "Publish topic=%s, feedback topic=%s", args.command_topic, args.joint_state_topic)
        _log_info(logger, "Joint order=%s", joint_names)
        if clipped_count > 0:
            _log_warn(logger, "Clipped %d trajectory values to known joint limits.", clipped_count)

        warmup_path: list[np.ndarray] = []
        if not args.no_feedback_required:
            if not node.wait_for_feedback(args.wait_feedback_sec):
                _log_error(logger, "No complete /joint_states feedback within %.2fs.", args.wait_feedback_sec)
                return 1
            assert node.state.q is not None
            warmup_path = _interpolate(node.state.q, replay_path[0], args.warmup_steps)
        elif args.warmup_steps > 0:
            _log_warn(logger, "--warmup-steps ignored because --no-feedback-required is set.")

        for idx, q in enumerate(warmup_path, start=1):
            node.publish_target(q)
            if node.state.q is not None and (time.time() - node.state.stamp) > args.feedback_timeout_sec:
                _log_error(logger, "Stale feedback during warmup (> %.2fs).", args.feedback_timeout_sec)
                return 1
            if idx == 1 or idx == len(warmup_path) or idx % max(1, len(warmup_path) // 5) == 0:
                _log_info(logger, "Warmup waypoint %d/%d", idx, len(warmup_path))
            node.sleep_with_spin(period_sec)

        if args.hold_sec > 0.0:
            node.publish_target(replay_path[0])
            node.sleep_with_spin(args.hold_sec)

        for idx, q in enumerate(replay_path, start=1):
            node.publish_target(q)
            if (
                not args.no_feedback_required
                and node.state.q is not None
                and (time.time() - node.state.stamp) > args.feedback_timeout_sec
            ):
                _log_error(logger, "Stale feedback during replay (> %.2fs).", args.feedback_timeout_sec)
                return 1
            if idx == 1 or idx == len(replay_path) or idx % max(1, len(replay_path) // 10) == 0:
                _log_info(logger, "Replay waypoint %d/%d", idx, len(replay_path))
            node.sleep_with_spin(period_sec)

        if args.final_hold_sec > 0.0:
            node.publish_target(replay_path[-1])
            node.sleep_with_spin(args.final_hold_sec)

        _log_info(logger, "Replay complete.")
        return 0
    except KeyboardInterrupt:
        _log_warn(logger, "Replay interrupted by user.")
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
