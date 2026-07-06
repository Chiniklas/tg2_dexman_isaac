#!/usr/bin/env python3
"""Execute an offline trajectory (.h5) through feedback_control_bridge.

This script reads a trajectory dataset from an H5 file and publishes it as
`sensor_msgs/JointState` on `/arm/command_joint_states`.

Runtime flow:
1) Move to default init pose (same init target as test_init_and_homing.py)
2) Hold init pose for 3 seconds
3) Wait for: input("press enter to start execution")
4) Replay offline trajectory

Default target file:
  .../deployment/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5

Working command (from a sourced ROS 2 workspace):
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30

Dry-run check:
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30 \
  --dry-run

Mid-execution interruption:
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30 \
  --pause-at-replay-step 120 \
  --hand-open-offset-ratio 0.08

Custom hand path override:
python3 src/inference_offline/tests/execute_offline_traj.py \
  --traj-file src/inference_offline/tests/teacher_policies/tr3e4tbm/traj_env_0_episode_5.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30 \
  --pause-at-replay-step 125 \
  --custom-hand-traj \
  --custom-hand-postpone-steps 15 \
  --custom-hand-pre-open-percent 60 \
  --custom-hand-post-open-percent 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
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
# Joint limits for RIGHT_ARM_HAND_JOINTS from
# assets/tg2_inspirehand/urdf/tg2_with_hands_no_legs.urdf.
JOINT_LOWER_LIMITS = np.asarray(
    [-2.96, -2.0, 0.0, -2.0, -2.9671, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = np.asarray(
    [0.0, -0.1, 2.5, 0.0, 2.9671, 0.6, 0.5, 1.1, 1.1, 1.1, 1.1, 1.2, 0.5],
    dtype=np.float64,
)

INIT_STEPS_DEFAULT = 120
INIT_HOLD_SEC_DEFAULT = 3.0
MAX_STEP_EPS = 1e-6

DEFAULT_TRAJ_FILE = str(
    Path(__file__).resolve().parent / "offline_tarjs" / "1m0lvpzs" / "traj_env_0_file_1.h5"
)


def _log_info(logger, msg: str, *args: object) -> None:
    logger.info(msg % args if args else msg)


def _log_warn(logger, msg: str, *args: object) -> None:
    logger.warning(msg % args if args else msg)


def _log_error(logger, msg: str, *args: object) -> None:
    logger.error(msg % args if args else msg)


def _interpolate(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps < 2:
        return [a.astype(np.float64), b.astype(np.float64)]
    out: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, num=steps):
        out.append(((1.0 - alpha) * a + alpha * b).astype(np.float64))
    return out


def _hand_pos_to_ratio(pos: np.ndarray) -> np.ndarray:
    lo = HAND_JOINT_LIMITS[:, 0]
    hi = HAND_JOINT_LIMITS[:, 1]
    span = np.maximum(1e-6, hi - lo)
    ratio = 1.0 - ((pos - lo) / span)
    return np.clip(ratio, 0.0, 1.0)


def _hand_ratio_to_pos(ratio: np.ndarray) -> np.ndarray:
    lo = HAND_JOINT_LIMITS[:, 0]
    hi = HAND_JOINT_LIMITS[:, 1]
    ratio = np.clip(ratio, 0.0, 1.0)
    return hi - ratio * (hi - lo)


def _sim_hand_pos_to_ratio(pos: np.ndarray) -> np.ndarray:
    """Convert sim hand joint positions to bridge/service angleRatio in [0, 1]."""
    lo = JOINT_LOWER_LIMITS[7:]
    hi = JOINT_UPPER_LIMITS[7:]
    span = np.maximum(1e-6, hi - lo)
    # Sim semantics: lo=open, hi=close. Service semantics: 1=open, 0=close.
    ratio = 1.0 - ((pos - lo) / span)
    return np.clip(ratio, 0.0, 1.0)


def _hand_ratio_to_bridge_pos(ratio: np.ndarray) -> np.ndarray:
    """Encode desired hand angleRatio into command JointState positions."""
    lo = JOINT_LOWER_LIMITS[7:]
    hi = JOINT_UPPER_LIMITS[7:]
    ratio = np.clip(ratio, 0.0, 1.0)
    return hi - ratio * (hi - lo)


def _bridge_hand_pos_to_ratio(pos: np.ndarray) -> np.ndarray:
    """Decode command JointState hand positions back to bridge angleRatio in [0, 1]."""
    lo = JOINT_LOWER_LIMITS[7:]
    hi = JOINT_UPPER_LIMITS[7:]
    span = np.maximum(1e-6, hi - lo)
    ratio = 1.0 - ((pos - lo) / span)
    return np.clip(ratio, 0.0, 1.0)


def _apply_hand_open_offsets_to_replay_path(
    replay_path: list[np.ndarray],
    offset_pre: float,
    offset_post: float,
    switch_step_1based: int,
    transition_steps: int,
) -> tuple[list[np.ndarray], int]:
    """Apply hand ratio offsets on replay waypoints, optionally switching at a replay step."""
    if transition_steps < 1:
        raise ValueError("--hand-offset-transition-steps must be >= 1.")
    if not replay_path:
        return [], 0

    eps = 1e-12
    if abs(offset_pre) <= eps and abs(offset_post) <= eps:
        return [q.copy() for q in replay_path], 0

    out: list[np.ndarray] = []
    clipped_count = 0
    for i, q in enumerate(replay_path, start=1):
        offset = offset_pre
        if switch_step_1based > 0 and i >= switch_step_1based and abs(offset_post - offset_pre) > eps:
            if i >= (switch_step_1based + transition_steps):
                offset = offset_post
            else:
                alpha = float(i - switch_step_1based + 1) / float(transition_steps)
                alpha = min(max(alpha, 0.0), 1.0)
                offset = (1.0 - alpha) * offset_pre + alpha * offset_post
        q_new = q.copy()
        ratio = _bridge_hand_pos_to_ratio(q_new[7:])
        ratio_offset = ratio + offset
        ratio_clipped = np.clip(ratio_offset, 0.0, 1.0)
        clipped_count += int(np.count_nonzero(np.abs(ratio_offset - ratio_clipped) > 1e-9))
        q_new[7:] = _hand_ratio_to_bridge_pos(ratio_clipped)
        out.append(q_new)
    return out, clipped_count


def _apply_uniform_hand_ratio_to_path(
    replay_path: list[np.ndarray],
    ratio_pre: float,
    ratio_post: float,
    switch_step_1based: int,
    transition_steps: int,
) -> tuple[list[np.ndarray], int]:
    """Overwrite hand commands with uniform angleRatio targets, optionally switching at a replay step."""
    if transition_steps < 1:
        raise ValueError("--hand-offset-transition-steps must be >= 1.")
    if not replay_path:
        return [], 0

    eps = 1e-12
    out: list[np.ndarray] = []
    clipped_count = 0
    for i, q in enumerate(replay_path, start=1):
        ratio_scalar = ratio_pre
        if switch_step_1based > 0 and i >= switch_step_1based and abs(ratio_post - ratio_pre) > eps:
            if i >= (switch_step_1based + transition_steps):
                ratio_scalar = ratio_post
            else:
                alpha = float(i - switch_step_1based + 1) / float(transition_steps)
                alpha = min(max(alpha, 0.0), 1.0)
                ratio_scalar = (1.0 - alpha) * ratio_pre + alpha * ratio_post
        ratio = np.full(6, ratio_scalar, dtype=np.float64)
        ratio_clipped = np.clip(ratio, 0.0, 1.0)
        clipped_count += int(np.count_nonzero(np.abs(ratio - ratio_clipped) > 1e-9))
        q_new = q.copy()
        q_new[7:] = _hand_ratio_to_bridge_pos(ratio_clipped)
        out.append(q_new)
    return out, clipped_count


def _apply_custom_hand_ratio_to_replay_path(
    replay_path: list[np.ndarray],
    ratio_init: float,
    ratio_pre: float,
    ratio_post: float,
    warmup_prefix_steps: int,
    switch_step_1based: int,
    transition_steps: int,
    postpone_steps: int,
) -> tuple[list[np.ndarray], int]:
    """Keep init during warmup, then blend init->pre during replay before pre->post switch.

    `postpone_steps` delays only the replay-phase init->pre transition. The later
    pre->post switch uses only transition smoothing and is not postponed.
    """
    if transition_steps < 1:
        raise ValueError("--hand-offset-transition-steps must be >= 1.")
    if postpone_steps < 0:
        raise ValueError("--custom-hand-postpone-steps must be >= 0.")
    if not replay_path:
        return [], 0

    def _base_ratio(step_1based: int) -> float:
        if step_1based <= warmup_prefix_steps:
            return ratio_init

        replay_step_1based = step_1based - warmup_prefix_steps
        if replay_step_1based <= postpone_steps:
            return ratio_init

        transition_start = postpone_steps + 1
        transition_end = postpone_steps + transition_steps
        if replay_step_1based <= transition_end and abs(ratio_pre - ratio_init) > 1e-12:
            if transition_steps == 1:
                return ratio_pre
            alpha = float(replay_step_1based - transition_start) / float(transition_steps - 1)
            alpha = min(max(alpha, 0.0), 1.0)
            return (1.0 - alpha) * ratio_init + alpha * ratio_pre
        return ratio_pre

    eps = 1e-12
    switch_base_ratio = _base_ratio(switch_step_1based) if switch_step_1based > 0 else ratio_pre
    out: list[np.ndarray] = []
    clipped_count = 0
    for i, q in enumerate(replay_path, start=1):
        ratio_scalar = _base_ratio(i)
        if switch_step_1based > 0 and i >= switch_step_1based and abs(ratio_post - switch_base_ratio) > eps:
            transition_start = switch_step_1based
            if i >= (transition_start + transition_steps):
                ratio_scalar = ratio_post
            else:
                alpha = float(i - transition_start + 1) / float(transition_steps)
                alpha = min(max(alpha, 0.0), 1.0)
                ratio_scalar = (1.0 - alpha) * switch_base_ratio + alpha * ratio_post
        ratio = np.full(6, ratio_scalar, dtype=np.float64)
        ratio_clipped = np.clip(ratio, 0.0, 1.0)
        clipped_count += int(np.count_nonzero(np.abs(ratio - ratio_clipped) > 1e-9))
        q_new = q.copy()
        q_new[7:] = _hand_ratio_to_bridge_pos(ratio_clipped)
        out.append(q_new)
    return out, clipped_count


def _percent_open_to_ratio(percent_open: float, label: str) -> float:
    if percent_open < 0.0 or percent_open > 100.0:
        raise ValueError(f"{label} must be within [0, 100], got {percent_open}")
    return float(percent_open) / 100.0


def _extract_obs_joint_targets(
    obs: np.ndarray,
    joint_start: int,
    dof: int,
) -> tuple[np.ndarray, int]:
    if joint_start < 0:
        raise ValueError("--obs-joint-start must be >= 0.")
    if dof <= 0:
        raise ValueError("dof must be > 0.")
    if obs.shape[1] < (joint_start + dof):
        raise ValueError(
            f"obs shape {obs.shape} is too small for joint slice "
            f"[{joint_start}:{joint_start + dof}]"
        )

    joint_sim = obs[:, joint_start : joint_start + dof].copy()
    lo = JOINT_LOWER_LIMITS.reshape(1, -1)
    hi = JOINT_UPPER_LIMITS.reshape(1, -1)
    clipped = np.clip(joint_sim, lo, hi)
    clipped_count = int(np.count_nonzero(np.abs(joint_sim - clipped) > 1e-9))
    return clipped, clipped_count


def _sim_joint_to_bridge_command(sim_joint_targets: np.ndarray) -> np.ndarray:
    cmd = sim_joint_targets.copy()
    hand_ratio = _sim_hand_pos_to_ratio(sim_joint_targets[:, 7:])
    cmd[:, 7:] = _hand_ratio_to_bridge_pos(hand_ratio)
    return cmd


def _max_abs_step(path: list[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    out = 0.0
    for i in range(1, len(path)):
        out = max(out, float(np.max(np.abs(path[i] - path[i - 1]))))
    return out


def _step_phase(step_idx: int, init_len: int, hold_len: int, replay_len: int) -> str:
    # step_idx is edge index between full[step_idx-1] -> full[step_idx]
    if 1 <= step_idx <= max(0, init_len - 1):
        return "init"
    if hold_len > 0 and step_idx == init_len:
        return "init->init_hold"
    if hold_len > 1 and (init_len + 1) <= step_idx <= (init_len + hold_len - 1):
        return "init_hold"
    if replay_len > 0 and step_idx == (init_len + hold_len):
        return "init_hold->replay"
    return "replay"


def _densify_path_with_step_limit(path: list[np.ndarray], max_step_rad: float) -> tuple[list[np.ndarray], int]:
    """Insert interpolation points so every consecutive step is <= max_step_rad."""
    if len(path) < 2:
        return [p.copy() for p in path], 0
    if max_step_rad <= 0.0:
        raise ValueError("--max-command-step-rad must be > 0.")

    out: list[np.ndarray] = [path[0].copy()]
    inserted = 0
    for i in range(1, len(path)):
        prev = out[-1]
        nxt = path[i]
        max_delta = float(np.max(np.abs(nxt - prev)))
        segments = max(1, int(math.ceil(max_delta / max_step_rad)))
        inserted += max(0, segments - 1)
        for k in range(1, segments + 1):
            alpha = float(k) / float(segments)
            q = ((1.0 - alpha) * prev + alpha * nxt).astype(np.float64)
            out.append(q)
    return out, inserted


def _subdivide_path(path: list[np.ndarray], substeps: int) -> tuple[list[np.ndarray], int]:
    """Uniformly split each segment into `substeps` pieces (1 means no change)."""
    if substeps < 1:
        raise ValueError("--replay-substeps must be >= 1.")
    if len(path) < 2 or substeps == 1:
        return [p.copy() for p in path], 0

    out: list[np.ndarray] = [path[0].copy()]
    inserted = 0
    for i in range(1, len(path)):
        prev = path[i - 1]
        nxt = path[i]
        inserted += substeps - 1
        for k in range(1, substeps + 1):
            alpha = float(k) / float(substeps)
            q = ((1.0 - alpha) * prev + alpha * nxt).astype(np.float64)
            out.append(q)
    return out, inserted


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
        super().__init__("execute_offline_traj")
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


def _load_traj(
    traj_file: str,
    dataset_key: str,
    batch_index: int,
    max_steps: int,
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to read trajectory files. Install it in the ROS 2 Python environment."
        ) from exc

    path = os.path.expanduser(traj_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found in '{path}'. Keys: {list(f.keys())}")
        arr = np.asarray(f[dataset_key], dtype=np.float64)

    if arr.ndim == 3:
        if batch_index < 0 or batch_index >= arr.shape[1]:
            raise ValueError(
                f"batch-index {batch_index} out of range for shape {arr.shape} (valid: 0..{arr.shape[1]-1})"
            )
        arr = arr[:, batch_index, :]
    elif arr.ndim != 2:
        raise ValueError(f"Expected dataset rank 2 or 3, got {arr.shape}")

    if max_steps > 0:
        arr = arr[: max_steps]
    if arr.shape[0] == 0:
        raise ValueError("Trajectory has zero timesteps after slicing.")

    return arr


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute H5 trajectory via /arm/command_joint_states.")
    p.add_argument("--traj-file", default=DEFAULT_TRAJ_FILE, help="Path to trajectory .h5 file.")
    p.add_argument("--dataset-key", default="obs", help="Dataset key to replay (default: obs).")
    p.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index if dataset shape is [T, B, A]. Ignored for [T, A].",
    )
    p.add_argument("--max-steps", type=int, default=0, help="Replay at most this many timesteps (0 = full).")
    p.add_argument("--command-topic", default="/arm/command_joint_states")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--rate", type=float, default=30.0, help="Publish rate (Hz).")
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=90,
        help="Interpolation points from init pose to first trajectory waypoint (0 = disabled).",
    )
    p.add_argument("--hold-sec", type=float, default=2.0, help="Hold final waypoint for this duration.")
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
    p.add_argument(
        "--obs-joint-start",
        type=int,
        default=0,
        help="Start index in obs vector for 13 joint targets.",
    )
    p.add_argument(
        "--hand-open-offset-ratio",
        type=float,
        default=0.0,
        help="Optional additive offset on mapped hand angleRatio (positive opens, negative closes).",
    )
    p.add_argument(
        "--hand-open-offset-ratio-pre",
        type=float,
        default=None,
        help="Optional pre-interruption hand ratio offset. If unset, uses --hand-open-offset-ratio.",
    )
    p.add_argument(
        "--hand-open-offset-ratio-post",
        type=float,
        default=None,
        help="Optional post-interruption hand ratio offset. If unset, uses --hand-open-offset-ratio.",
    )
    p.add_argument(
        "--hand-offset-transition-steps",
        type=int,
        default=10,
        help="Smoothing steps for pre->post hand offset change (1 = hard switch).",
    )
    p.add_argument(
        "--custom-hand-traj",
        action="store_true",
        help=(
            "Override hand commands with manual open percentages while keeping the arm trajectory from the file. "
            "Init uses --custom-hand-init-open-percent, replay uses pre/post percentages around the pause step."
        ),
    )
    p.add_argument(
        "--custom-hand-init-open-percent",
        type=float,
        default=100.0,
        help="Init/hold hand open percentage for --custom-hand-traj (0=close, 100=full open).",
    )
    p.add_argument(
        "--custom-hand-pre-open-percent",
        type=float,
        default=None,
        help="Replay hand open percentage before the pause/switch step for --custom-hand-traj.",
    )
    p.add_argument(
        "--custom-hand-post-open-percent",
        type=float,
        default=None,
        help="Replay hand open percentage from the pause/switch step onward for --custom-hand-traj.",
    )
    p.add_argument(
        "--custom-hand-postpone-steps",
        type=int,
        default=0,
        help=(
            "Delay the initial custom-hand init->pre closing by this many replay steps. "
            "Does not delay the later pre->post switch after pause."
        ),
    )
    p.add_argument(
        "--disable-smoothing",
        action="store_true",
        help="Disable replay trajectory smoothing/interpolation.",
    )
    p.add_argument(
        "--replay-substeps",
        type=int,
        default=1,
        help="Uniform replay interpolation factor (1 = none, 2 = insert 1 point per segment).",
    )
    p.add_argument(
        "--pause-at-replay-step",
        type=int,
        default=0,
        help="Pause before publishing this trajectory replay waypoint after warmup (1-based). 0 disables.",
    )
    p.add_argument(
        "--pause-at-midpoint",
        action="store_true",
        help="Pause once at the replay midpoint (before publish), then press enter to resume.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate and print stats only (no publish).")
    cli_args = remove_ros_args(args=argv or sys.argv)[1:]
    return p.parse_args(cli_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pause_at_replay_step < 0:
        raise ValueError("--pause-at-replay-step must be >= 0.")
    if args.hand_offset_transition_steps < 1:
        raise ValueError("--hand-offset-transition-steps must be >= 1.")
    if args.custom_hand_postpone_steps < 0:
        raise ValueError("--custom-hand-postpone-steps must be >= 0.")
    if args.custom_hand_traj:
        if args.custom_hand_pre_open_percent is None:
            raise ValueError("--custom-hand-pre-open-percent is required when --custom-hand-traj is set.")
        if args.custom_hand_post_open_percent is None:
            raise ValueError("--custom-hand-post-open-percent is required when --custom-hand-traj is set.")

    joint_names = RIGHT_ARM_HAND_JOINTS
    dof = len(joint_names)

    obs_traj = _load_traj(
        traj_file=args.traj_file,
        dataset_key=args.dataset_key,
        batch_index=args.batch_index,
        max_steps=args.max_steps,
    )

    traj_sim, clipped_count = _extract_obs_joint_targets(
        obs=obs_traj,
        joint_start=args.obs_joint_start,
        dof=dof,
    )
    traj = _sim_joint_to_bridge_command(traj_sim)
    source_desc = (
        f"obs[{args.obs_joint_start}:{args.obs_joint_start + dof}] "
        "joint targets -> bridge command semantics"
    )

    rclpy.init(args=argv)
    node = JointStateCommandNode(
        args.command_topic,
        args.joint_state_topic,
        joint_names,
        RIGHT_ARM_HAND_COMMAND_JOINTS,
    )
    logger = node.get_logger()

    try:
        if args.pause_at_midpoint and args.pause_at_replay_step > 0:
            _log_warn(
                logger,
                "Both --pause-at-midpoint and --pause-at-replay-step were set; using --pause-at-replay-step.",
            )

        if not node.wait_for_feedback(args.wait_feedback_sec):
            _log_error(logger, "No /joint_states received within %.2fs.", args.wait_feedback_sec)
            return 1

        q_start = node.state.q.copy()
        if q_start is None:
            _log_error(logger, "Start vector is unavailable.")
            return 1

        rate_hz = max(1e-3, args.rate)
        period_sec = 1.0 / rate_hz
        init_hold_ticks = int(math.ceil(max(0.0, INIT_HOLD_SEC_DEFAULT) * rate_hz))

        q_hand_ratio = _hand_pos_to_ratio(RIGHT_HAND_INIT_SIM_POS)
        custom_hand_clipped_count = 0
        custom_hand_ratio_init = 0.0
        custom_hand_ratio_pre = 0.0
        custom_hand_ratio_post = 0.0
        if args.custom_hand_traj:
            custom_hand_ratio_init = _percent_open_to_ratio(
                float(args.custom_hand_init_open_percent),
                "--custom-hand-init-open-percent",
            )
            custom_hand_ratio_pre = _percent_open_to_ratio(
                float(args.custom_hand_pre_open_percent),
                "--custom-hand-pre-open-percent",
            )
            custom_hand_ratio_post = _percent_open_to_ratio(
                float(args.custom_hand_post_open_percent),
                "--custom-hand-post-open-percent",
            )
            q_hand_ratio = np.full(6, custom_hand_ratio_init, dtype=np.float64)
        q_hand_init = _hand_ratio_to_pos(q_hand_ratio)
        q_init = np.concatenate([RIGHT_ARM_INIT.copy(), q_hand_init], axis=0)

        init_path = _interpolate(q_start, q_init, INIT_STEPS_DEFAULT)
        init_hold_path = [q_init.copy() for _ in range(init_hold_ticks)]

        q_first = traj[0]
        warmup = _interpolate(q_init, q_first, args.warmup_steps) if args.warmup_steps > 0 else []
        actual_replay_raw = [traj[i].astype(np.float64) for i in range(traj.shape[0])]
        replay_raw = warmup + (actual_replay_raw[1:] if warmup else actual_replay_raw)
        warmup_uniform, warmup_uniform_inserted = _subdivide_path(warmup, args.replay_substeps)
        actual_replay_uniform, actual_replay_uniform_inserted = _subdivide_path(
            actual_replay_raw,
            args.replay_substeps,
        )
        replay_uniform_inserted = warmup_uniform_inserted + actual_replay_uniform_inserted
        replay_uniform = warmup_uniform + (actual_replay_uniform[1:] if warmup_uniform else actual_replay_uniform)
        warmup_prefix_steps = len(warmup_uniform)
        if args.disable_smoothing:
            replay_path = list(replay_uniform)
            replay_inserted = 0
        else:
            warmup_path, warmup_inserted = _densify_path_with_step_limit(
                warmup_uniform,
                max_step_rad=args.max_command_step_rad,
            )
            actual_replay_path, actual_replay_inserted = _densify_path_with_step_limit(
                actual_replay_uniform,
                max_step_rad=args.max_command_step_rad,
            )
            replay_inserted = warmup_inserted + actual_replay_inserted
            replay_path = warmup_path + (actual_replay_path[1:] if warmup_path else actual_replay_path)
            warmup_prefix_steps = len(warmup_path)

        actual_replay_steps = max(0, len(replay_path) - warmup_prefix_steps)
        pause_step = args.pause_at_replay_step
        if args.pause_at_midpoint and pause_step == 0 and actual_replay_steps > 0:
            pause_step = int(math.ceil(actual_replay_steps / 2.0))
        pause_step_effective = (
            warmup_prefix_steps + pause_step
            if 1 <= pause_step <= actual_replay_steps
            else 0
        )

        offset_pre = (
            float(args.hand_open_offset_ratio_pre)
            if args.hand_open_offset_ratio_pre is not None
            else float(args.hand_open_offset_ratio)
        )
        offset_post = (
            float(args.hand_open_offset_ratio_post)
            if args.hand_open_offset_ratio_post is not None
            else float(args.hand_open_offset_ratio)
        )
        replay_path, hand_offset_clipped_count = _apply_hand_open_offsets_to_replay_path(
            replay_path=replay_path,
            offset_pre=offset_pre,
            offset_post=offset_post,
            switch_step_1based=pause_step_effective,
            transition_steps=args.hand_offset_transition_steps,
        )
        init_path, init_offset_clipped_count = _apply_hand_open_offsets_to_replay_path(
            replay_path=init_path,
            offset_pre=offset_pre,
            offset_post=offset_pre,
            switch_step_1based=0,
            transition_steps=1,
        )
        init_hold_path, init_hold_offset_clipped_count = _apply_hand_open_offsets_to_replay_path(
            replay_path=init_hold_path,
            offset_pre=offset_pre,
            offset_post=offset_pre,
            switch_step_1based=0,
            transition_steps=1,
        )
        if args.custom_hand_traj:
            if (
                abs(offset_pre) > 1e-12
                or abs(offset_post) > 1e-12
            ):
                _log_warn(
                    logger,
                    "Ignoring hand-open-offset-ratio settings because --custom-hand-traj is active.",
                )
            replay_path, custom_replay_clipped_count = _apply_custom_hand_ratio_to_replay_path(
                replay_path=replay_path,
                ratio_init=custom_hand_ratio_init,
                ratio_pre=custom_hand_ratio_pre,
                ratio_post=custom_hand_ratio_post,
                warmup_prefix_steps=warmup_prefix_steps,
                switch_step_1based=pause_step_effective,
                transition_steps=args.hand_offset_transition_steps,
                postpone_steps=args.custom_hand_postpone_steps,
            )
            init_path, custom_init_clipped_count = _apply_uniform_hand_ratio_to_path(
                replay_path=init_path,
                ratio_pre=custom_hand_ratio_init,
                ratio_post=custom_hand_ratio_init,
                switch_step_1based=0,
                transition_steps=1,
            )
            init_hold_path, custom_hold_clipped_count = _apply_uniform_hand_ratio_to_path(
                replay_path=init_hold_path,
                ratio_pre=custom_hand_ratio_init,
                ratio_post=custom_hand_ratio_init,
                switch_step_1based=0,
                transition_steps=1,
            )
            custom_hand_clipped_count = (
                custom_replay_clipped_count + custom_init_clipped_count + custom_hold_clipped_count
            )

        full = init_path + init_hold_path + replay_path

        init_max = _max_abs_step(init_path)
        hold_max = _max_abs_step(init_hold_path)
        replay_max = _max_abs_step(replay_path)
        init_to_hold = (
            float(np.max(np.abs(init_hold_path[0] - init_path[-1])))
            if init_path and init_hold_path
            else 0.0
        )
        hold_to_replay = (
            float(np.max(np.abs(replay_path[0] - init_hold_path[-1])))
            if replay_path and init_hold_path
            else 0.0
        )

        max_step = 0.0
        max_step_idx = 0
        for i in range(1, len(full)):
            step = float(np.max(np.abs(full[i] - full[i - 1])))
            if step > max_step:
                max_step = step
                max_step_idx = i

        max_step_phase = _step_phase(
            step_idx=max_step_idx,
            init_len=len(init_path),
            hold_len=len(init_hold_path),
            replay_len=len(replay_path),
        )

        _log_info(
            logger,
            "Safety delta breakdown (rad): init=%.6f, init_hold=%.6f, replay=%.6f, init->hold=%.6f, hold->replay=%.6f",
            init_max,
            hold_max,
            replay_max,
            init_to_hold,
            hold_to_replay,
        )

        if (max_step - args.max_command_step_rad) > MAX_STEP_EPS:
            _log_error(
                logger,
                "Aborting: per-step delta %.6f rad at step %d (phase=%s) exceeds "
                "--max-command-step-rad %.6f rad (tolerance=%.1e). Increase warmup-steps "
                "or replay-substeps, or raise the threshold if intentional.",
                max_step,
                max_step_idx,
                max_step_phase,
                args.max_command_step_rad,
                MAX_STEP_EPS,
            )
            return 1

        _log_info(logger, "Replaying trajectory file: %s", os.path.expanduser(args.traj_file))
        _log_info(logger, "Dataset=%s shape=%s -> dof=%d", args.dataset_key, str(tuple(obs_traj.shape)), dof)
        _log_info(logger, "Replay mapping: %s", source_desc)
        if replay_path:
            replay_hand_ratio = _bridge_hand_pos_to_ratio(np.stack([q[7:] for q in replay_path], axis=0))
            _log_info(
                logger,
                "Hand angleRatio range after mapping: min=%.3f max=%.3f",
                float(np.min(replay_hand_ratio)),
                float(np.max(replay_hand_ratio)),
            )
        if abs(offset_pre) > 1e-12 or abs(offset_post) > 1e-12:
            _log_info(
                logger,
                "Applied hand-open-offset-ratio pre=%.4f, post=%.4f (positive=open, negative=close)",
                offset_pre,
                offset_post,
            )
            if pause_step_effective > 0:
                _log_info(
                    logger,
                    "Hand offset switch step=%d/%d actual replay (%d/%d replay path; post applies from this step onward)",
                    pause_step,
                    actual_replay_steps,
                    pause_step_effective,
                    len(replay_path),
                )
                if abs(offset_post - offset_pre) > 1e-12:
                    _log_info(
                        logger,
                        "Hand offset transition smoothing: %d step(s) for pre->post ramp",
                        args.hand_offset_transition_steps,
                    )
            elif abs(offset_post - offset_pre) > 1e-12:
                _log_warn(
                    logger,
                    "Post hand offset differs from pre but no valid pause/switch step is active; using pre for all steps.",
                )
            if hand_offset_clipped_count > 0:
                _log_warn(
                    logger,
                    "Clamped %d hand ratio entries to [0,1] after applying hand-open-offset-ratio.",
                    hand_offset_clipped_count,
                )
            init_offset_total = init_offset_clipped_count + init_hold_offset_clipped_count
            if init_offset_total > 0:
                _log_warn(
                    logger,
                    "Clamped %d init/hold hand ratio entries to [0,1] after applying pre hand offset.",
                    init_offset_total,
                )
        if args.custom_hand_traj:
            _log_info(
                logger,
                "Custom hand traj active: init=%.1f%% open, pre-pause=%.1f%% open, post-pause=%.1f%% open",
                100.0 * custom_hand_ratio_init,
                100.0 * custom_hand_ratio_pre,
                100.0 * custom_hand_ratio_post,
            )
            if warmup_prefix_steps > 0:
                _log_info(
                    logger,
                    "Custom hand warmup: holding init percentage for first %d replay-path step(s)",
                    warmup_prefix_steps,
                )
            if abs(custom_hand_ratio_pre - custom_hand_ratio_init) > 1e-12:
                _log_info(
                    logger,
                    "Custom hand pre-pause: after warmup, delaying init->pre by %d replay step(s), then blending over %d step(s)",
                    args.custom_hand_postpone_steps,
                    args.hand_offset_transition_steps,
                )
            if pause_step_effective > 0:
                _log_info(
                    logger,
                    "Custom hand switch step=%d/%d actual replay (%d/%d replay path; post percentage applies from this step onward)",
                    pause_step,
                    actual_replay_steps,
                    pause_step_effective,
                    len(replay_path),
                )
                if abs(custom_hand_ratio_post - custom_hand_ratio_pre) > 1e-12:
                    _log_info(
                        logger,
                        "Custom hand transition smoothing: %d step(s) for pre->post ramp",
                        args.hand_offset_transition_steps,
                    )
            elif abs(custom_hand_ratio_post - custom_hand_ratio_pre) > 1e-12:
                _log_warn(
                    logger,
                    "Custom post hand percentage differs from pre but no valid pause/switch step is active; using pre for all replay steps.",
                )
            if custom_hand_clipped_count > 0:
                _log_warn(
                    logger,
                    "Clamped %d custom hand ratio entries to [0,1].",
                    custom_hand_clipped_count,
                )
        if clipped_count > 0:
            _log_warn(logger, "Clamped %d obs joint entries outside configured joint limits.", clipped_count)
        _log_info(logger, "Publish topic=%s, feedback topic=%s", args.command_topic, args.joint_state_topic)
        _log_info(logger, "Start=%s", np.array2string(q_start, precision=3))
        _log_info(logger, "Init enabled: init_steps=%d init_hold_sec=%.2f", INIT_STEPS_DEFAULT, INIT_HOLD_SEC_DEFAULT)
        if args.replay_substeps > 1:
            _log_info(
                logger,
                "Replay substeps enabled: %d -> %d waypoints (inserted %d, substeps=%d)",
                len(replay_raw),
                len(replay_uniform),
                replay_uniform_inserted,
                args.replay_substeps,
            )
        else:
            _log_info(logger, "Replay substeps disabled: using %d base replay waypoints", len(replay_raw))
        if args.disable_smoothing:
            _log_info(logger, "Replay smoothing disabled: using %d replay waypoints", len(replay_path))
        else:
            _log_info(
                logger,
                "Replay smoothing enabled: %d -> %d waypoints (inserted %d)",
                len(replay_uniform),
                len(replay_path),
                replay_inserted,
            )
        _log_info(logger, "First traj waypoint=%s", np.array2string(q_first, precision=3))
        if pause_step > 0:
            if pause_step > actual_replay_steps:
                _log_warn(
                    logger,
                    "Pause step %d is beyond actual replay length %d; pause will not trigger.",
                    pause_step,
                    actual_replay_steps,
                )
            else:
                _log_info(
                    logger,
                    "Replay pause configured at actual replay step %d/%d (replay-path step %d/%d, before publish).",
                    pause_step,
                    actual_replay_steps,
                    pause_step_effective,
                    len(replay_path),
                )
        _log_info(
            logger,
            "Total waypoints=%d (init=%d + init_hold=%d + replay=%d), max_step=%.6f rad",
            len(full),
            len(init_path),
            len(init_hold_path),
            len(replay_path),
            max_step,
        )

        if args.dry_run:
            try:
                input("press enter to start execution")
            except EOFError:
                _log_warn(logger, "No stdin available; continuing dry-run completion.")
            except KeyboardInterrupt:
                _log_warn(logger, "Dry-run canceled by user.")
                return 1
            _log_info(logger, "Dry-run complete. No commands were published.")
            return 0

        hold_ticks = int(math.ceil(max(0.0, args.hold_sec) * rate_hz))
        last_log = time.time()

        _log_info(logger, "Execution phase: init")
        init_exec_path = init_path + init_hold_path
        for idx, q_cmd in enumerate(init_exec_path):
            if not rclpy.ok():
                return 1
            node.publish_target(q_cmd)
            node.sleep_with_spin(period_sec)

            if node.state.q is not None and (time.time() - node.state.stamp) > args.feedback_timeout_sec:
                _log_error(logger, "Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
                return 1

            if node.state.q is not None and (time.time() - last_log) > 0.5:
                err = float(np.max(np.abs(q_cmd - node.state.q)))
                _log_info(logger, "Init waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(init_exec_path), err)
                last_log = time.time()

        try:
            input("press enter to start execution")
        except EOFError:
            _log_warn(logger, "No stdin available; continuing execution.")
        except KeyboardInterrupt:
            _log_warn(logger, "Execution canceled by user before trajectory replay.")
            return 1

        _log_info(
            logger,
            "Resuming after init pause: replay steps=%d, total command waypoints=%d",
            len(replay_path),
            len(full),
        )
        _log_info(logger, "Execution phase: replay")
        for idx, q_cmd in enumerate(replay_path):
            step_1based = idx + 1
            if pause_step_effective > 0 and step_1based == pause_step_effective:
                try:
                    input(
                        f"Paused at actual replay step {pause_step}/{actual_replay_steps} "
                        f"(path step {step_1based}/{len(replay_path)}). press enter to resume"
                    )
                except EOFError:
                    _log_warn(logger, "No stdin available at pause step; continuing replay.")
                except KeyboardInterrupt:
                    _log_warn(logger, "Execution canceled by user at pause step %d.", pause_step)
                    return 1

            if not rclpy.ok():
                return 1
            node.publish_target(q_cmd)
            node.sleep_with_spin(period_sec)

            if node.state.q is not None and (time.time() - node.state.stamp) > args.feedback_timeout_sec:
                _log_error(logger, "Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
                return 1

            if node.state.q is not None and (time.time() - last_log) > 0.5:
                err = float(np.max(np.abs(q_cmd - node.state.q)))
                _log_info(logger, "Replay waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(replay_path), err)
                last_log = time.time()

        q_goal = replay_path[-1]
        for _ in range(hold_ticks):
            if not rclpy.ok():
                return 1
            node.publish_target(q_goal)
            node.sleep_with_spin(period_sec)

        _log_info(logger, "Trajectory replay complete. Held final pose for %.2fs.", args.hold_sec)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
