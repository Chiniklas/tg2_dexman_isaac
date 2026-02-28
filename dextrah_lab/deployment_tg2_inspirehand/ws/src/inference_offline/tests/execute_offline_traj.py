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
  /home/chi/projects/tg2_dexman_isaac/dextrah_lab/deployment_tg2_inspirehand/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5

Working command (inside tiangong-noetic-ws container):
python3 /tiangong_infra_ws/ws/src/inference_offline/tests/execute_offline_traj.py \
--traj-file /tiangong_infra_ws/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
--dataset-key obs \
--obs-joint-start 0 \
--rate 30

Dry-run check:
  python3 /tiangong_infra_ws/ws/src/inference_offline/tests/execute_offline_traj.py \
    --traj-file /tiangong_infra_ws/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
    --dataset-key obs \
    --obs-joint-start 0 \
    --rate 30 \
    --dry-run

midtime interruption:
python3 /tiangong_infra_ws/ws/src/inference_offline/tests/execute_offline_traj.py \
  --traj-file /tiangong_infra_ws/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key obs \
  --obs-joint-start 0 \
  --rate 30 \
  --pause-at-replay-step 120

"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
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

DEFAULT_TRAJ_FILE = (
    "/home/chi/projects/tg2_dexman_isaac/dextrah_lab/deployment_tg2_inspirehand/ws/src/"
    "inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5"
)


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
    return lo + ratio * (hi - lo)


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
    return lo + ratio * (hi - lo)


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


class JointStateWatcher:
    def __init__(self, topic: str, joint_names: list[str]) -> None:
        self._joint_names = joint_names
        self.state = FeedbackState()
        self._sub = rospy.Subscriber(topic, JointState, self._cb, queue_size=10)

    def _cb(self, msg: JointState) -> None:
        idx = {name: i for i, name in enumerate(msg.name)}
        q = np.zeros(len(self._joint_names), dtype=np.float64)
        for j, name in enumerate(self._joint_names):
            i = idx.get(name)
            if i is None or i >= len(msg.position):
                return
            q[j] = float(msg.position[i])
        self.state = FeedbackState(stamp=time.time(), q=q)


def _publish_target(pub: rospy.Publisher, joint_names: list[str], q_cmd: np.ndarray) -> None:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = list(joint_names)
    msg.position = q_cmd.astype(float).tolist()
    msg.velocity = []
    msg.effort = []
    pub.publish(msg)


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
            "h5py is required to read trajectory files. Install it in the ROS python environment."
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


def parse_args() -> argparse.Namespace:
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
        help="Interpolation points from init pose to first trajectory waypoint.",
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
        help="Pause before publishing this replay waypoint (1-based). 0 disables.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate and print stats only (no publish).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.pause_at_replay_step < 0:
        raise ValueError("--pause-at-replay-step must be >= 0.")

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
    hand_ratio = _sim_hand_pos_to_ratio(traj_sim[:, 7:])
    traj = _sim_joint_to_bridge_command(traj_sim)
    source_desc = (
        f"obs[{args.obs_joint_start}:{args.obs_joint_start + dof}] "
        "joint targets -> bridge command semantics"
    )

    rospy.init_node("execute_offline_traj", anonymous=True)
    pub = rospy.Publisher(args.command_topic, JointState, queue_size=10)
    watcher = JointStateWatcher(args.joint_state_topic, joint_names)

    t0 = time.time()
    while not rospy.is_shutdown() and watcher.state.q is None and (time.time() - t0) < args.wait_feedback_sec:
        time.sleep(0.05)
    if watcher.state.q is None:
        rospy.logerr("No /joint_states received within %.2fs.", args.wait_feedback_sec)
        return 1

    q_start = watcher.state.q.copy()
    if q_start is None:
        rospy.logerr("Start vector is unavailable.")
        return 1

    rate_hz = max(1e-3, args.rate)
    init_hold_ticks = int(math.ceil(max(0.0, INIT_HOLD_SEC_DEFAULT) * rate_hz))

    q_hand_ratio = _hand_pos_to_ratio(RIGHT_HAND_INIT_SIM_POS)
    q_hand_init = _hand_ratio_to_pos(q_hand_ratio)
    q_init = np.concatenate([RIGHT_ARM_INIT.copy(), q_hand_init], axis=0)

    init_path = _interpolate(q_start, q_init, INIT_STEPS_DEFAULT)
    init_hold_path = [q_init.copy() for _ in range(init_hold_ticks)]

    q_first = traj[0]
    warmup = _interpolate(q_init, q_first, max(0, args.warmup_steps))
    traj_tail = [traj[i].astype(np.float64) for i in range(1, traj.shape[0])]
    replay_raw = warmup + traj_tail
    replay_uniform, replay_uniform_inserted = _subdivide_path(replay_raw, args.replay_substeps)
    if args.disable_smoothing:
        replay_path = list(replay_uniform)
        replay_inserted = 0
    else:
        replay_path, replay_inserted = _densify_path_with_step_limit(
            replay_uniform,
            max_step_rad=args.max_command_step_rad,
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

    rospy.loginfo(
        "Safety delta breakdown (rad): init=%.6f, init_hold=%.6f, replay=%.6f, init->hold=%.6f, hold->replay=%.6f",
        init_max,
        hold_max,
        replay_max,
        init_to_hold,
        hold_to_replay,
    )

    if (max_step - args.max_command_step_rad) > MAX_STEP_EPS:
        rospy.logerr(
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

    rospy.loginfo("Replaying trajectory file: %s", os.path.expanduser(args.traj_file))
    rospy.loginfo("Dataset=%s shape=%s -> dof=%d", args.dataset_key, str(tuple(obs_traj.shape)), dof)
    rospy.loginfo("Replay mapping: %s", source_desc)
    rospy.loginfo(
        "Hand angleRatio range after mapping: min=%.3f max=%.3f",
        float(np.min(hand_ratio)),
        float(np.max(hand_ratio)),
    )
    if clipped_count > 0:
        rospy.logwarn("Clamped %d obs joint entries outside configured joint limits.", clipped_count)
    rospy.loginfo("Publish topic=%s, feedback topic=%s", args.command_topic, args.joint_state_topic)
    rospy.loginfo("Start=%s", np.array2string(q_start, precision=3))
    rospy.loginfo("Init enabled: init_steps=%d init_hold_sec=%.2f", INIT_STEPS_DEFAULT, INIT_HOLD_SEC_DEFAULT)
    if args.replay_substeps > 1:
        rospy.loginfo(
            "Replay substeps enabled: %d -> %d waypoints (inserted %d, substeps=%d)",
            len(replay_raw),
            len(replay_uniform),
            replay_uniform_inserted,
            args.replay_substeps,
        )
    else:
        rospy.loginfo("Replay substeps disabled: using %d base replay waypoints", len(replay_raw))
    if args.disable_smoothing:
        rospy.loginfo("Replay smoothing disabled: using %d replay waypoints", len(replay_path))
    else:
        rospy.loginfo(
            "Replay smoothing enabled: %d -> %d waypoints (inserted %d)",
            len(replay_uniform),
            len(replay_path),
            replay_inserted,
        )
    rospy.loginfo("First traj waypoint=%s", np.array2string(q_first, precision=3))
    if args.pause_at_replay_step > 0:
        if args.pause_at_replay_step > len(replay_path):
            rospy.logwarn(
                "Pause step %d is beyond replay length %d; pause will not trigger.",
                args.pause_at_replay_step,
                len(replay_path),
            )
        else:
            rospy.loginfo(
                "Replay pause configured at step %d/%d (before publish).",
                args.pause_at_replay_step,
                len(replay_path),
            )
    rospy.loginfo(
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
            rospy.logwarn("No stdin available; continuing dry-run completion.")
        except KeyboardInterrupt:
            rospy.logwarn("Dry-run canceled by user.")
            return 1
        rospy.loginfo("Dry-run complete. No commands were published.")
        return 0

    rate = rospy.Rate(rate_hz)
    hold_ticks = int(math.ceil(max(0.0, args.hold_sec) * rate_hz))
    last_log = time.time()

    rospy.loginfo("Execution phase: init")
    init_exec_path = init_path + init_hold_path
    for idx, q_cmd in enumerate(init_exec_path):
        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_cmd)

        if watcher.state.q is not None and (time.time() - watcher.state.stamp) > args.feedback_timeout_sec:
            rospy.logerr("Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
            return 1

        if watcher.state.q is not None and (time.time() - last_log) > 0.5:
            err = float(np.max(np.abs(q_cmd - watcher.state.q)))
            rospy.loginfo("Init waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(init_exec_path), err)
            last_log = time.time()
        rate.sleep()

    try:
        input("press enter to start execution")
    except EOFError:
        rospy.logwarn("No stdin available; continuing execution.")
    except KeyboardInterrupt:
        rospy.logwarn("Execution canceled by user before trajectory replay.")
        return 1

    rospy.loginfo(
        "Resuming after init pause: replay steps=%d, total command waypoints=%d",
        len(replay_path),
        len(full),
    )
    rospy.loginfo("Execution phase: replay")
    for idx, q_cmd in enumerate(replay_path):
        step_1based = idx + 1
        if args.pause_at_replay_step > 0 and step_1based == args.pause_at_replay_step:
            try:
                input(f"Paused at replay step {step_1based}/{len(replay_path)}. press enter to resume")
            except EOFError:
                rospy.logwarn("No stdin available at pause step; continuing replay.")
            except KeyboardInterrupt:
                rospy.logwarn("Execution canceled by user at pause step %d.", step_1based)
                return 1

        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_cmd)

        if watcher.state.q is not None and (time.time() - watcher.state.stamp) > args.feedback_timeout_sec:
            rospy.logerr("Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
            return 1

        if watcher.state.q is not None and (time.time() - last_log) > 0.5:
            err = float(np.max(np.abs(q_cmd - watcher.state.q)))
            rospy.loginfo("Replay waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(replay_path), err)
            last_log = time.time()
        rate.sleep()

    q_goal = replay_path[-1]
    for _ in range(hold_ticks):
        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_goal)
        rate.sleep()

    rospy.loginfo("Trajectory replay complete. Held final pose for %.2fs.", args.hold_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
