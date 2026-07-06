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
  cmd_pos = min + ratio * (max - min)

Examples:
python3 test_init_and_homing.py --mode homing --steps 120 --rate 30 --hold-sec 3
python3 test_init_and_homing.py --mode init --steps 120 --rate 30 --hold-sec 3
"""

from __future__ import annotations

import argparse
import math
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
    # Publish hand joint values such that the bridge sends this exact ratio.
    return lo + ratio * (hi - lo)


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


def parse_args() -> argparse.Namespace:
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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    joint_names = RIGHT_ARM_HAND_JOINTS
    dof = len(joint_names)

    rospy.init_node("test_init_and_homing", anonymous=True)
    pub = rospy.Publisher(args.command_topic, JointState, queue_size=10)
    watcher = JointStateWatcher(args.joint_state_topic, joint_names)

    # Wait for at least one feedback frame so we can start from real state.
    t0 = time.time()
    while not rospy.is_shutdown() and watcher.state.q is None and (time.time() - t0) < args.wait_feedback_sec:
        time.sleep(0.05)
    if watcher.state.q is None and not args.start:
        rospy.logerr("No /joint_states received within %.2fs, and --start not provided.", args.wait_feedback_sec)
        return 1

    q_start = _parse_vec(args.start, "--start", dof) if args.start else watcher.state.q.copy()
    if q_start is None:
        rospy.logerr("Start vector is unavailable.")
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
        q_arm_goal = _parse_vec(args.homing_pose, "--homing-pose", 7) if args.homing_pose else RIGHT_ARM_HOMING.copy()
        q_hand_ratio = (
            _parse_vec(args.homing_hand, "--homing-hand", 6)
            if args.homing_hand
            else RIGHT_HAND_HOMING_RATIO.copy()
        )
        goal_label = "homing"
    q_hand_goal = _hand_ratio_to_pos(q_hand_ratio)
    q_goal = np.concatenate([q_arm_goal, q_hand_goal], axis=0)
    full = _interpolate(q_start, q_goal, args.steps)

    # Basic safety gate on trajectory smoothness.
    max_step = 0.0
    for i in range(1, len(full)):
        step = float(np.max(np.abs(full[i] - full[i - 1])))
        max_step = max(max_step, step)
    if max_step > args.max_command_step_rad:
        rospy.logerr(
            "Aborting: per-step delta %.4f rad exceeds --max-command-step-rad %.4f rad.",
            max_step,
            args.max_command_step_rad,
        )
        return 1

    rospy.loginfo("Moving right arm+hand (%d DoF) via %s", dof, args.command_topic)
    rospy.loginfo("Start=%s", np.array2string(q_start, precision=3))
    rospy.loginfo("Goal (%s)=%s", goal_label, np.array2string(q_goal, precision=3))
    rospy.loginfo("Hand ratio goal (%s)=%s", goal_label, np.array2string(np.clip(q_hand_ratio, 0.0, 1.0), precision=3))
    rospy.loginfo("Waypoints=%d max_step=%.4f rad", len(full), max_step)

    if args.dry_run:
        return 0

    rate = rospy.Rate(max(1e-3, args.rate))
    hold_ticks = int(math.ceil(max(0.0, args.hold_sec) * args.rate))
    last_log = time.time()

    for idx, q_cmd in enumerate(full):
        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_cmd)

        if watcher.state.q is not None and (time.time() - watcher.state.stamp) > args.feedback_timeout_sec:
            rospy.logerr("Stale joint feedback (> %.2fs). Aborting.", args.feedback_timeout_sec)
            return 1

        if watcher.state.q is not None and (time.time() - last_log) > 0.5:
            err = float(np.max(np.abs(q_cmd - watcher.state.q)))
            rospy.loginfo("Waypoint %d/%d max|cmd-feedback|=%.4f rad", idx + 1, len(full), err)
            last_log = time.time()
        rate.sleep()

    # Hold goal pose for stability.
    for _ in range(hold_ticks):
        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_goal)
        rate.sleep()

    rospy.loginfo("Move complete (mode=%s, held for %.2fs).", goal_label, args.hold_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
