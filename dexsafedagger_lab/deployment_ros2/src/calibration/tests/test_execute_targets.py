#!/usr/bin/env python3
"""Publish a simple joint-space trajectory to test feedback_control_bridge.

Control instructions:
1) Verify bridge topics are available:
   rostopic list | grep -E "/arm/command_joint_states|/joint_states"
2) Check current start state (without moving):
   python3 test_execute_targets.py --delta "0,0,0,0,0,0,0,0,0,0,0,0,0" --dry-run
3) Move with incremental delta from current state (recommended first test):
   python3 test_execute_targets.py --delta "0.05,0,0,0,0,0,0,0,0,0,0,0,0" --dry-run
   python3 test_execute_targets.py --delta "0.05,0,0,0,0,0,0,0,0,0,0,0,0" --steps 80 --rate 30
4) Move using absolute joint target:
   python3 test_execute_targets.py --target "q1,q2,...,q13" --dry-run
   python3 test_execute_targets.py --target "q1,q2,...,q13" --steps 80 --rate 30

Notes:
- Use either --target (absolute) or --delta (incremental), not both.
- Command order is right arm (7) + right hand (6): shoulder..wrist, little/ring/middle/index/thumb0/thumb1.
- The script returns to start pose after the test.
- Hand-control semantics: this script computes joint-space deltas/targets, but the bridge maps hand joints to
  `/inspire_hand/set_angle_flexible/right_hand` absolute `angleRatio` values in [0, 1].
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
    p = argparse.ArgumentParser(description="Test trajectory execution via feedback_control_bridge.")
    p.add_argument("--command-topic", default="/arm/command_joint_states")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument(
        "--target",
        default="",
        help="Target joint vector in radians with 13 values: q1..q13",
    )
    p.add_argument(
        "--delta",
        default="",
        help="Incremental joint vector in radians with 13 values: dq1..dq13 (applied to start).",
    )
    p.add_argument(
        "--start",
        default="",
        help="Optional start joint vector in radians. If omitted, uses latest /joint_states.",
    )
    p.add_argument("--steps", type=int, default=60, help="Interpolation points from start->target.")
    p.add_argument("--rate", type=float, default=30.0, help="Command publish rate (Hz).")
    p.add_argument("--hold-sec", type=float, default=1.0, help="Hold time at each endpoint.")
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

    rospy.init_node("test_execute_targets", anonymous=True)
    pub = rospy.Publisher(args.command_topic, JointState, queue_size=10)
    watcher = JointStateWatcher(args.joint_state_topic, joint_names)

    # Wait for at least one feedback frame so we can start from the real robot state.
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

    has_any_delta = bool(args.delta)

    if args.target and has_any_delta:
        rospy.logerr("Use either --target (absolute) or delta mode (--delta), not both.")
        return 1
    if not args.target and not has_any_delta:
        rospy.logerr("Provide either --target or delta mode (--delta).")
        return 1
    if args.target:
        q_target = _parse_vec(args.target, "--target", dof)
    else:
        q_target = q_start.copy()
        q_target += _parse_vec(args.delta, "--delta", dof)

    up = _interpolate(q_start, q_target, args.steps)
    down = _interpolate(q_target, q_start, args.steps)
    full = up + down[1:]

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

    rospy.loginfo("Executing %d waypoints on right arm+hand (%d DoF) via %s", len(full), dof, args.command_topic)
    rospy.loginfo("Start=%s", np.array2string(q_start, precision=3))
    rospy.loginfo("Target=%s", np.array2string(q_target, precision=3))
    rospy.loginfo("Max per-step joint delta: %.4f rad", max_step)

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

    # Hold final/start pose for stability.
    for _ in range(hold_ticks):
        if rospy.is_shutdown():
            return 1
        _publish_target(pub, joint_names, q_start)
        rate.sleep()

    rospy.loginfo("Trajectory execution test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
