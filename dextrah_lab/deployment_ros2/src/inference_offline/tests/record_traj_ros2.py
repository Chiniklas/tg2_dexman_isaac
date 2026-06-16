#!/usr/bin/env python3
"""Record runtime robot joint feedback from feedback_control_bridge.

This recorder intentionally has no Isaac Sim, RL-Games, checkpoint, object, or
teacher/student rollout modes.  It only subscribes to the bridge-published
``sensor_msgs/msg/JointState`` stream, extracts the requested robot joints, and
writes one trajectory file.

Typical usage from a sourced ROS 2 workspace:

python3 src/inference_offline/tests/record_traj_ros2.py \
  --joint-state-topic /joint_states \
  --duration-sec 10 \
  --save-dir src/inference_offline/tests/runtime_trajs

Press Ctrl-C to stop early; collected samples are still saved.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import rclpy
import yaml
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

JOINT_NAME_ALIASES = {
    "little_joint_0": ("little_joint_0", "right_little_1_joint"),
    "ring_joint_0": ("ring_joint_0", "right_ring_1_joint"),
    "middle_joint_0": ("middle_joint_0", "right_middle_1_joint"),
    "index_joint_0": ("index_joint_0", "right_index_1_joint"),
    "thumb_joint_0": ("thumb_joint_0", "right_thumb_1_joint"),
    "thumb_joint_1": ("thumb_joint_1", "right_thumb_2_joint"),
}

DEFAULT_SAVE_DIR = str(Path(__file__).resolve().parent / "runtime_trajs")


@dataclass
class JointSample:
    ros_time: float
    receive_time: float
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    source_names: list[str]


def _log_info(logger, msg: str, *args: object) -> None:
    logger.info(msg % args if args else msg)


def _log_warn(logger, msg: str, *args: object) -> None:
    logger.warning(msg % args if args else msg)


def _parse_joint_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--joint-names must contain at least one joint name.")
    return names


def _message_time_sec(msg: JointState, fallback: float) -> float:
    stamp = msg.header.stamp
    stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return stamp_sec if stamp_sec > 0.0 else fallback


def _extract_joint_vector(
    msg: JointState,
    joint_names: list[str],
    values: Iterable[float],
    fill_value: float,
) -> tuple[np.ndarray, list[str]] | None:
    value_list = list(values)
    if not value_list:
        return np.full(len(joint_names), fill_value, dtype=np.float64), ["" for _ in joint_names]

    idx_by_name = {name: i for i, name in enumerate(msg.name)}
    out = np.zeros(len(joint_names), dtype=np.float64)
    source_names: list[str] = []

    for out_idx, joint_name in enumerate(joint_names):
        aliases = JOINT_NAME_ALIASES.get(joint_name, (joint_name,))
        src_idx = None
        src_name = ""
        for alias in aliases:
            candidate = idx_by_name.get(alias)
            if candidate is not None:
                src_idx = candidate
                src_name = alias
                break
        if src_idx is None or src_idx >= len(value_list):
            return None
        out[out_idx] = float(value_list[src_idx])
        source_names.append(src_name)

    return out, source_names


class RuntimeTrajectoryRecorder(Node):
    def __init__(
        self,
        joint_state_topic: str,
        joint_names: list[str],
        min_period_sec: float,
        require_velocity: bool,
        require_effort: bool,
    ) -> None:
        super().__init__("runtime_traj_recorder")
        self._joint_state_topic = joint_state_topic
        self._joint_names = list(joint_names)
        self._min_period_sec = max(0.0, float(min_period_sec))
        self._require_velocity = bool(require_velocity)
        self._require_effort = bool(require_effort)
        self._samples: list[JointSample] = []
        self._last_record_receive_time: float | None = None
        self._missing_warned = False
        self._source_names: list[str] = []

        self._sub = self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)

    @property
    def samples(self) -> list[JointSample]:
        return self._samples

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def source_names(self) -> list[str]:
        return self._source_names

    def _joint_state_cb(self, msg: JointState) -> None:
        now = time.time()
        if self._last_record_receive_time is not None and now - self._last_record_receive_time < self._min_period_sec:
            return

        position_result = _extract_joint_vector(
            msg=msg,
            joint_names=self._joint_names,
            values=msg.position,
            fill_value=np.nan,
        )
        if position_result is None:
            if not self._missing_warned:
                _log_warn(
                    self.get_logger(),
                    "Ignoring %s until it contains all requested joints. Requested=%s",
                    self._joint_state_topic,
                    self._joint_names,
                )
                self._missing_warned = True
            return

        position, source_names = position_result
        velocity_result = _extract_joint_vector(
            msg=msg,
            joint_names=self._joint_names,
            values=msg.velocity,
            fill_value=np.nan,
        )
        effort_result = _extract_joint_vector(
            msg=msg,
            joint_names=self._joint_names,
            values=msg.effort,
            fill_value=np.nan,
        )

        if self._require_velocity and velocity_result is None:
            return
        if self._require_effort and effort_result is None:
            return

        velocity = (
            velocity_result[0]
            if velocity_result is not None
            else np.full(len(self._joint_names), np.nan, dtype=np.float64)
        )
        effort = (
            effort_result[0]
            if effort_result is not None
            else np.full(len(self._joint_names), np.nan, dtype=np.float64)
        )

        self._samples.append(
            JointSample(
                ros_time=_message_time_sec(msg, now),
                receive_time=now,
                position=position,
                velocity=velocity,
                effort=effort,
                source_names=source_names,
            )
        )
        self._source_names = source_names
        self._last_record_receive_time = now


def _stack_samples(samples: list[JointSample], key: str) -> np.ndarray:
    if not samples:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack([getattr(sample, key) for sample in samples], axis=0).astype(np.float64)


def _write_trajectory(
    save_dir: str,
    output_name: str,
    recorder: RuntimeTrajectoryRecorder,
    started_at: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    samples = recorder.samples
    if not samples:
        raise RuntimeError("No joint feedback samples were recorded; nothing to save.")

    run_dir = Path(os.path.expanduser(save_dir)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    base = run_dir / output_name
    h5_path = str(base.with_suffix(".h5"))
    yaml_path = str(base.with_suffix(".yaml"))

    position = _stack_samples(samples, "position")
    velocity = _stack_samples(samples, "velocity")
    effort = _stack_samples(samples, "effort")
    ros_time = np.asarray([sample.ros_time for sample in samples], dtype=np.float64)
    receive_time = np.asarray([sample.receive_time for sample in samples], dtype=np.float64)
    relative_time = ros_time - ros_time[0]

    meta = {
        "timestamp": started_at,
        "source": "feedback_control_bridge_joint_states",
        "joint_state_topic": args.joint_state_topic,
        "sample_count": int(position.shape[0]),
        "dof": int(position.shape[1]),
        "duration_sec": float(relative_time[-1]) if relative_time.size > 0 else 0.0,
        "stop_reason": getattr(args, "stop_reason", "unknown"),
        "joint_names": recorder.joint_names,
        "source_joint_names": recorder.source_names,
        "datasets": {
            "obs": list(position.shape),
            "joint_position": list(position.shape),
            "joint_velocity": list(velocity.shape),
            "joint_effort": list(effort.shape),
            "ros_time": list(ros_time.shape),
            "receive_time": list(receive_time.shape),
            "relative_time": list(relative_time.shape),
        },
        "notes": (
            "obs is an alias of joint_position for compatibility with offline trajectory tools. "
            "All values come from the ROS 2 JointState feedback topic."
        ),
    }

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("obs", data=position, compression="gzip", compression_opts=3)
        f.create_dataset("joint_position", data=position, compression="gzip", compression_opts=3)
        f.create_dataset("joint_velocity", data=velocity, compression="gzip", compression_opts=3)
        f.create_dataset("joint_effort", data=effort, compression="gzip", compression_opts=3)
        f.create_dataset("ros_time", data=ros_time)
        f.create_dataset("receive_time", data=receive_time)
        f.create_dataset("relative_time", data=relative_time)
        f.attrs["timestamp"] = started_at
        f.attrs["source"] = meta["source"]
        f.attrs["joint_state_topic"] = args.joint_state_topic
        f.attrs["sample_count"] = int(position.shape[0])
        f.attrs["dof"] = int(position.shape[1])
        f.attrs["duration_sec"] = float(meta["duration_sec"])
        f.attrs["stop_reason"] = str(meta["stop_reason"])

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)

    return h5_path, yaml_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record runtime trajectories from feedback_control_bridge /joint_states.")
    p.add_argument("--joint-state-topic", default="/joint_states", help="JointState feedback topic from the bridge.")
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory for the output .h5/.yaml files.")
    p.add_argument(
        "--output-name",
        default=None,
        help="Output stem without extension. Defaults to runtime_traj_<timestamp>.",
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Recording duration. Use 0 to record until Ctrl-C or --max-samples is reached.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Stop after this many samples. Use 0 for unlimited.")
    p.add_argument(
        "--min-period-sec",
        type=float,
        default=0.0,
        help="Minimum receive-time spacing between saved samples. Use 0 to save every message.",
    )
    p.add_argument(
        "--wait-feedback-sec",
        type=float,
        default=5.0,
        help="How long to wait for the first complete feedback sample before aborting.",
    )
    p.add_argument(
        "--joint-names",
        default=",".join(RIGHT_ARM_HAND_JOINTS),
        help="Comma-separated canonical joint names to record, in saved vector order.",
    )
    p.add_argument(
        "--require-velocity",
        action="store_true",
        help="Only save samples whose JointState contains velocity for all requested joints.",
    )
    p.add_argument(
        "--require-effort",
        action="store_true",
        help="Only save samples whose JointState contains effort for all requested joints.",
    )
    cli_args = remove_ros_args(args=argv or sys.argv)[1:]
    return p.parse_args(cli_args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration_sec < 0.0:
        raise ValueError("--duration-sec must be >= 0.")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0.")
    if args.min_period_sec < 0.0:
        raise ValueError("--min-period-sec must be >= 0.")
    if args.wait_feedback_sec < 0.0:
        raise ValueError("--wait-feedback-sec must be >= 0.")

    joint_names = _parse_joint_names(args.joint_names)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"runtime_traj_{started_at}"

    rclpy.init(args=argv)
    recorder = RuntimeTrajectoryRecorder(
        joint_state_topic=args.joint_state_topic,
        joint_names=joint_names,
        min_period_sec=args.min_period_sec,
        require_velocity=args.require_velocity,
        require_effort=args.require_effort,
    )
    logger = recorder.get_logger()

    _log_info(logger, "Recording feedback topic=%s", args.joint_state_topic)
    _log_info(logger, "Joint order=%s", joint_names)

    start_monotonic = time.monotonic()
    first_sample_deadline = start_monotonic + args.wait_feedback_sec
    stop_reason = "interrupted"

    try:
        while rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.05)

            sample_count = len(recorder.samples)
            now = time.monotonic()
            if sample_count == 0 and now > first_sample_deadline:
                _log_warn(logger, "No complete feedback sample received within %.2fs.", args.wait_feedback_sec)
                return 1
            if args.max_samples > 0 and sample_count >= args.max_samples:
                stop_reason = "max_samples"
                break
            if args.duration_sec > 0.0 and (now - start_monotonic) >= args.duration_sec:
                stop_reason = "duration"
                break
    except KeyboardInterrupt:
        stop_reason = "ctrl_c"
    finally:
        sample_count = len(recorder.samples)
        if sample_count == 0:
            _log_warn(logger, "No samples recorded; no files written.")
            recorder.destroy_node()
            rclpy.shutdown()
            return 1

        args.stop_reason = stop_reason
        h5_path, yaml_path = _write_trajectory(
            save_dir=args.save_dir,
            output_name=output_name,
            recorder=recorder,
            started_at=started_at,
            args=args,
        )
        _log_info(logger, "Saved %d samples (%s): %s", sample_count, stop_reason, h5_path)
        _log_info(logger, "Saved metadata: %s", yaml_path)
        recorder.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
