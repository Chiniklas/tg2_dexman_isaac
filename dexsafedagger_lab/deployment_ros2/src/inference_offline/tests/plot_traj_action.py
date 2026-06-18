#!/usr/bin/env python3
"""Plot mapped trajectory action targets per joint dimension.

Loads a trajectory dataset (typically normalized actions in [-1, 1]), applies
the same mapping used by execute_offline_traj.py:
1) clip normalized actions to [-1, 1]
2) absolute-action scaling to joint limits
3) hand semantic conversion (sim joint-space -> bridge command-space)

Then makes two per-dimension plots of mapped joint targets:
- arm plot: 7 DoF
- hand plot: 6 DoF

Usage:
1) Show plot window only:
python3 /tiangong_infra_ws/ws/src/inference_offline/tests/plot_traj_action.py \
  --traj-file /tiangong_infra_ws/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key action

2) Show plot and save PNG:
python3 /tiangong_infra_ws/ws/src/inference_offline/tests/plot_traj_action.py \
  --traj-file /tiangong_infra_ws/ws/src/inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5 \
  --dataset-key action \
  --output /tiangong_infra_ws/ws/src/inference_offline/tests/mapped_action_targets.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


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

JOINT_LOWER_LIMITS = np.asarray(
    [-2.96, -2.0, 0.0, -2.0, -2.9671, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    dtype=np.float64,
)
JOINT_UPPER_LIMITS = np.asarray(
    [0.0, -0.1, 2.5, 0.0, 2.9671, 0.6, 0.5, 1.1, 1.1, 1.1, 1.1, 1.2, 0.5],
    dtype=np.float64,
)

DEFAULT_TRAJ_FILE = (
    "/home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/deployment_tg2_inspirehand/ws/src/"
    "inference_offline/tests/offline_tarjs/1m0lvpzs/traj_env_0_file_1.h5"
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _load_actions(
    traj_file: str,
    dataset_key: str,
    batch_index: int,
    action_indices: list[int],
    max_steps: int,
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required. Install python3-h5py in this environment.") from exc

    path = os.path.expanduser(traj_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found. Keys: {list(f.keys())}")
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

    if action_indices:
        if any(i < 0 or i >= arr.shape[1] for i in action_indices):
            raise ValueError(
                f"action-indices out of range for action dim {arr.shape[1]}: {action_indices}"
            )
        arr = arr[:, action_indices]

    return arr


def _scale_normalized_to_joint_limits(actions: np.ndarray) -> np.ndarray:
    clipped = np.clip(actions, -1.0, 1.0)
    return 0.5 * (clipped + 1.0) * (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) + JOINT_LOWER_LIMITS


def _sim_joint_to_bridge_command(sim_joint_targets: np.ndarray) -> np.ndarray:
    cmd = sim_joint_targets.copy()
    hand_lo = JOINT_LOWER_LIMITS[7:]
    hand_hi = JOINT_UPPER_LIMITS[7:]
    cmd[:, 7:] = hand_lo + hand_hi - sim_joint_targets[:, 7:]
    cmd[:, 7:] = np.clip(cmd[:, 7:], hand_lo, hand_hi)
    return cmd


def _max_step(arr: np.ndarray) -> float:
    if arr.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(arr[1:] - arr[:-1])))


def _max_step_info(arr: np.ndarray) -> tuple[float, int, int]:
    """Return (max_delta, step_idx, joint_idx).

    step_idx is the second sample index in the edge arr[step_idx-1] -> arr[step_idx].
    """
    if arr.shape[0] < 2:
        return 0.0, 0, 0
    diffs = np.abs(arr[1:] - arr[:-1])  # [T-1, D]
    flat = int(np.argmax(diffs))
    edge_idx, joint_idx = np.unravel_index(flat, diffs.shape)
    step_idx = int(edge_idx + 1)
    return float(diffs[edge_idx, joint_idx]), step_idx, int(joint_idx)


def _plot_group(
    plt,
    mapped: np.ndarray,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    start_idx: int,
    title: str,
    max_step_idx: int,
    max_joint_idx: int,
    max_step_val: float,
):
    n = len(names)
    t = np.arange(mapped.shape[0], dtype=np.int64)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(8, int(1.8 * n))), sharex=True)
    if n == 1:
        axes = [axes]
    for i, name in enumerate(names):
        j = start_idx + i
        ax = axes[i]
        ax.plot(t, mapped[:, j], linewidth=1.3, color="#1f77b4")
        ax.axhline(lower[j], linestyle="--", linewidth=0.8, color="#d62728", alpha=0.5)
        ax.axhline(upper[j], linestyle="--", linewidth=0.8, color="#2ca02c", alpha=0.5)
        ax.axvline(max_step_idx, linestyle="--", linewidth=0.8, color="#9467bd", alpha=0.6)
        if j == max_joint_idx and 0 < max_step_idx < mapped.shape[0]:
            y_prev = float(mapped[max_step_idx - 1, j])
            y_curr = float(mapped[max_step_idx, j])
            ax.scatter([max_step_idx - 1, max_step_idx], [y_prev, y_curr], color="#d62728", s=20, zorder=4)
            ax.text(
                max_step_idx,
                y_curr,
                f" max {max_step_val:.4f} rad @ {max_step_idx-1}->{max_step_idx}",
                color="#d62728",
                fontsize=8,
                va="bottom",
            )
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot mapped trajectory action targets per joint dimension.")
    p.add_argument("--traj-file", default=DEFAULT_TRAJ_FILE, help="Path to trajectory .h5 file.")
    p.add_argument("--dataset-key", default="action", help="Dataset key (default: action).")
    p.add_argument("--batch-index", type=int, default=0, help="Batch index for [T,B,A] datasets.")
    p.add_argument(
        "--action-indices",
        default="",
        help="Optional comma-separated action indices to select 13 dimensions.",
    )
    p.add_argument("--max-steps", type=int, default=0, help="Limit to first N steps (0 = full).")
    p.add_argument("--traj-scale", type=float, default=1.0, help="Scale for normalized actions before clipping.")
    p.add_argument("--traj-offset", type=float, default=0.0, help="Offset for normalized actions before clipping.")
    p.add_argument("--output", default="", help="Optional PNG path. If empty, no file is saved.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    action_indices = _parse_int_list(args.action_indices) if args.action_indices else []

    raw = _load_actions(
        traj_file=args.traj_file,
        dataset_key=args.dataset_key,
        batch_index=args.batch_index,
        action_indices=action_indices,
        max_steps=args.max_steps,
    )

    dof = len(RIGHT_ARM_HAND_JOINTS)
    if raw.shape[1] != dof:
        raise ValueError(
            f"Trajectory action dimension must be {dof} after selection, got {raw.shape[1]}."
        )

    norm = args.traj_scale * raw + args.traj_offset
    norm_clipped = np.clip(norm, -1.0, 1.0)
    clipped_count = int(np.count_nonzero(np.abs(norm - norm_clipped) > 1e-9))
    sim = _scale_normalized_to_joint_limits(norm_clipped)
    mapped = _sim_joint_to_bridge_command(sim)

    # Use Agg backend automatically for headless environments.
    import matplotlib

    display_available = bool(os.environ.get("DISPLAY"))
    no_tk_backend = False
    if not display_available:
        matplotlib.use("Agg")
    can_show = display_available
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name != "tkinter":
            raise
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        can_show = False
        no_tk_backend = True
        print("tkinter is missing; falling back to non-interactive plotting backend (Agg).")
        print("Install python3-tk to enable interactive matplotlib windows.")

    arm_names = RIGHT_ARM_HAND_JOINTS[:7]
    hand_names = RIGHT_ARM_HAND_JOINTS[7:]
    common = (
        f"file={Path(args.traj_file).name}, steps={mapped.shape[0]}, "
        f"clipped_entries={clipped_count}, max_step={_max_step(mapped):.4f} rad"
    )
    max_step_val, max_step_idx, max_joint_idx = _max_step_info(mapped)
    max_joint_name = RIGHT_ARM_HAND_JOINTS[max_joint_idx]
    fig_arm = _plot_group(
        plt=plt,
        mapped=mapped,
        names=arm_names,
        lower=JOINT_LOWER_LIMITS,
        upper=JOINT_UPPER_LIMITS,
        start_idx=0,
        title=f"Mapped Trajectory Action Targets (Arm 7 DoF)\n{common}",
        max_step_idx=max_step_idx,
        max_joint_idx=max_joint_idx,
        max_step_val=max_step_val,
    )
    fig_hand = _plot_group(
        plt=plt,
        mapped=mapped,
        names=hand_names,
        lower=JOINT_LOWER_LIMITS,
        upper=JOINT_UPPER_LIMITS,
        start_idx=7,
        title=f"Mapped Trajectory Action Targets (Hand 6 DoF)\n{common}",
        max_step_idx=max_step_idx,
        max_joint_idx=max_joint_idx,
        max_step_val=max_step_val,
    )

    if args.output:
        out = Path(os.path.expanduser(args.output))
        out.parent.mkdir(parents=True, exist_ok=True)
        arm_out = out.with_name(f"{out.stem}_arm{out.suffix or '.png'}")
        hand_out = out.with_name(f"{out.stem}_hand{out.suffix or '.png'}")
        fig_arm.savefig(arm_out, dpi=160)
        fig_hand.savefig(hand_out, dpi=160)
        print(f"Saved arm plot: {arm_out}")
        print(f"Saved hand plot: {hand_out}")
    print(
        f"Mapped max per-step delta: {max_step_val:.4f} rad on joint '{max_joint_name}' "
        f"at step {max_step_idx-1}->{max_step_idx}"
    )
    print(f"Clipped normalized entries: {clipped_count}")
    if can_show:
        plt.show()
    else:
        if no_tk_backend:
            print("Interactive window disabled because tkinter is unavailable in this environment.")
        else:
            print("No DISPLAY found; cannot open interactive window in this environment.")
        plt.close(fig_arm)
        plt.close(fig_hand)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
