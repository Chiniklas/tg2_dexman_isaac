"""Replay batch teacher trajectories (.h5) one-by-one in Isaac Lab.

Example:
python replay_teacher_traj.py \
  --task dextrah_tg2_inspirehand \
  --num_envs 1 \
  --objects_dir test_object \
  --max_pose_angle 90 \
  --traj_dir /path/to/offline_tarjs/teacher_recorded_data_YYYYMMDD_HHMMSS/data/<object_name> \
  --action_key action \
  --num_replays_per_file 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import pathlib
import re
import shutil

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Replay batch teacher trajectories one-by-one.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="dextrah_tg2_inspirehand", help="Task name.")
parser.add_argument(
    "--objects_dir",
    type=str,
    default="test_object",
    help="Name of objects directory under assets/ (e.g., test_object).",
)
parser.add_argument(
    "--object_name",
    type=str,
    default=None,
    help="Optional: pick a single object (folder name inside <objects_dir>/USD).",
)
parser.add_argument(
    "--max_pose_angle",
    type=float,
    default=90.0,
    help="Max palm pose angle (degrees).",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=5.0,
    help="Episode timeout in seconds.",
)
parser.add_argument(
    "--traj_dir",
    type=str,
    required=True,
    help="Directory containing trajectory .h5 files.",
)
parser.add_argument(
    "--traj_glob",
    type=str,
    default="traj_env_*.h5",
    help="Glob pattern inside --traj_dir to match trajectory files (supports file/episode naming).",
)
parser.add_argument("--action_key", type=str, default="action", help="Dataset key containing actions.")
parser.add_argument(
    "--max_steps_per_traj",
    type=int,
    default=0,
    help="Replay at most this many steps per trajectory. 0 = full.",
)
parser.add_argument(
    "--num_replays_per_file",
    type=int,
    default=1,
    help="Replay each file this many times.",
)
parser.add_argument(
    "--stop_on_all_done",
    action="store_true",
    default=False,
    help="Stop replay for current file when all envs are done.",
)
parser.add_argument(
    "--keep_randomization",
    action="store_true",
    default=False,
    help="Keep env randomization and done behavior from task defaults.",
)
parser.add_argument(
    "--max_files",
    type=int,
    default=0,
    help="Replay at most this many files. 0 = all matched files.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="batch_replay_teacher_logs",
    help="Directory to save replay summary YAML.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup  # noqa: F401
import dextrah_lab.tasks.dextrah_kuka_inspirehand.gym_setup  # noqa: F401
import dextrah_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401


@dataclass
class ReplayStats:
    path: str
    replay_index: int
    replay_steps: int
    mean_reward: float
    ever_lift_success_per_env: list[bool]
    source_ever_lift_success: bool | None


def _prepare_single_object_dir(base_objects_dir: str, object_name: str) -> str:
    """Create a single-object view while preserving original one-hot length."""
    root_path = pathlib.Path(__file__).resolve().parents[1]
    assets_dir = root_path / "assets"
    source_usd_dir = assets_dir / base_objects_dir / "USD"

    src_object_dir = source_usd_dir / object_name
    if not src_object_dir.is_dir():
        raise FileNotFoundError(f"Object '{object_name}' not found under {source_usd_dir}")

    sub_dirs = sorted([p.name for p in source_usd_dir.iterdir() if p.is_dir()])

    target_dir_name = f"{base_objects_dir}_single_pick"
    target_usd_dir = assets_dir / target_dir_name / "USD"
    if target_usd_dir.exists():
        shutil.rmtree(target_usd_dir)
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    chosen_usd = src_object_dir / f"{object_name}.usd"
    if not chosen_usd.is_file():
        raise FileNotFoundError(f"USD file not found for object '{object_name}' at {chosen_usd}")

    for name in sub_dirs:
        obj_dir = target_usd_dir / name
        obj_dir.mkdir(parents=True, exist_ok=True)
        link_path = obj_dir / f"{name}.usd"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(chosen_usd)

    return target_dir_name


def _traj_sort_key(path: Path) -> tuple[int, int, str]:
    m = re.search(r"traj_env_(\d+)_file_(\d+)\.h5$", path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), path.name
    m = re.search(r"traj_env_(\d+)_episode_(\d+)\.h5$", path.name)
    if m:
        return int(m.group(1)), int(m.group(2)), path.name
    return (10**9, 10**9, path.name)


def _collect_traj_files(traj_dir: str, traj_glob: str, max_files: int) -> list[Path]:
    root = Path(traj_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--traj_dir does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--traj_dir is not a directory: {root}")

    files = sorted(root.glob(traj_glob), key=_traj_sort_key)
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{traj_glob}' in {root}")
    return files


def _compute_lift_success_flags(ov_env) -> torch.Tensor:
    table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0)
    return ov_env.object_pos[:, 2] > lift_height_thresh


def _load_actions(path: Path, action_key: str) -> tuple[np.ndarray, bool | None]:
    with h5py.File(path, "r") as f:
        if action_key not in f:
            raise KeyError(f"Dataset '{action_key}' not found in '{path}'. Keys: {list(f.keys())}")
        actions = np.asarray(f[action_key], dtype=np.float32)
        source_success = None
        if "ever_lift_success" in f:
            source_success = bool(np.asarray(f["ever_lift_success"]).item())
    if actions.ndim not in (2, 3):
        raise ValueError(f"Expected action tensor rank 2 or 3, got shape {actions.shape} in {path}")
    return actions, source_success


def _prepare_step_action(
    actions: np.ndarray,
    step: int,
    env_num_envs: int,
    env_num_actions: int,
    device: torch.device,
) -> torch.Tensor:
    if actions.ndim == 2:
        if env_num_envs != 1:
            raise ValueError(
                f"Trajectory shape is [T,A]={actions.shape} but env_num_envs={env_num_envs}. "
                "Use --num_envs 1 for per-env trajectory files."
            )
        step_action = actions[step][None, :]
    else:
        if actions.shape[1] != env_num_envs:
            raise ValueError(f"Trajectory batch size B={actions.shape[1]} but env_num_envs={env_num_envs}.")
        step_action = actions[step]

    if step_action.shape[-1] != env_num_actions:
        raise ValueError(
            f"Action dimension mismatch: trajectory has {step_action.shape[-1]}, env expects {env_num_actions}."
        )

    action = torch.from_numpy(step_action).to(device=device, dtype=torch.float32)
    return torch.clamp(action, -1.0, 1.0)


def _replay_one(env: gym.Env, path: Path, replay_index: int) -> ReplayStats:
    actions, src_success = _load_actions(path, args_cli.action_key)
    ov_env = env.unwrapped
    device = ov_env.device
    num_envs = ov_env.num_envs
    num_actions = ov_env.num_actions

    total_steps = int(actions.shape[0])
    if args_cli.max_steps_per_traj > 0:
        total_steps = min(total_steps, int(args_cli.max_steps_per_traj))

    obs = env.reset()[0]
    _ = obs
    rewards: list[float] = []
    ever_lift = torch.zeros((num_envs,), dtype=torch.bool, device=device)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break
        action = _prepare_step_action(actions, step, num_envs, num_actions, device)
        _, reward, out_of_reach, timed_out, _ = env.step(action)
        rewards.append(float(reward.mean().item()))
        ever_lift |= _compute_lift_success_flags(ov_env)

        if args_cli.stop_on_all_done:
            dones = out_of_reach | timed_out
            if bool(torch.all(dones).item()):
                break

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    return ReplayStats(
        path=str(path),
        replay_index=replay_index,
        replay_steps=len(rewards),
        mean_reward=mean_reward,
        ever_lift_success_per_env=[bool(v) for v in ever_lift.detach().cpu().tolist()],
        source_ever_lift_success=src_success,
    )


def main() -> None:
    if args_cli.num_replays_per_file < 1:
        raise ValueError("--num_replays_per_file must be >= 1.")
    files = _collect_traj_files(args_cli.traj_dir, args_cli.traj_glob, args_cli.max_files)
    print(f"[INFO] Found {len(files)} trajectory files.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.object_name:
        env_cfg.objects_dir = _prepare_single_object_dir(args_cli.objects_dir, args_cli.object_name)
        if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
            env_cfg.valid_objects_dir.append(env_cfg.objects_dir)
        print(f"[INFO] Using single object '{args_cli.object_name}' via objects_dir='{env_cfg.objects_dir}'.")
    else:
        env_cfg.objects_dir = args_cli.objects_dir
    env_cfg.max_pose_angle = float(args_cli.max_pose_angle)

    if args_cli.episode_length_s <= 0.0:
        raise ValueError("--episode_length_s must be > 0.")
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if hasattr(env_cfg, "distillation_episode_length_s"):
        env_cfg.distillation_episode_length_s = float(args_cli.episode_length_s)
    print(f"[INFO] Overriding episode_length_s to {env_cfg.episode_length_s:.3f}s")

    if not args_cli.keep_randomization:
        env_cfg.disable_out_of_reach_done = True
        env_cfg.disable_arm_randomization = True
        env_cfg.disable_dome_light_randomization = True

    env = gym.make(args_cli.task, cfg=env_cfg)

    all_stats: list[ReplayStats] = []
    done = False
    for path in files:
        if done or not simulation_app.is_running():
            break
        for replay_idx in range(1, int(args_cli.num_replays_per_file) + 1):
            if not simulation_app.is_running():
                done = True
                break
            stats = _replay_one(env, path, replay_idx)
            all_stats.append(stats)
            print(
                f"[Replay] file={path.name} pass={replay_idx}/{args_cli.num_replays_per_file} "
                f"steps={stats.replay_steps} mean_reward={stats.mean_reward:.3f} "
                f"source_success={stats.source_ever_lift_success} "
                f"replay_success={stats.ever_lift_success_per_env}"
            )

    env.close()

    os.makedirs(args_cli.save_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args_cli.save_dir, f"batch_teacher_replay_summary_{stamp}.yaml")
    summary = {
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "objects_dir": str(env_cfg.objects_dir),
        "episode_length_s": float(args_cli.episode_length_s),
        "traj_dir": str(Path(args_cli.traj_dir).expanduser().resolve()),
        "traj_glob": args_cli.traj_glob,
        "files_matched": len(files),
        "files_replayed": len({s.path for s in all_stats}),
        "num_replays_per_file": int(args_cli.num_replays_per_file),
        "action_key": args_cli.action_key,
        "max_steps_per_traj": int(args_cli.max_steps_per_traj),
        "stop_on_all_done": bool(args_cli.stop_on_all_done),
        "keep_randomization": bool(args_cli.keep_randomization),
        "replays_completed": len(all_stats),
        "mean_reward_across_replays": float(np.mean([s.mean_reward for s in all_stats])) if all_stats else 0.0,
        "details": [
            {
                "path": s.path,
                "replay_index": s.replay_index,
                "replay_steps": s.replay_steps,
                "mean_reward": s.mean_reward,
                "source_ever_lift_success": s.source_ever_lift_success,
                "replay_ever_lift_success_per_env": s.ever_lift_success_per_env,
            }
            for s in all_stats
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, default_flow_style=False)
    print(f"[INFO] Saved replay summary: {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
