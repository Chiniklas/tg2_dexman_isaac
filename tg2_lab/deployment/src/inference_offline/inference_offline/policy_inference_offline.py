#!/usr/bin/env python3
"""Replay recorded trajectory files (.h5) in Isaac Lab from the ROS 2 deployment workspace."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
from rclpy.utilities import remove_ros_args

from inference_offline.repo_support import ensure_repo_root_on_path


@dataclass
class TrajStats:
    path: str
    replay_steps: int
    mean_reward: float
    ever_lift_success_per_env: list[bool]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay offline trajectory files in Isaac Lab.")
    parser.add_argument("--traj_file", type=str, required=True, help="Path to a single trajectory .h5 file.")
    parser.add_argument("--action_key", type=str, default="action", help="Dataset key containing actions.")
    parser.add_argument("--video", action="store_true", default=False, help="Record viewport video (Isaac Sim).")
    parser.add_argument("--video_length", type=int, default=200, help="Viewport video length in steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Viewport video interval in steps.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of simulated environments.")
    parser.add_argument("--task", type=str, default="dextrah_tg2_inspirehand", help="Task name.")
    parser.add_argument(
        "--objects_dir",
        type=str,
        default="_single_object",
        help="Objects directory token (or full path). Default uses assets/_single_object.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--max_steps_per_traj",
        type=int,
        default=0,
        help="Maximum replay steps per file. Use 0 to replay full trajectory.",
    )
    parser.add_argument(
        "--num_replays",
        type=int,
        default=0,
        help="Number of times to replay the same trajectory. Use 0 for continuous replay.",
    )
    parser.add_argument(
        "--stop_on_all_done",
        action="store_true",
        default=False,
        help="Stop replay for current trajectory once all environments are done.",
    )
    parser.add_argument(
        "--keep_randomization",
        action="store_true",
        default=False,
        help="Keep env randomization and done behavior as configured by task defaults.",
    )
    parser.add_argument("--save_dir", type=str, default="offline_replay_logs", help="Directory to save replay summary.")
    parser.add_argument(
        "--repo_root",
        type=str,
        default="",
        help="Optional explicit path to the tg2_dexman_isaac repository root.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _resolve_traj_file(traj_file: str) -> str:
    path = Path(traj_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    return str(path)


def _resolve_objects_dir_token(raw: str) -> str:
    """Accept a token (e.g. '_single_object') or a full folder path."""
    p = Path(raw).expanduser()
    if p.is_absolute() or "/" in raw:
        return p.name
    return raw


def _compute_lift_success_flags(ov_env: Any) -> torch.Tensor:
    table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0)
    return ov_env.object_pos[:, 2] > lift_height_thresh


def _load_actions(path: str, action_key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if action_key not in f:
            raise KeyError(f"Dataset '{action_key}' not found in '{path}'. Keys: {list(f.keys())}")
        actions = np.asarray(f[action_key], dtype=np.float32)
    if actions.ndim not in (2, 3):
        raise ValueError(
            f"Expected action tensor with rank 2 or 3, got shape {actions.shape} in {path}"
        )
    return actions


def _prepare_step_action(
    actions: np.ndarray,
    step: int,
    env_num_envs: int,
    env_num_actions: int,
    device: torch.device,
) -> torch.Tensor:
    if actions.ndim == 2:
        # [T, A] -> requires env_num_envs == 1
        if env_num_envs != 1:
            raise ValueError(
                f"Trajectory has shape [T,A]={actions.shape} but env_num_envs={env_num_envs}. "
                "Use --num_envs 1 for per-env trajectory files."
            )
        step_action = actions[step][None, :]
    else:
        # [T, B, A] -> B must match env_num_envs
        if actions.shape[1] != env_num_envs:
            raise ValueError(
                f"Trajectory has batch size B={actions.shape[1]} but env_num_envs={env_num_envs}."
            )
        step_action = actions[step]

    if step_action.shape[-1] != env_num_actions:
        raise ValueError(
            f"Action dimension mismatch: file has {step_action.shape[-1]}, env expects {env_num_actions}."
        )

    action = torch.from_numpy(step_action).to(device=device, dtype=torch.float32)
    action = torch.clamp(action, -1.0, 1.0)
    return action


def _replay_one(env: gym.Env, path: str, args_cli: argparse.Namespace, simulation_app) -> TrajStats:
    actions = _load_actions(path, args_cli.action_key)
    ov_env = env.env
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

    steps_ran = len(rewards)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    flags = [bool(v) for v in ever_lift.detach().cpu().tolist()]
    return TrajStats(path=path, replay_steps=steps_ran, mean_reward=mean_reward, ever_lift_success_per_env=flags)


def main(argv: list[str] | None = None) -> int:
    raw_args = remove_ros_args(args=argv or sys.argv)[1:]
    parser = build_arg_parser()
    args_cli, hydra_args = parser.parse_known_args(raw_args)
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        repo_root = ensure_repo_root_on_path(args_cli.repo_root or None)
        _ = repo_root

        global gym, h5py, np, torch, yaml
        import gymnasium as gym
        import h5py
        import numpy as np
        import torch
        import yaml

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config

        import tg2_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401

        @hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
        def run(env_cfg, _agent_cfg: dict) -> None:
            traj_path = _resolve_traj_file(args_cli.traj_file)

            env_cfg.scene.num_envs = int(args_cli.num_envs)
            env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed
            env_cfg.simulate_stereo = True
            objects_dir_token = _resolve_objects_dir_token(args_cli.objects_dir)
            env_cfg.objects_dir = objects_dir_token
            if (
                hasattr(env_cfg, "valid_objects_dir")
                and isinstance(env_cfg.valid_objects_dir, list)
                and objects_dir_token not in env_cfg.valid_objects_dir
            ):
                env_cfg.valid_objects_dir.append(objects_dir_token)

            if not args_cli.keep_randomization:
                env_cfg.disable_out_of_reach_done = True
                env_cfg.disable_arm_randomization = True
                env_cfg.disable_dome_light_randomization = True

            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            if not env.env.simulate_stereo:
                raise RuntimeError("Offline replay expects simulate_stereo=True.")

            replay_count = 0
            all_stats: list[TrajStats] = []
            while simulation_app.is_running():
                replay_count += 1
                stats = _replay_one(env, traj_path, args_cli, simulation_app)
                all_stats.append(stats)
                print(
                    f"[Replay {replay_count}] steps={stats.replay_steps} mean_reward={stats.mean_reward:.3f} "
                    f"ever_lift_success_per_env={stats.ever_lift_success_per_env} file={stats.path}"
                )
                if args_cli.num_replays > 0 and replay_count >= args_cli.num_replays:
                    break

            os.makedirs(args_cli.save_dir, exist_ok=True)
            last_stats = (
                all_stats[-1]
                if all_stats
                else TrajStats(
                    path=traj_path,
                    replay_steps=0,
                    mean_reward=0.0,
                    ever_lift_success_per_env=[],
                )
            )
            mean_reward_across_replays = float(np.mean([s.mean_reward for s in all_stats])) if all_stats else 0.0
            summary = {
                "task": args_cli.task,
                "num_envs": int(args_cli.num_envs),
                "objects_dir": objects_dir_token,
                "traj_file": last_stats.path,
                "action_key": args_cli.action_key,
                "max_steps_per_traj": int(args_cli.max_steps_per_traj),
                "num_replays": int(args_cli.num_replays),
                "replays_completed": int(replay_count),
                "stop_on_all_done": bool(args_cli.stop_on_all_done),
                "keep_randomization": bool(args_cli.keep_randomization),
                "replay_steps_last": last_stats.replay_steps,
                "mean_reward_last": last_stats.mean_reward,
                "ever_lift_success_per_env_last": last_stats.ever_lift_success_per_env,
                "mean_reward_across_replays": mean_reward_across_replays,
            }
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(args_cli.save_dir, f"offline_replay_summary_{stamp}.yaml")
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(summary, f, default_flow_style=False)
            print(f"Saved offline replay summary: {out_path}")

        run()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
