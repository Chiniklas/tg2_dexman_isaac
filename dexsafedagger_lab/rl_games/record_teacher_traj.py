#!/usr/bin/env python3
"""Replay a teacher checkpoint and record trajectories.

This script mirrors the structure of:
`deployment_tg2_inspirehand/ws/src/inference_offline/tests/student_traj_recorder.py`,
but runs RL-Games teacher policy rollout in teacher env mode.

Teacher mode forced by this script:
  - distillation=False
  - simulate_stereo=False
  - disable_dome_light_randomization=True

# single object
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/rl_games/record_teacher_traj.py \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 1 \
  --checkpoint /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/pretrained_ckpts/teacher_eval/1wdf56lx/dexsafedagger_lstm.pth \
  --record_data \
  --deterministic \
  --num_episodes 5 \
  --max_steps_per_episode 120 \
  env.objects_dir=test_object \
  env.distillation=False \
  env.disable_dome_light_randomization=True

# batch object
python /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/rl_games/record_teacher_traj.py \
  --task dexsafedagger_tg2_inspirehand \
  --teacher_object_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/assets/teacher_eval \
  --teacher_policy_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/pretrained_ckpts/teacher_eval \
  --record_data \
  --deterministic \
  --num_episodes 5 \
  --max_steps_per_episode 120 \
  --headless \
  env.distillation=False \
  env.disable_dome_light_randomization=True
"""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

# CLI args parsed before Isaac Sim launch. Matches student_traj_recorder parser.
parser = argparse.ArgumentParser(description="Replay a teacher checkpoint and record trajectories.")
parser.add_argument("--video", action="store_true", default=False, help="Record viewport video in Isaac Sim.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded viewport video.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between viewport recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="dexsafedagger_tg2_inspirehand", help="Gym task name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to teacher checkpoint (single-object mode).")
parser.add_argument(
    "--teacher_object_dir",
    type=str,
    default=None,
    help="Root directory containing object folders (batch multi-object mode).",
)
parser.add_argument(
    "--teacher_policy_dir",
    type=str,
    default=None,
    help="Root directory containing per-object teacher policy folders (batch multi-object mode).",
)
parser.add_argument("--student_cfg", type=str, default=None, help="Path to student RL-Games yaml.")
parser.add_argument("--obs_key", type=str, default="policy", help="Observation key consumed by the policy.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=10,
    help="Target number of successful episodes (ever_lift_success=True). Use 0 for unlimited attempts.",
)
parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    default=0,
    help="Maximum steps per episode. Use 0 to run until all envs are done.",
)
parser.add_argument(
    "--max_trials_per_object",
    type=int,
    default=16,
    help="Batch pooled mode only: max attempted episodes per object. Use 0 for unlimited.",
)
parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions.")
parser.add_argument(
    "--save_dir",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher_trajs"),
    help="Output root directory. Run data is stored under <save_dir>/<timestamp>/data/<object_name>/.",
)
parser.add_argument("--record_data", action="store_true", default=False, help="Save trajectory data to disk.")
parser.add_argument("--create_video", action="store_true", default=False, help="Create per-env MP4 videos.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Remove known args for Hydra and launch app.
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_SIM_APP_CLOSED = False

import h5py
import numpy as np
import torch
import yaml
import gymnasium as gym

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401

import dexsafedagger_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401

_ENV_HOLDER = {"env": None}


def _shutdown_simulation(env: gym.Env | None = None) -> None:
    global _SIM_APP_CLOSED
    if env is not None:
        try:
            env.close()
        except Exception as exc:
            print(f"[WARN] Failed to close env cleanly: {exc}")
    if not _SIM_APP_CLOSED:
        simulation_app.close()
        _SIM_APP_CLOSED = True


def _register_rlgames_env() -> None:
    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception:
        pass
    try:
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: _ENV_HOLDER["env"]}
        )
    except Exception:
        # Some rl_games versions disallow re-register. Keep existing binding and rely on _ENV_HOLDER update.
        pass


def _to_numpy_stacked(items: list[torch.Tensor]) -> np.ndarray:
    if not items:
        return np.empty((0,), dtype=np.float32)
    return torch.stack(items, dim=0).cpu().numpy()


def _obs_to_tensor(obs: Any, obs_key: str) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().float()
    if isinstance(obs, dict):
        for key in (obs_key, "policy", "obs"):
            if key in obs:
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().float()
                return torch.as_tensor(value).detach().cpu().float()
        first_val = next(iter(obs.values()))
        return torch.as_tensor(first_val).detach().cpu().float()
    return torch.as_tensor(obs).detach().cpu().float()


def _extract_base_obs(obs: Any) -> Any:
    """Normalize gym reset/step outputs to the observation payload used by the policy."""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        return obs["obs"]
    return obs


def _compute_lift_success_flags(ov_env: Any) -> torch.Tensor:
    table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0)
    return ov_env.object_pos[:, 2] > lift_height_thresh


@dataclass
class RolloutStats:
    episode_rewards: list[float]
    episode_lengths: list[int]
    success_rates: list[float]
    per_env_ever_lift_success: list[list[bool]]
    episode_success_flags: list[bool]
    attempted_episodes: int
    successful_episodes: int


class TrajectoryRecorder:
    """Teacher trajectory recorder (student-recorder style, no image datasets)."""

    def __init__(
        self,
        run_dir: str,
        object_name: str,
        num_envs: int,
        obs_key: str,
        lift_height_thresh: float | None = None,
        skip_early_terminated: bool = True,
        save_success_only: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.obs_key = str(obs_key)
        self.lift_height_thresh = lift_height_thresh
        self.skip_early_terminated = bool(skip_early_terminated)
        self.save_success_only = bool(save_success_only)

        self.current_episode_idx: int | None = None
        self.saved_traj_count = 0
        self.successful_saved_traj_count = 0
        self.skipped_early_terminated_traj_count = 0
        self.skipped_unsuccessful_traj_count = 0

        self.last_flush_saved = 0
        self.last_flush_successful = 0
        self.last_flush_skipped = 0
        self.last_flush_skipped_unsuccessful = 0

        self.run_dir = str(run_dir)
        self.data_dir = os.path.join(self.run_dir, "data", str(object_name))
        os.makedirs(self.data_dir, exist_ok=True)

        self._reset_buffer()

    def start_episode(self, episode_idx: int) -> None:
        self.current_episode_idx = int(episode_idx)

    def _reset_buffer(self) -> None:
        self.buffer: dict[str, list[torch.Tensor]] = {
            "obs": [],
            "action": [],
            "prev_action": [],
            "reward": [],
            "done": [],
            "object_pos": [],
        }

    def record_step(
        self,
        obs: Any,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        object_pos: torch.Tensor | None,
    ) -> None:
        self.buffer["obs"].append(_obs_to_tensor(obs, self.obs_key))
        self.buffer["action"].append(action.detach().cpu())
        self.buffer["prev_action"].append(prev_action.detach().cpu())
        self.buffer["reward"].append(reward.detach().cpu())
        self.buffer["done"].append(done.detach().cpu())
        if object_pos is not None:
            self.buffer["object_pos"].append(object_pos.detach().cpu())

    def flush(self) -> None:
        self.last_flush_saved = 0
        self.last_flush_successful = 0
        self.last_flush_skipped = 0
        self.last_flush_skipped_unsuccessful = 0
        if len(self.buffer["reward"]) == 0:
            return

        payload = {k: _to_numpy_stacked(v) for k, v in self.buffer.items() if len(v) > 0}
        timestamp = str(datetime.now())
        steps = int(len(self.buffer["reward"]))
        episode_idx = self.current_episode_idx

        for env_id in range(self.num_envs):
            env_payload: dict[str, np.ndarray] = {}
            for key, value in payload.items():
                if value.ndim >= 2 and value.shape[0] == steps and value.shape[1] == self.num_envs:
                    env_payload[key] = value[:, env_id]
                else:
                    env_payload[key] = value

            if self.skip_early_terminated and "done" in env_payload:
                done_flags = np.asarray(env_payload["done"], dtype=np.bool_).reshape(-1)
                if done_flags.size > 1 and np.any(done_flags[:-1]):
                    self.last_flush_skipped += 1
                    self.skipped_early_terminated_traj_count += 1
                    continue

            ever_lift_success = False
            if self.lift_height_thresh is not None and "object_pos" in env_payload:
                obj = env_payload["object_pos"]
                if obj.ndim >= 2 and obj.shape[-1] >= 3:
                    ever_lift_success = bool(np.any(obj[..., 2] > self.lift_height_thresh))

            if self.save_success_only and not ever_lift_success:
                self.last_flush_skipped_unsuccessful += 1
                self.skipped_unsuccessful_traj_count += 1
                continue

            if episode_idx is None:
                raise RuntimeError("Recorder episode index is unset; call start_episode() before recording.")
            base = os.path.join(self.data_dir, f"traj_env_{env_id}_episode_{episode_idx}")
            h5_path = f"{base}.h5"
            yaml_path = f"{base}.yaml"
            meta = {
                "timestamp": timestamp,
                "env_id": int(env_id),
                "source_num_envs": self.num_envs,
                "steps": steps,
                "episode_idx": int(episode_idx) if episode_idx is not None else None,
                "lift_height_thresh": float(self.lift_height_thresh) if self.lift_height_thresh is not None else None,
                "ever_lift_success": bool(ever_lift_success),
                "datasets": {k: list(v.shape) for k, v in env_payload.items()},
            }

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, default_flow_style=False)

            with h5py.File(h5_path, "w") as f:
                for key, value in env_payload.items():
                    f.create_dataset(key, data=value, compression="gzip", compression_opts=3)
                f.create_dataset("ever_lift_success", data=np.array(bool(ever_lift_success), dtype=np.uint8))
                for key, value in meta.items():
                    if key != "datasets":
                        f.attrs[key] = value

            self.saved_traj_count += 1
            self.last_flush_saved += 1
            if ever_lift_success:
                self.successful_saved_traj_count += 1
                self.last_flush_successful += 1

        self._reset_buffer()


class TeacherPolicyReplayer:
    def __init__(
        self,
        env: gym.Env,
        agent_cfg: dict[str, Any],
        checkpoint_path: str,
        deterministic: bool,
    ) -> None:
        self.env = env
        self.ov_env = env.unwrapped
        self.agent_cfg = copy.deepcopy(agent_cfg)
        self.deterministic = bool(deterministic)

        self.num_envs = int(self.ov_env.num_envs)
        self.num_actions = int(self.ov_env.num_actions)
        self.device = torch.device(self.agent_cfg["params"]["config"]["device"])

        self.agent_cfg["params"]["load_checkpoint"] = True
        self.agent_cfg["params"]["load_path"] = checkpoint_path
        self.agent_cfg["params"]["config"]["num_actors"] = self.num_envs

        runner = Runner()
        runner.load(self.agent_cfg)
        self.agent: BasePlayer = runner.create_player()
        self.agent.restore(checkpoint_path)
        self.agent.reset()

        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.is_rnn = bool(self.agent.is_rnn)

    def reset_policy_state(self, obs: Any) -> None:
        self.prev_actions.zero_()
        base_obs = obs["obs"] if isinstance(obs, dict) and "obs" in obs else obs
        _ = self.agent.get_batch_size(base_obs, 1)
        if self.is_rnn:
            self.agent.init_rnn()

    @torch.no_grad()
    def get_actions(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        prev_action_snapshot = self.prev_actions.clone()
        obs_torch = self.agent.obs_to_torch(obs)
        action = self.agent.get_action(obs_torch, is_deterministic=self.deterministic)
        action = action.detach()
        self.prev_actions = action
        return action, prev_action_snapshot

    def reset_rnn_for_done_envs(self, done_indices: torch.Tensor) -> None:
        if not self.is_rnn or self.agent.states is None or done_indices.numel() == 0:
            return
        for state in self.agent.states:
            state[:, done_indices, :] = 0.0


def _run_rollouts(
    env: gym.Env,
    replayer: TeacherPolicyReplayer,
    num_episodes: int,
    max_steps_per_episode: int,
    recorder: TrajectoryRecorder | None,
) -> RolloutStats:
    stats = RolloutStats(
        episode_rewards=[],
        episode_lengths=[],
        success_rates=[],
        per_env_ever_lift_success=[],
        episode_success_flags=[],
        attempted_episodes=0,
        successful_episodes=0,
    )

    episode_idx = 0
    successful_episode_count = 0
    prefetched_reset_obs = None
    while simulation_app.is_running():
        if num_episodes > 0 and successful_episode_count >= num_episodes:
            break

        if prefetched_reset_obs is None:
            obs = env.reset()
        else:
            obs = prefetched_reset_obs
            prefetched_reset_obs = None
        obs = _extract_base_obs(obs)
        replayer.reset_policy_state(obs)

        episode_idx += 1
        if recorder is not None:
            recorder.start_episode(episode_idx)
        ep_reward = 0.0
        ep_length = 0
        dones = torch.zeros((replayer.num_envs,), dtype=torch.bool, device=replayer.device)
        ever_lift_success = torch.zeros((replayer.num_envs,), dtype=torch.bool, device=replayer.device)
        timed_out = False

        while simulation_app.is_running() and not torch.all(dones):
            if max_steps_per_episode > 0 and ep_length >= max_steps_per_episode:
                timed_out = True
                break

            action, prev_action = replayer.get_actions(obs)
            obs, reward, dones, _ = env.step(action)

            if recorder is not None:
                recorder.record_step(
                    obs=obs,
                    action=action,
                    prev_action=prev_action,
                    reward=reward,
                    done=dones,
                    object_pos=getattr(replayer.ov_env, "object_pos", None),
                )

            ep_reward += float(reward.mean().item())
            ep_length += 1
            ever_lift_success |= _compute_lift_success_flags(replayer.ov_env)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            replayer.reset_rnn_for_done_envs(done_indices)

        if recorder is not None:
            recorder.flush()
            print(
                f"[Episode {episode_idx}] len={ep_length} "
                f"saved={recorder.last_flush_saved} "
                f"saved_success={recorder.last_flush_successful} "
                f"skipped_early_term={recorder.last_flush_skipped} "
                f"skipped_unsuccessful={recorder.last_flush_skipped_unsuccessful} "
                f"total_saved={recorder.saved_traj_count} "
                f"total_saved_success={recorder.successful_saved_traj_count}"
            )

        success_rate = float(ever_lift_success.float().mean().item())
        per_env_flags = [bool(v) for v in ever_lift_success.detach().cpu().tolist()]
        episode_success = bool(any(per_env_flags))
        if episode_success:
            successful_episode_count += 1

        stats.episode_rewards.append(ep_reward)
        stats.episode_lengths.append(ep_length)
        stats.success_rates.append(success_rate)
        stats.per_env_ever_lift_success.append(per_env_flags)
        stats.episode_success_flags.append(episode_success)
        stats.attempted_episodes = int(episode_idx)
        stats.successful_episodes = int(successful_episode_count)

        print(
            f"[Episode {episode_idx}] reward={ep_reward:.2f} "
            f"length={ep_length} ever_lift_success_per_env={per_env_flags} "
            f"episode_success={episode_success} "
            f"successful_episodes={successful_episode_count}/{num_episodes if num_episodes > 0 else 'unlimited'}"
        )

        # Enforce an explicit env reset at timeout episode boundaries.
        if timed_out and simulation_app.is_running() and (
            num_episodes == 0 or successful_episode_count < num_episodes
        ):
            prefetched_reset_obs = env.reset()

    return stats


def _sanitize_path_component(raw: Any) -> str:
    value = str(raw).strip()
    if value == "":
        return "unknown_object"
    return value.replace("/", "_").replace("\\", "_")


def _checkpoint_object_hint(checkpoint_path: str) -> str | None:
    parent = os.path.basename(os.path.dirname(str(checkpoint_path).rstrip("/\\")))
    if parent.strip() == "":
        return None
    return parent


def _resolve_record_object_name(ov_env: Any, objects_dir: Any, checkpoint_path: str) -> tuple[str, str]:
    objects_dir_token = str(objects_dir)
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    usd_root = os.path.join(module_dir, "assets", objects_dir_token, "USD")

    def _exists_under_usd_root(name: str) -> bool:
        if name.strip() == "":
            return False
        return os.path.isdir(os.path.join(usd_root, name))

    ckpt_hint = _checkpoint_object_hint(checkpoint_path)
    if ckpt_hint is not None and _exists_under_usd_root(ckpt_hint):
        return ckpt_hint, "checkpoint_parent"

    obj_names = getattr(ov_env, "object_names", None)
    obj_indices = getattr(ov_env, "multi_object_idx", None)
    if isinstance(obj_names, list) and len(obj_names) > 0:
        idx0 = 0
        try:
            if obj_indices is not None:
                idx_tensor = torch.as_tensor(obj_indices).flatten()
                if idx_tensor.numel() > 0:
                    idx0 = int(idx_tensor[0].item())
            idx0 = max(0, min(idx0, len(obj_names) - 1))
            runtime_name = str(obj_names[idx0])
            if runtime_name.strip() != "":
                return _sanitize_path_component(runtime_name), "env.object_names"
        except Exception:
            pass
        fallback_name = str(obj_names[0])
        if fallback_name.strip() != "":
            return _sanitize_path_component(fallback_name), "env.object_names_fallback"

    if ckpt_hint is not None:
        return _sanitize_path_component(ckpt_hint), "checkpoint_parent_unverified"

    raw = str(objects_dir)
    normalized = raw.rstrip("/\\")
    base = os.path.basename(normalized) if normalized else raw
    return _sanitize_path_component(base), "objects_dir_token"


def _list_named_subdirs(path: pathlib.Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return sorted([p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _normalize_teacher_object_root(teacher_object_dir: str) -> pathlib.Path:
    root = pathlib.Path(teacher_object_dir).expanduser().resolve()
    usd_dir = root / "USD"
    return usd_dir if usd_dir.is_dir() else root


def _validate_teacher_policy_object_dirs(teacher_policy_dir: str, teacher_object_dir: str) -> tuple[list[str], pathlib.Path]:
    policy_path = pathlib.Path(teacher_policy_dir).expanduser().resolve()
    object_root = _normalize_teacher_object_root(teacher_object_dir)

    policy_names = _list_named_subdirs(policy_path)
    object_names = _list_named_subdirs(object_root)

    policy_set = set(policy_names)
    object_set = set(object_names)
    if policy_set != object_set:
        missing_in_policy = sorted(object_set - policy_set)
        missing_in_object = sorted(policy_set - object_set)
        raise ValueError(
            "Teacher policy/object folder name mismatch.\n"
            f"Missing policy folders for objects: {missing_in_policy}\n"
            f"Missing object folders for policies: {missing_in_object}"
        )
    return sorted(policy_names), object_root


def _resolve_teacher_checkpoint(policy_root_dir: str, object_name: str) -> str:
    policy_object_dir = pathlib.Path(policy_root_dir).expanduser().resolve() / object_name
    if not policy_object_dir.is_dir():
        raise FileNotFoundError(f"Policy folder missing for object '{object_name}': {policy_object_dir}")

    preferred = policy_object_dir / "dexsafedagger_lstm.pth"
    if preferred.is_file():
        return str(preferred)

    direct_candidates = sorted([p for p in policy_object_dir.glob("*.pth") if p.is_file()])
    if len(direct_candidates) == 1:
        return str(direct_candidates[0])
    if len(direct_candidates) > 1:
        raise ValueError(
            f"Multiple checkpoint files found in {policy_object_dir}: {[str(p.name) for p in direct_candidates]}"
        )

    recursive_candidates = sorted([p for p in policy_object_dir.rglob("*.pth") if p.is_file()])
    if len(recursive_candidates) == 1:
        return str(recursive_candidates[0])
    if len(recursive_candidates) == 0:
        raise FileNotFoundError(f"No checkpoint (*.pth) found under {policy_object_dir}")
    raise ValueError(
        f"Multiple checkpoint files found under {policy_object_dir}: {[str(p) for p in recursive_candidates]}"
    )


def _prepare_single_object_override(object_source_root: pathlib.Path, object_name: str, run_stamp: str) -> tuple[str, pathlib.Path]:
    assets_dir = pathlib.Path(__file__).resolve().parents[1] / "assets"
    target_dir_name = f"__teacher_record_single_{run_stamp}_{object_name}"
    target_root = assets_dir / target_dir_name
    if target_root.exists():
        shutil.rmtree(target_root)

    target_usd_dir = target_root / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    source_object_dir = object_source_root / object_name
    if not source_object_dir.is_dir():
        raise FileNotFoundError(f"Object folder missing for '{object_name}': {source_object_dir}")
    (target_usd_dir / object_name).symlink_to(source_object_dir, target_is_directory=True)
    return target_dir_name, target_root


def _prepare_multi_object_override(
    object_source_root: pathlib.Path, object_names: list[str], run_stamp: str
) -> tuple[str, pathlib.Path]:
    assets_dir = pathlib.Path(__file__).resolve().parents[1] / "assets"
    target_dir_name = f"__teacher_record_multi_{run_stamp}"
    target_root = assets_dir / target_dir_name
    if target_root.exists():
        shutil.rmtree(target_root)

    target_usd_dir = target_root / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    for object_name in object_names:
        source_object_dir = object_source_root / object_name
        if not source_object_dir.is_dir():
            raise FileNotFoundError(f"Object folder missing for '{object_name}': {source_object_dir}")
        (target_usd_dir / object_name).symlink_to(source_object_dir, target_is_directory=True)
    return target_dir_name, target_root


def _as_bool_mask(values: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    try:
        mask = torch.as_tensor(values, device=device, dtype=torch.bool).flatten()
    except Exception:
        return torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if mask.numel() == 1:
        mask = mask.repeat(num_envs)
    if mask.numel() < num_envs:
        padded = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        padded[: mask.numel()] = mask
        return padded
    return mask[:num_envs]


def _resolve_eval_object_names_and_idx(
    eval_env: Any, num_envs: int, device: torch.device, fallback_names: list[str] | None = None
) -> tuple[list[str], torch.Tensor]:
    eval_object_names = list(getattr(eval_env, "object_names", []))
    if len(eval_object_names) == 0 and fallback_names is not None and len(fallback_names) > 0:
        eval_object_names = list(fallback_names)
    if len(eval_object_names) == 0:
        eval_object_names = ["object_0"]
    eval_object_names = [str(name) for name in eval_object_names]

    eval_object_idx = getattr(eval_env, "multi_object_idx", None)
    if eval_object_idx is None:
        eval_object_idx = torch.zeros((num_envs,), dtype=torch.long, device=device)
    else:
        eval_object_idx = torch.as_tensor(eval_object_idx, dtype=torch.long, device=device).flatten()
        if eval_object_idx.numel() < num_envs:
            padded = torch.zeros((num_envs,), dtype=torch.long, device=device)
            padded[: eval_object_idx.numel()] = eval_object_idx
            eval_object_idx = padded
        elif eval_object_idx.numel() > num_envs:
            eval_object_idx = eval_object_idx[:num_envs]
    eval_object_idx = torch.clamp(eval_object_idx, min=0, max=len(eval_object_names) - 1)
    return eval_object_names, eval_object_idx


def _slice_env_obs(obs: Any, env_id: int) -> Any:
    if isinstance(obs, dict):
        sliced = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value[env_id : env_id + 1]
            elif isinstance(value, np.ndarray):
                sliced[key] = value[env_id : env_id + 1]
            else:
                sliced[key] = value
        return sliced
    if isinstance(obs, torch.Tensor):
        return obs[env_id : env_id + 1]
    if isinstance(obs, np.ndarray):
        return obs[env_id : env_id + 1]
    return obs


def _collect_single_object(
    env_cfg_template: Any,
    agent_cfg_template: dict[str, Any],
    checkpoint_path: str,
    run_dir: str,
    object_name_hint: str | None = None,
    object_name_source_hint: str | None = None,
    objects_dir_override: str | None = None,
) -> dict[str, Any]:
    env_cfg = copy.deepcopy(env_cfg_template)
    agent_cfg = copy.deepcopy(agent_cfg_template)

    if objects_dir_override is not None:
        env_cfg.objects_dir = objects_dir_override
        if hasattr(env_cfg, "valid_objects_dir") and isinstance(env_cfg.valid_objects_dir, list):
            if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
                env_cfg.valid_objects_dir.append(env_cfg.objects_dir)

    print(
        f"[INFO] Creating env for object={object_name_hint if object_name_hint is not None else 'auto'} "
        f"objects_dir={getattr(env_cfg, 'objects_dir', 'n/a')}"
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    _ENV_HOLDER["env"] = env
    print("[INFO] Env ready.")

    try:
        resolved_checkpoint_path = str(retrieve_file_path(checkpoint_path))
        if object_name_hint is None:
            object_name, object_name_source = _resolve_record_object_name(
                ov_env=env.unwrapped,
                objects_dir=getattr(env.unwrapped.cfg, "objects_dir", "unknown_object"),
                checkpoint_path=resolved_checkpoint_path,
            )
        else:
            object_name = _sanitize_path_component(object_name_hint)
            object_name_source = str(object_name_source_hint or "explicit")

        print(f"[INFO] Object collection start: {object_name} (source={object_name_source})")
        print(f"[INFO] Checkpoint: {resolved_checkpoint_path}")
        print(
            "[INFO] Teacher env settings: "
            f"distillation={getattr(env.unwrapped.cfg, 'distillation', 'n/a')} "
            f"simulate_stereo={getattr(env.unwrapped.cfg, 'simulate_stereo', 'n/a')} "
            f"disable_dome_light_randomization={getattr(env.unwrapped.cfg, 'disable_dome_light_randomization', 'n/a')}"
        )

        replayer = TeacherPolicyReplayer(
            env=env,
            agent_cfg=agent_cfg,
            checkpoint_path=resolved_checkpoint_path,
            deterministic=args_cli.deterministic,
        )
        print("[INFO] Teacher policy loaded.")

        recorder = None
        if args_cli.record_data:
            lift_height_thresh = None
            if hasattr(env.unwrapped, "cfg") and hasattr(env.unwrapped.cfg, "table_cfg"):
                table_center_z = env.unwrapped.cfg.table_cfg.init_state.pos[2]
                table_top_z = table_center_z + 0.5 * env.unwrapped.cfg.table_size_z
                lift_height_thresh = float(table_top_z + getattr(env.unwrapped.cfg, "object_height_thresh", 0.0))

            recorder = TrajectoryRecorder(
                run_dir=run_dir,
                object_name=object_name,
                num_envs=replayer.num_envs,
                obs_key=args_cli.obs_key,
                lift_height_thresh=lift_height_thresh,
                skip_early_terminated=True,
                save_success_only=True,
            )
            print(f"[INFO] Recording enabled (success-only). Output dir: {recorder.data_dir}")

        stats = _run_rollouts(
            env=env,
            replayer=replayer,
            num_episodes=args_cli.num_episodes,
            max_steps_per_episode=args_cli.max_steps_per_episode,
            recorder=recorder,
        )

        result = {
            "episode_rewards": stats.episode_rewards,
            "episode_lengths": stats.episode_lengths,
            "success_rates": stats.success_rates,
            "per_env_ever_lift_success": stats.per_env_ever_lift_success,
            "episode_success_flags": stats.episode_success_flags,
            "mean_reward": float(np.mean(stats.episode_rewards)) if stats.episode_rewards else 0.0,
            "std_reward": float(np.std(stats.episode_rewards)) if stats.episode_rewards else 0.0,
            "mean_length": float(np.mean(stats.episode_lengths)) if stats.episode_lengths else 0.0,
            "std_length": float(np.std(stats.episode_lengths)) if stats.episode_lengths else 0.0,
            "mean_success": float(np.mean(stats.success_rates)) if stats.success_rates else 0.0,
            "std_success": float(np.std(stats.success_rates)) if stats.success_rates else 0.0,
            "attempted_episodes": int(stats.attempted_episodes),
            "successful_episodes": int(stats.successful_episodes),
            "target_successful_episodes": int(args_cli.num_episodes),
            "checkpoint": resolved_checkpoint_path,
            "task": args_cli.task,
            "num_envs": int(replayer.num_envs),
            "objects_dir": str(env.unwrapped.cfg.objects_dir),
            "object_name": str(object_name),
            "object_name_source": str(object_name_source),
            "teacher_mode": True,
            "deterministic": bool(args_cli.deterministic),
            "obs_key": str(args_cli.obs_key),
            "student_cfg_arg": args_cli.student_cfg,
            "num_episodes_arg": int(args_cli.num_episodes),
            "max_steps_per_episode_arg": int(args_cli.max_steps_per_episode),
            "create_video_arg": bool(args_cli.create_video),
            "record_data": bool(args_cli.record_data),
            "save_success_only": True,
            "run_dir": str(run_dir),
        }
        if recorder is not None:
            result.update(
                {
                    "record_run_dir": recorder.run_dir,
                    "saved_traj_count": int(recorder.saved_traj_count),
                    "successful_saved_traj_count": int(recorder.successful_saved_traj_count),
                    "skipped_early_terminated_traj_count": int(recorder.skipped_early_terminated_traj_count),
                    "skipped_unsuccessful_traj_count": int(recorder.skipped_unsuccessful_traj_count),
                }
            )
        return result
    finally:
        _ENV_HOLDER["env"] = None
        env.close()


def _collect_multi_object_teacher_pool(
    env_cfg_template: Any,
    agent_cfg_template: dict[str, Any],
    run_dir: str,
    objects_dir_override: str,
    object_names: list[str],
    teacher_policy_dir: str,
) -> dict[str, Any]:
    env_cfg = copy.deepcopy(env_cfg_template)
    if hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = len(object_names)
    env_cfg.objects_dir = objects_dir_override
    if hasattr(env_cfg, "valid_objects_dir") and isinstance(env_cfg.valid_objects_dir, list):
        if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
            env_cfg.valid_objects_dir.append(env_cfg.objects_dir)
    env_cfg.multi_object_eval = True

    print(
        f"[INFO] Creating pooled env for batch mode: num_envs={len(object_names)} objects_dir={env_cfg.objects_dir}"
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    rl_device = torch.device(agent_cfg_template["params"]["config"]["device"])
    clip_obs = agent_cfg_template["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg_template["params"]["env"].get("clip_actions", float("inf"))
    env = RlGamesVecEnvWrapper(env, str(rl_device), clip_obs, clip_actions)
    _ENV_HOLDER["env"] = env
    print("[INFO] Pooled env ready.")

    try:
        ov_env = env.unwrapped
        num_envs = int(ov_env.num_envs)
        if num_envs != len(object_names):
            raise RuntimeError(f"Expected {len(object_names)} envs in pooled mode, but got {num_envs}.")

        eval_object_names, eval_object_idx = _resolve_eval_object_names_and_idx(
            ov_env, num_envs, rl_device, fallback_names=object_names
        )
        checkpoint_map = {
            object_name: str(retrieve_file_path(_resolve_teacher_checkpoint(teacher_policy_dir, object_name)))
            for object_name in eval_object_names
        }

        players_by_name: dict[str, BasePlayer] = {}
        for object_name in eval_object_names:
            cfg = copy.deepcopy(agent_cfg_template)
            cfg["params"]["load_checkpoint"] = True
            cfg["params"]["load_path"] = checkpoint_map[object_name]
            cfg["params"]["config"]["num_actors"] = num_envs
            runner = Runner()
            runner.load(cfg)
            player: BasePlayer = runner.create_player()
            player.restore(checkpoint_map[object_name])
            player.reset()
            players_by_name[object_name] = player
            print(f"[INFO] Loaded teacher player for {object_name}: {checkpoint_map[object_name]}")

        env_object_names = []
        for env_id in range(num_envs):
            obj_idx = int(eval_object_idx[env_id].item())
            env_object_names.append(eval_object_names[obj_idx])

        lift_height_thresh = None
        if hasattr(ov_env, "cfg") and hasattr(ov_env.cfg, "table_cfg"):
            table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
            table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
            lift_height_thresh = float(table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0))

        target = int(args_cli.num_episodes)
        if target <= 0:
            raise ValueError("Batch pooled mode requires --num_episodes > 0.")
        max_trials_per_object = int(args_cli.max_trials_per_object)

        per_object_attempted = {name: 0 for name in eval_object_names}
        per_object_successful = {name: 0 for name in eval_object_names}
        per_object_rewards = {name: [] for name in eval_object_names}
        per_object_lengths = {name: [] for name in eval_object_names}
        per_object_success_flags = {name: [] for name in eval_object_names}
        per_object_saved = {name: 0 for name in eval_object_names}
        per_object_saved_success = {name: 0 for name in eval_object_names}
        per_object_skipped_unsuccessful = {name: 0 for name in eval_object_names}

        recorders_by_env: dict[int, TrajectoryRecorder] = {}
        for env_id, object_name in enumerate(env_object_names):
            recorders_by_env[env_id] = TrajectoryRecorder(
                run_dir=run_dir,
                object_name=object_name,
                num_envs=1,
                obs_key=args_cli.obs_key,
                lift_height_thresh=lift_height_thresh,
                skip_early_terminated=True,
                save_success_only=True,
            )

        def _object_done(object_name: str) -> bool:
            if per_object_successful[object_name] >= target:
                return True
            if max_trials_per_object > 0 and per_object_attempted[object_name] >= max_trials_per_object:
                return True
            return False

        global_episode_idx = 0
        while simulation_app.is_running():
            if all(_object_done(name) for name in eval_object_names):
                break

            global_episode_idx += 1
            obs = env.reset()
            obs_base = _extract_base_obs(obs)

            for player in players_by_name.values():
                _ = player.get_batch_size(obs_base, 1)
                if player.is_rnn:
                    player.init_rnn()

            active_env_mask = torch.zeros((num_envs,), dtype=torch.bool, device=rl_device)
            for env_id, object_name in enumerate(env_object_names):
                if not _object_done(object_name):
                    active_env_mask[env_id] = True
                    per_object_attempted[object_name] += 1
                    recorders_by_env[env_id].start_episode(per_object_attempted[object_name])

            dones = torch.zeros((num_envs,), dtype=torch.bool, device=rl_device)
            ever_lift_success = torch.zeros((num_envs,), dtype=torch.bool, device=rl_device)
            ep_rewards = torch.zeros((num_envs,), dtype=torch.float32, device=rl_device)
            ep_lengths = torch.zeros((num_envs,), dtype=torch.long, device=rl_device)
            prev_actions = torch.zeros((num_envs, int(ov_env.num_actions)), dtype=torch.float32, device=rl_device)
            step_idx = 0

            while simulation_app.is_running() and not bool(dones.all().item()):
                if args_cli.max_steps_per_episode > 0 and step_idx >= args_cli.max_steps_per_episode:
                    break

                actions = None
                for obj_idx, object_name in enumerate(eval_object_names):
                    obj_mask = eval_object_idx == obj_idx
                    if not bool(obj_mask.any().item()):
                        continue
                    player = players_by_name[object_name]
                    obs_t = player.obs_to_torch(obs_base)
                    action_by_obj = player.get_action(obs_t, is_deterministic=args_cli.deterministic)
                    action_by_obj = torch.as_tensor(action_by_obj, device=rl_device)
                    if actions is None:
                        actions = torch.zeros_like(action_by_obj)
                    actions[obj_mask] = action_by_obj[obj_mask]
                if actions is None:
                    raise RuntimeError("No teacher actions were produced in pooled batch mode.")

                active_before_step = ~dones
                obs_next, reward, step_dones, _ = env.step(actions)

                reward_t = torch.as_tensor(reward, device=rl_device).flatten()
                if reward_t.numel() == 1:
                    reward_t = reward_t.repeat(num_envs)
                elif reward_t.numel() < num_envs:
                    padded_reward = torch.zeros((num_envs,), dtype=torch.float32, device=rl_device)
                    padded_reward[: reward_t.numel()] = reward_t
                    reward_t = padded_reward
                else:
                    reward_t = reward_t[:num_envs]

                step_dones_mask = _as_bool_mask(step_dones, num_envs, rl_device)

                object_pos_all = getattr(ov_env, "object_pos", None)
                for env_id in range(num_envs):
                    if not bool(active_env_mask[env_id].item()) or not bool(active_before_step[env_id].item()):
                        continue
                    recorders_by_env[env_id].record_step(
                        obs=_slice_env_obs(obs_next, env_id),
                        action=actions[env_id : env_id + 1],
                        prev_action=prev_actions[env_id : env_id + 1],
                        reward=reward_t[env_id : env_id + 1],
                        done=step_dones_mask[env_id : env_id + 1],
                        object_pos=object_pos_all[env_id : env_id + 1] if object_pos_all is not None else None,
                    )

                ep_rewards[active_before_step] += reward_t[active_before_step]
                ep_lengths[active_before_step] += 1
                ever_lift_success |= _compute_lift_success_flags(ov_env)
                dones |= step_dones_mask

                done_indices = step_dones_mask.nonzero(as_tuple=False).flatten()
                if done_indices.numel() > 0:
                    for player in players_by_name.values():
                        if player.is_rnn and player.states is not None:
                            for state in player.states:
                                state[:, done_indices, :] = 0.0

                prev_actions = actions
                obs_base = _extract_base_obs(obs_next)
                step_idx += 1

            progress_parts = []
            for env_id, object_name in enumerate(env_object_names):
                if not bool(active_env_mask[env_id].item()):
                    if per_object_successful[object_name] >= target:
                        progress_parts.append(f"{object_name}:{per_object_successful[object_name]}/{target}(done)")
                    elif max_trials_per_object > 0 and per_object_attempted[object_name] >= max_trials_per_object:
                        progress_parts.append(
                            f"{object_name}:{per_object_successful[object_name]}/{target}(capped)"
                        )
                    else:
                        progress_parts.append(f"{object_name}:{per_object_successful[object_name]}/{target}(inactive)")
                    continue

                recorder = recorders_by_env[env_id]
                recorder.flush()
                success = bool(ever_lift_success[env_id].item())
                if success:
                    per_object_successful[object_name] += 1
                per_object_rewards[object_name].append(float(ep_rewards[env_id].item()))
                per_object_lengths[object_name].append(int(ep_lengths[env_id].item()))
                per_object_success_flags[object_name].append(success)
                per_object_saved[object_name] += int(recorder.last_flush_saved)
                per_object_saved_success[object_name] += int(recorder.last_flush_successful)
                per_object_skipped_unsuccessful[object_name] += int(recorder.last_flush_skipped_unsuccessful)
                progress_parts.append(f"{object_name}:{per_object_successful[object_name]}/{target}")

            print(f"[Batch Episode {global_episode_idx}] " + " ".join(progress_parts))

        per_object_results: dict[str, dict[str, Any]] = {}
        for object_name in eval_object_names:
            rewards = per_object_rewards[object_name]
            lengths = per_object_lengths[object_name]
            success_flags = per_object_success_flags[object_name]
            per_object_results[object_name] = {
                "checkpoint": checkpoint_map[object_name],
                "object_name": object_name,
                "attempted_episodes": int(per_object_attempted[object_name]),
                "successful_episodes": int(per_object_successful[object_name]),
                "target_successful_episodes": int(target),
                "max_trials_per_object": int(max_trials_per_object),
                "reached_target": bool(per_object_successful[object_name] >= target),
                "hit_trial_limit": bool(
                    max_trials_per_object > 0 and per_object_attempted[object_name] >= max_trials_per_object
                ),
                "episode_rewards": rewards,
                "episode_lengths": lengths,
                "episode_success_flags": success_flags,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "std_reward": float(np.std(rewards)) if rewards else 0.0,
                "mean_length": float(np.mean(lengths)) if lengths else 0.0,
                "std_length": float(np.std(lengths)) if lengths else 0.0,
                "mean_success": float(np.mean([float(v) for v in success_flags])) if success_flags else 0.0,
                "saved_traj_count": int(per_object_saved[object_name]),
                "successful_saved_traj_count": int(per_object_saved_success[object_name]),
                "skipped_unsuccessful_traj_count": int(per_object_skipped_unsuccessful[object_name]),
            }

        return {
            "mode": "multi_object_multi_teacher_pooled",
            "task": args_cli.task,
            "num_envs": num_envs,
            "objects_dir": str(env_cfg.objects_dir),
            "object_order": eval_object_names,
            "target_successful_episodes_per_object": int(target),
            "max_trials_per_object": int(max_trials_per_object),
            "objects_reached_target_count": int(
                sum(1 for name in eval_object_names if per_object_successful[name] >= target)
            ),
            "attempted_episodes_total": int(sum(v["attempted_episodes"] for v in per_object_results.values())),
            "successful_episodes_total": int(sum(v["successful_episodes"] for v in per_object_results.values())),
            "saved_traj_count_total": int(sum(v["saved_traj_count"] for v in per_object_results.values())),
            "successful_saved_traj_count_total": int(
                sum(v["successful_saved_traj_count"] for v in per_object_results.values())
            ),
            "deterministic": bool(args_cli.deterministic),
            "max_steps_per_episode_arg": int(args_cli.max_steps_per_episode),
            "record_data": bool(args_cli.record_data),
            "save_success_only": True,
            "run_dir": str(run_dir),
            "per_object": per_object_results,
        }
    finally:
        _ENV_HOLDER["env"] = None
        env.close()


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    if args_cli.num_episodes < 0:
        raise ValueError("--num_episodes must be >= 0.")
    if args_cli.max_steps_per_episode < 0:
        raise ValueError("--max_steps_per_episode must be >= 0.")
    if args_cli.max_trials_per_object < 0:
        raise ValueError("--max_trials_per_object must be >= 0.")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed

    # Force teacher rollout mode.
    if hasattr(env_cfg, "distillation"):
        env_cfg.distillation = False
    if hasattr(env_cfg, "simulate_stereo"):
        env_cfg.simulate_stereo = False
    if hasattr(env_cfg, "disable_dome_light_randomization"):
        env_cfg.disable_dome_light_randomization = True

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args_cli.save_dir, run_stamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Run dir: {run_dir}")
    _register_rlgames_env()

    batch_mode = args_cli.teacher_policy_dir is not None or args_cli.teacher_object_dir is not None
    if batch_mode:
        if args_cli.teacher_policy_dir is None or args_cli.teacher_object_dir is None:
            raise ValueError("Batch mode requires both --teacher_policy_dir and --teacher_object_dir.")
        if args_cli.checkpoint is not None:
            raise ValueError("Do not pass --checkpoint with --teacher_policy_dir/--teacher_object_dir.")

        object_names, object_source_root = _validate_teacher_policy_object_dirs(
            args_cli.teacher_policy_dir, args_cli.teacher_object_dir
        )
        print(
            f"[INFO] Batch mode enabled: {len(object_names)} objects, "
            f"target_successes_per_object={args_cli.num_episodes}"
        )
        if args_cli.num_envs is not None and int(args_cli.num_envs) != len(object_names):
            print(
                f"[WARN] Overriding --num_envs={args_cli.num_envs} to {len(object_names)} "
                "for pooled multi-object batch mode."
            )
        objects_dir_override, temp_root = _prepare_multi_object_override(
            object_source_root=object_source_root,
            object_names=object_names,
            run_stamp=run_stamp,
        )
        print(f"[INFO] Using pooled objects override: {objects_dir_override}")
        try:
            results = _collect_multi_object_teacher_pool(
                env_cfg_template=env_cfg,
                agent_cfg_template=agent_cfg,
                run_dir=run_dir,
                objects_dir_override=objects_dir_override,
                object_names=object_names,
                teacher_policy_dir=args_cli.teacher_policy_dir,
            )
        finally:
            if temp_root.exists():
                shutil.rmtree(temp_root)
        results.update(
            {
                "teacher_object_dir": str(pathlib.Path(args_cli.teacher_object_dir).expanduser().resolve()),
                "teacher_policy_dir": str(pathlib.Path(args_cli.teacher_policy_dir).expanduser().resolve()),
                "save_dir": str(args_cli.save_dir),
            }
        )
    else:
        if args_cli.checkpoint is None:
            raise ValueError("Single-object mode requires --checkpoint.")
        results = _collect_single_object(
            env_cfg_template=env_cfg,
            agent_cfg_template=agent_cfg,
            checkpoint_path=args_cli.checkpoint,
            run_dir=run_dir,
        )
        results.update({"mode": "single_object_single_teacher", "save_dir": str(args_cli.save_dir)})

    out_path = os.path.join(run_dir, "teacher_replay_results.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    print(f"Saved replay summary: {out_path}")
    print("[INFO] Collection target reached. Shutting down simulator.")
    _shutdown_simulation()


if __name__ == "__main__":
    try:
        main()
    finally:
        _shutdown_simulation()
