#!/usr/bin/env python3
"""Replay a student policy and record trajectories.

This script is intentionally standalone. It does not import helper classes from
`distillation_new/replay.py` or `distillation_new/data_recorder.py`.

# default student checkpoint:
/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/runs/dextrah-tg2-inspirehand-both_26-10-53-04/nn/dextrah_student_safe_dagger.pth.pth

# default object dir:
/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/distill_multi_objects

# command:
python /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/deployment_tg2_inspirehand/ws/src/inference_offline/tests/student_traj_recorder.py   
--checkpoint /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/runs/dextrah-tg2-inspirehand-both_26-10-53-04/nn/dextrah_student_safe_dagger.pth.pth   
--task dextrah_tg2_inspirehand   
--num_envs 4   
--enable_cameras   
--record_data   
--create_video   
--num_episodes 1   
--max_steps_per_episode 120   
--max_records_per_file 120   
--deterministic   
--headless   
env.objects_dir=distill_multi_objects   
env.distillation=True



"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

# CLI args that should be parsed before Isaac Sim launch.
parser = argparse.ArgumentParser(description="Replay a student checkpoint and record trajectories.")
parser.add_argument("--video", action="store_true", default=False, help="Record viewport video in Isaac Sim.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded viewport video.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between viewport recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="dextrah_tg2_inspirehand", help="Gym task name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint.")
parser.add_argument("--student_cfg", type=str, default=None, help="Path to student RL-Games yaml.")
parser.add_argument("--obs_key", type=str, default="policy", help="Observation key consumed by the student.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of replay episodes.")
parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    default=0,
    help="Max steps per episode. Use 0 to run until all envs are done.",
)
parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions (mu).")
parser.add_argument("--save_dir", type=str, default="student_replay_logs", help="Output directory.")
parser.add_argument("--record_data", action="store_true", default=False, help="Save trajectory data to disk.")
parser.add_argument(
    "--max_records_per_file",
    type=int,
    default=1000,
    help="Maximum replay steps per recorded HDF5 file.",
)
parser.add_argument("--create_video", action="store_true", default=False, help="Create per-env MP4 videos.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Remove known args for Hydra and launch app.
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch
import yaml
import gymnasium as gym
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import dextrah_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401


def adjust_state_dict_keys(checkpoint_state_dict: dict[str, Any], model_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Best-effort remap for checkpoints with/without `_orig_mod` in key paths."""
    adjusted_state_dict: dict[str, Any] = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        with_orig = ".".join(parts)
        if with_orig in model_state_dict:
            adjusted_state_dict[with_orig] = value
            continue

        no_orig = key.replace("_orig_mod.", "")
        if no_orig in model_state_dict:
            adjusted_state_dict[no_orig] = value
            continue

        adjusted_state_dict[key] = value
    return adjusted_state_dict


def register_stereo_transformer_builder() -> None:
    """Register rl_games network builder required by stereo transformer config."""
    try:
        from dextrah_lab.distillation_new.a2c_stereo_transformer import (
            A2CBuilder as A2CStereoTransformerBuilder,
        )
    except ImportError:
        from dextrah_lab.distillation.a2c_stereo_transformer import (  # type: ignore
            A2CBuilder as A2CStereoTransformerBuilder,
        )
    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    needle = pathlib.Path("dextrah_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_stereo_transformer.yaml")
    for parent in [start, *start.parents]:
        if (parent / needle).exists():
            return parent
    raise FileNotFoundError(
        "Could not locate repository root containing "
        f"'{needle.as_posix()}'. Please pass --student_cfg explicitly."
    )


def _to_numpy_stacked(items: list[torch.Tensor]) -> np.ndarray:
    if not items:
        return np.empty((0,), dtype=np.float32)
    return torch.stack(items, dim=0).cpu().numpy()


@dataclass
class RolloutStats:
    episode_rewards: list[float]
    episode_lengths: list[int]
    success_rates: list[float]
    per_env_ever_lift_success: list[list[bool]]


class TrajectoryRecorder:
    """Standalone trajectory recorder with optional MP4 generation."""

    def __init__(
        self,
        save_dir: str,
        num_envs: int,
        stereo: bool,
        max_records_per_file: int,
        create_video: bool,
        obs_key: str,
        lift_height_thresh: float | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.stereo = stereo
        self.max_records_per_file = max(1, int(max_records_per_file))
        self.create_video = create_video
        self.obs_key = obs_key
        self.lift_height_thresh = lift_height_thresh
        self.file_counter = 0
        self.recording_step_counter = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(save_dir, f"student_replay_{timestamp}")
        self.data_dir = os.path.join(self.run_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.video_dir = ""
        if self.create_video:
            self.video_dir = os.path.join(self.run_dir, "videos")
            os.makedirs(self.video_dir, exist_ok=True)

        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self.buffer: dict[str, list[torch.Tensor]] = {
            "obs": [],
            "img_left": [],
            "img_right": [],
            "action": [],
            "prev_action": [],
            "reward": [],
            "done": [],
            "object_pos": [],
        }

    def record_step(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        prev_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        obs_tensor = obs[self.obs_key] if self.obs_key in obs else obs["policy"]
        self.buffer["obs"].append(obs_tensor.detach().cpu())
        if "img_left" in obs:
            self.buffer["img_left"].append(obs["img_left"].detach().cpu())
        if self.stereo and "img_right" in obs:
            self.buffer["img_right"].append(obs["img_right"].detach().cpu())
        if "aux_info" in obs and "object_pos" in obs["aux_info"]:
            self.buffer["object_pos"].append(obs["aux_info"]["object_pos"].detach().cpu())
        self.buffer["action"].append(action.detach().cpu())
        self.buffer["prev_action"].append(prev_action.detach().cpu())
        self.buffer["reward"].append(reward.detach().cpu())
        self.buffer["done"].append(done.detach().cpu())

        self.recording_step_counter += 1
        if self.recording_step_counter >= self.max_records_per_file:
            self.flush()

    def flush(self) -> None:
        if self.recording_step_counter == 0:
            return

        payload = {
            k: _to_numpy_stacked(v)
            for k, v in self.buffer.items()
            if len(v) > 0
        }
        timestamp = str(datetime.now())
        chunk_steps = int(self.recording_step_counter)

        for env_id in range(self.num_envs):
            env_payload: dict[str, np.ndarray] = {}
            for key, value in payload.items():
                # Most datasets are [T, B, ...]; split by env along B axis.
                if value.ndim >= 2 and value.shape[0] == chunk_steps and value.shape[1] == self.num_envs:
                    env_payload[key] = value[:, env_id]
                else:
                    env_payload[key] = value

            ever_lift_success = False
            if self.lift_height_thresh is not None and "object_pos" in env_payload:
                obj = env_payload["object_pos"]
                if obj.ndim >= 2 and obj.shape[-1] >= 3:
                    ever_lift_success = bool(np.any(obj[..., 2] > self.lift_height_thresh))

            base = os.path.join(self.data_dir, f"traj_env_{env_id}_file_{self.file_counter}")
            h5_path = f"{base}.h5"
            yaml_path = f"{base}.yaml"
            meta = {
                "timestamp": timestamp,
                "env_id": int(env_id),
                "source_num_envs": self.num_envs,
                "steps": chunk_steps,
                "stereo": bool(self.stereo),
                "lift_height_thresh": float(self.lift_height_thresh) if self.lift_height_thresh is not None else None,
                "ever_lift_success": bool(ever_lift_success),
                "datasets": {k: list(v.shape) for k, v in env_payload.items()},
            }

            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(meta, f, default_flow_style=False)

            with h5py.File(h5_path, "w") as f:
                for k, v in env_payload.items():
                    f.create_dataset(k, data=v, compression="gzip", compression_opts=3)
                # Save scalar success flag in file content for fast filtering.
                f.create_dataset("ever_lift_success", data=np.array(bool(ever_lift_success), dtype=np.uint8))
                for k, v in meta.items():
                    if k != "datasets":
                        f.attrs[k] = v

        if self.create_video and "img_left" in payload:
            self._create_videos_from_chunk(payload["img_left"], self.file_counter)

        self.file_counter += 1
        self.recording_step_counter = 0
        self._reset_buffer()

    def _create_videos_from_chunk(self, images: np.ndarray, file_index: int) -> None:
        # images shape: [T, B, C, H, W]
        if images.ndim != 5:
            return
        try:
            import cv2
        except ImportError:
            print("[WARN] OpenCV not available. Skipping video generation.")
            return

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg is required for H.264 video export but was not found in PATH.")

        num_steps, num_envs, _, height, width = images.shape
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for env_id in range(min(self.num_envs, num_envs)):
            tmp_path = os.path.join(self.video_dir, f"env_{env_id}_file_{file_index}_tmp.mp4")
            out_path = os.path.join(self.video_dir, f"env_{env_id}_file_{file_index}_h264.mp4")

            writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open temporary mp4v writer for env {env_id}: {tmp_path}")

            for step in range(num_steps):
                frame = images[step, env_id]  # [C, H, W]
                frame = np.transpose(frame, (1, 2, 0))
                frame = np.clip(frame, 0.0, 1.0)
                frame_u8 = (frame * 255.0).astype(np.uint8)
                writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))
            writer.release()

            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    "ffmpeg H.264 transcode failed for "
                    f"{tmp_path} -> {out_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            os.remove(tmp_path)


class StudentPolicyReplayer:
    def __init__(
        self,
        env: gym.Env,
        student_cfg_path: str,
        checkpoint_path: str,
        obs_key: str,
        deterministic: bool,
    ) -> None:
        self.env = env
        self.ov_env = env.env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.obs_key = obs_key
        self.deterministic = deterministic

        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions

        network_params = load_yaml(student_cfg_path)["params"]
        self.normalize_input = network_params["config"].get("normalize_input", True)
        self.model_cfg = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": 1,
            "normalize_value": network_params["config"].get("normalize_value", True),
            "normalize_input": self.normalize_input,
        }

        register_stereo_transformer_builder()
        builder = ModelBuilder()
        network = builder.load(network_params)
        self.model = network.build(self.model_cfg).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self.is_aux = bool(getattr(self.model.a2c_network, "is_aux", False))
        self.is_rnn = self.model.is_rnn()
        self.seq_length = 1

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.hidden_states: tuple[torch.Tensor, ...] | None = None
        self.reset_policy_state()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        weights = torch.load(checkpoint_path, map_location=self.device)
        if "model" in weights:
            remapped = adjust_state_dict_keys(weights["model"], self.model.state_dict())
            self.model.load_state_dict(remapped)
        else:
            self.model.load_state_dict(weights)

        if self.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def reset_policy_state(self) -> None:
        self.prev_actions.zero_()
        if self.is_rnn:
            self.hidden_states = tuple(s.to(self.device) for s in self.model.get_default_rnn_state())
        else:
            self.hidden_states = None

    @torch.no_grad()
    def get_actions(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        prev_action_snapshot = self.prev_actions.clone()
        batch = {
            "is_train": False,
            "is_deterministic": self.deterministic,
            "obs": obs[self.obs_key],
            "prev_actions": self.prev_actions,
            "finetune_backbone": False,
        }

        for key in ("img", "rgb", "rgb_data", "img_left", "img_right"):
            if key in obs:
                batch[key] = obs[key]
        if "rgb" in obs and "rgb_data" not in batch:
            batch["rgb_data"] = obs["rgb"]

        if self.is_rnn and self.hidden_states is not None:
            batch["rnn_states"] = self.hidden_states
            batch["seq_length"] = self.seq_length
            batch["rnn_masks"] = None

        out = self.model(batch)
        mus = out["mus"]
        sigmas = out.get("sigmas", torch.zeros_like(mus))
        if self.deterministic:
            action = mus
        else:
            # Keep stochastic sampling robust even if checkpoint/network emits bad sigma signs.
            safe_sigmas = sigmas.abs().clamp(min=1e-6)
            action = torch.distributions.Normal(mus, safe_sigmas, validate_args=False).sample()
        action = torch.clamp(action, -1.0, 1.0)

        if self.is_rnn and self.hidden_states is not None:
            if self.is_aux:
                self.hidden_states = tuple(s.detach() for s in out["rnn_states"][0])
            else:
                self.hidden_states = tuple(s.detach() for s in out["rnn_states"])

        self.prev_actions = action.detach()
        return action.detach(), prev_action_snapshot

    def reset_rnn_for_done_envs(self, done_indices: torch.Tensor) -> None:
        if not self.is_rnn or self.hidden_states is None or done_indices.numel() == 0:
            return
        hs = list(self.hidden_states)
        for state in hs:
            state[:, done_indices] = 0.0
        self.hidden_states = tuple(hs)


def _compute_lift_success_flags(ov_env: Any) -> torch.Tensor:
    table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0)
    lift_success = ov_env.object_pos[:, 2] > lift_height_thresh
    return lift_success


def _run_rollouts(
    env: gym.Env,
    replayer: StudentPolicyReplayer,
    num_episodes: int,
    max_steps_per_episode: int,
    recorder: TrajectoryRecorder | None,
) -> RolloutStats:
    stats = RolloutStats(
        episode_rewards=[],
        episode_lengths=[],
        success_rates=[],
        per_env_ever_lift_success=[],
    )

    for ep in range(num_episodes):
        obs = env.reset()[0]
        replayer.reset_policy_state()
        dones = torch.zeros((replayer.num_envs,), dtype=torch.bool, device=replayer.device)
        ever_lift_success = torch.zeros((replayer.num_envs,), dtype=torch.bool, device=replayer.device)
        ep_reward = 0.0
        ep_length = 0

        while not torch.all(dones):
            if max_steps_per_episode > 0 and ep_length >= max_steps_per_episode:
                break

            action, prev_action = replayer.get_actions(obs)
            obs, reward, out_of_reach, timed_out, _ = env.step(action)
            dones = out_of_reach | timed_out

            if recorder is not None:
                recorder.record_step(obs, action, prev_action, reward, dones)

            ep_reward += float(reward.mean().item())
            ep_length += 1
            ever_lift_success |= _compute_lift_success_flags(replayer.ov_env)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            replayer.reset_rnn_for_done_envs(done_indices)

        if recorder is not None:
            recorder.flush()

        success_rate = float(ever_lift_success.float().mean().item())
        per_env_flags = [bool(v) for v in ever_lift_success.detach().cpu().tolist()]
        stats.episode_rewards.append(ep_reward)
        stats.episode_lengths.append(ep_length)
        stats.success_rates.append(success_rate)
        stats.per_env_ever_lift_success.append(per_env_flags)

        print(
            f"[Episode {ep + 1}/{num_episodes}] reward={ep_reward:.2f} "
            f"length={ep_length} ever_lift_success_per_env={per_env_flags}"
        )

    return stats


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed
    env_cfg.simulate_stereo = True

    if args_cli.record_data:
        env_cfg.disable_out_of_reach_done = True
        env_cfg.disable_arm_randomization = True
        env_cfg.disable_dome_light_randomization = False # default True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if not env.env.simulate_stereo:
        raise RuntimeError("This recorder expects a stereo student policy but simulate_stereo is disabled.")

    if args_cli.student_cfg is None:
        root = _find_repo_root(pathlib.Path(__file__).resolve())
        student_cfg = root / "dextrah_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_stereo_transformer.yaml"
    else:
        student_cfg = pathlib.Path(args_cli.student_cfg).expanduser().resolve()

    os.makedirs(args_cli.save_dir, exist_ok=True)
    replayer = StudentPolicyReplayer(
        env=env,
        student_cfg_path=str(student_cfg),
        checkpoint_path=args_cli.checkpoint,
        obs_key=args_cli.obs_key,
        deterministic=args_cli.deterministic,
    )
    recorder = None
    if args_cli.record_data:
        table_center_z = env.env.cfg.table_cfg.init_state.pos[2]
        table_top_z = table_center_z + 0.5 * env.env.cfg.table_size_z
        lift_height_thresh = table_top_z + getattr(env.env.cfg, "object_height_thresh", 0.0)
        recorder = TrajectoryRecorder(
            save_dir=args_cli.save_dir,
            num_envs=replayer.num_envs,
            stereo=True,
            max_records_per_file=args_cli.max_records_per_file,
            create_video=args_cli.create_video,
            obs_key=args_cli.obs_key,
            lift_height_thresh=float(lift_height_thresh),
        )

    stats = _run_rollouts(
        env=env,
        replayer=replayer,
        num_episodes=args_cli.num_episodes,
        max_steps_per_episode=args_cli.max_steps_per_episode,
        recorder=recorder,
    )

    results = {
        "episode_rewards": stats.episode_rewards,
        "episode_lengths": stats.episode_lengths,
        "success_rates": stats.success_rates,
        "per_env_ever_lift_success": stats.per_env_ever_lift_success,
        "mean_reward": float(np.mean(stats.episode_rewards)),
        "std_reward": float(np.std(stats.episode_rewards)),
        "mean_length": float(np.mean(stats.episode_lengths)),
        "std_length": float(np.std(stats.episode_lengths)),
        "mean_success": float(np.mean(stats.success_rates)),
        "std_success": float(np.std(stats.success_rates)),
        "checkpoint": args_cli.checkpoint,
        "student_cfg": str(student_cfg),
        "task": args_cli.task,
        "deterministic": bool(args_cli.deterministic),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args_cli.save_dir, f"student_replay_results_{stamp}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    print(f"Saved replay summary: {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
