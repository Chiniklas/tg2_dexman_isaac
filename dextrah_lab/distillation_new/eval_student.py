"""Evaluate a student checkpoint with SafeDagger runtime eval semantics.

This script mirrors the standalone student evaluation logic used in
`distillation_safedagger.py::evaluate_student`:
- lift success with hold-time gating
- unsafe episode rate
- unsafe reason proportions
- per-object metrics (lift/unsafe/unsafe-reason proportions)
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Evaluate a student policy checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the student policy checkpoint.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of eval episodes.")
parser.add_argument("--eval_max_steps", type=int, default=None, help="Max steps per episode (default: env limit).")
parser.add_argument(
    "--eval_lift_hold_s",
    type=float,
    default=None,
    help="Lift hold gate duration in seconds. If unset, uses distillation YAML default (fallback 0.5).",
)
parser.add_argument(
    "--objects_dir",
    type=str,
    default=None,
    help="Optional object assets directory name under dextrah_lab/assets (e.g. unseen_objects).",
)
parser.add_argument("--save_dir", type=str, default="eval_results", help="Directory to save evaluation results.")
parser.add_argument(
    "--metrics_output_json",
    "--metrics_output_npy",
    dest="metrics_output_json",
    type=str,
    default=None,
    help=(
        "Optional output path for metrics JSON. "
        "Legacy alias: --metrics_output_npy. "
        "Timestamp is always appended."
    ),
)
parser.add_argument(
    "--file_name_head",
    type=str,
    default=None,
    help="Optional output JSON filename prefix override.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra argv and app launch
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import dextrah_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401
from dextrah_lab.distillation_new.a2c_stereo_transformer import A2CBuilder as A2CStereoTransformerBuilder
from dextrah_lab.distillation_new.eval_utils import (
    UNSAFE_REASON_NAMES,
    classify_out_of_reach_reasons,
    unsafe_reason_percentages_from_counts,
)

_PROGRESS_TIME_INTERVAL_S = 20.0


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        key_with_orig_mod = ".".join(parts)
        if key_with_orig_mod in model_state_dict:
            adjusted_state_dict[key_with_orig_mod] = value
            continue

        key_no_orig_mod = key.replace("_orig_mod.", "")
        if key_no_orig_mod in model_state_dict:
            adjusted_state_dict[key_no_orig_mod] = value
            continue

        adjusted_state_dict[key] = value
    return adjusted_state_dict


def _reason_counts_checked(
    reason_idx: torch.Tensor,
    unsafe_mask: torch.Tensor,
    reason_names,
    reason_to_idx: dict[str, int],
    warn_label: str | None = None,
) -> tuple[dict[str, int], int]:
    unsafe_mask = unsafe_mask.to(dtype=torch.bool)
    counts = {
        str(name): int(((reason_idx == int(reason_to_idx[str(name)])) & unsafe_mask).sum().item())
        for name in reason_names
    }
    total_unsafe = int(unsafe_mask.sum().item())
    classified_total = int(sum(counts.values()))
    unknown_count = max(0, total_unsafe - classified_total)
    if unknown_count > 0:
        label = warn_label if warn_label is not None else "unsafe reason classification"
        raise RuntimeError(
            f"{label}: found {unknown_count} unclassified unsafe episodes "
            f"(unsafe_total={total_unsafe}, classified_total={classified_total}). "
            "Fail-fast mode is enabled; no fallback mapping is allowed."
        )
    return counts, total_unsafe


def _format_reason_percentages(reason_percentages: dict[str, float]) -> str:
    if not reason_percentages:
        return "none"
    ordered = sorted(reason_percentages.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join([f"{name}={value:.1f}%" for name, value in ordered])


def _reason_percentages_with_defaults(reason_percentages: dict[str, float]) -> dict[str, float]:
    return {
        name: float(reason_percentages.get(name, 0.0))
        for name in UNSAFE_REASON_NAMES
    }


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _save_metrics_json(metrics_payload: dict) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_head = None
    if isinstance(args_cli.file_name_head, str):
        candidate = args_cli.file_name_head.strip()
        if len(candidate) > 0:
            custom_head = candidate
    if args_cli.metrics_output_json is not None:
        raw_output_path = pathlib.Path(args_cli.metrics_output_json).expanduser()
        if not raw_output_path.is_absolute():
            raw_output_path = pathlib.Path(args_cli.save_dir) / raw_output_path
        raw_output_path = raw_output_path.resolve()
        stem = custom_head if custom_head is not None else raw_output_path.stem
        output_path = raw_output_path.with_name(f"{stem}_{timestamp}.json")
    else:
        if custom_head is not None:
            default_name = f"{custom_head}_{timestamp}.json"
        else:
            default_name = f"eval_metrics_{timestamp}.json"
        output_path = pathlib.Path(args_cli.save_dir).expanduser().resolve() / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics_payload), f, indent=2, sort_keys=True)
    print(f"[INFO] Saved evaluation metrics JSON to: {output_path}")
    return output_path


class StudentEvaluator:
    def __init__(self, env, config, checkpoint_path, eval_max_steps=None, eval_lift_hold_s=0.5):
        self.env = env
        self.eval_env = None
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.num_actions_student = self.num_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.eval_max_steps = eval_max_steps
        self.eval_lift_hold_s = max(0.0, float(eval_lift_hold_s or 0.0))
        self.student_obs_type = self.config["student"]["obs_type"]
        self.metric_object_names = list(getattr(self.ov_env, "object_names", []))

        self.unsafe_reason_names = UNSAFE_REASON_NAMES
        self.unsafe_reason_to_idx = {name: idx for idx, name in enumerate(self.unsafe_reason_names)}

        self.network_params = self.load_param_dict(self.config["student"]["cfg"])["params"]
        self.network = self.load_networks(self.network_params)
        self.value_size = 1
        self.normalize_value = self.network_params["config"]["normalize_value"]
        self.normalize_input = self.network_params["config"]["normalize_input"]

        self.model_config = {
            "actions_num": self.num_actions_student,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        self.student_model = self.network.build(self.model_config).to(self.device)
        self.load_checkpoint(checkpoint_path)

        self.is_rnn = self.student_model.is_rnn()
        self.is_aux = bool(
            hasattr(self.student_model, "a2c_network")
            and hasattr(self.student_model.a2c_network, "is_aux")
            and self.student_model.a2c_network.is_aux
        )

    def load_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path, map_location=self.device)
        if "model" in weights:
            weights["model"] = adjust_state_dict_keys(weights["model"], self.student_model.state_dict())
            self.student_model.load_state_dict(weights["model"])
        else:
            self.student_model.load_state_dict(weights)
        if self.normalize_input and "running_mean_std" in weights:
            self.student_model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def load_networks(self, params):
        builder = ModelBuilder()
        return builder.load(params)

    def load_param_dict(self, cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _zero_rnn_states(states, indices):
        if indices.numel() == 0:
            return
        indices = indices.flatten()
        for s in states:
            if s.dim() == 2:
                s[indices] = 0
            else:
                s[:, indices, ...] = 0

    def _get_student_actions_eval(self, obs, prev_actions, hidden_states):
        batch_dict = {
            "is_train": False,
            "obs": obs[self.student_obs_type],
            "prev_actions": prev_actions,
        }
        if "img" in obs:
            batch_dict["img"] = obs["img"]
            batch_dict["rgb_data"] = obs["rgb"]
            batch_dict["rgb"] = obs["rgb"]
        if "img_left" in obs:
            batch_dict["img_left"] = obs["img_left"]
            batch_dict["img_right"] = obs["img_right"]
        if self.is_rnn:
            batch_dict["rnn_states"] = hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None
        batch_dict["finetune_backbone"] = False

        res_dict = self.student_model(batch_dict)
        mus = res_dict["mus"]
        sigmas = res_dict["sigmas"]
        if self.is_rnn:
            if self.is_aux:
                hidden_states = [s for s in res_dict["rnn_states"][0]]
            else:
                hidden_states = [s for s in res_dict["rnn_states"]]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()
        selected_action = torch.clamp(selected_action, -1.0, 1.0)
        return selected_action.detach(), hidden_states

    def evaluate_student(self, num_episodes):
        if num_episodes <= 0:
            return None, None, None, None, {}
        eval_start_t = time.time()

        eval_env = self.eval_env if self.eval_env is not None else self.env
        eval_ov_env = eval_env.env
        if eval_ov_env.num_envs != self.num_envs:
            print(
                "Skipping eval: eval_env num_envs must match training num_envs "
                f"({eval_ov_env.num_envs} vs {self.num_envs})."
            )
            return None, None, None, None, {}

        num_envs = eval_ov_env.num_envs
        eval_object_names = list(getattr(eval_ov_env, "object_names", []))
        if len(eval_object_names) == 0:
            eval_object_names = list(self.metric_object_names)
        if len(eval_object_names) == 0:
            eval_object_names = ["object_0"]
        eval_object_names = [str(name) for name in eval_object_names]

        eval_object_idx = getattr(eval_ov_env, "multi_object_idx", None)
        if eval_object_idx is None:
            eval_object_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        else:
            eval_object_idx = torch.as_tensor(eval_object_idx, dtype=torch.long, device=self.device).flatten()
            if eval_object_idx.numel() < num_envs:
                padded = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
                padded[: eval_object_idx.numel()] = eval_object_idx
                eval_object_idx = padded
            elif eval_object_idx.numel() > num_envs:
                eval_object_idx = eval_object_idx[:num_envs]
        eval_object_idx = torch.clamp(eval_object_idx, min=0, max=len(eval_object_names) - 1)
        eval_object_env_counts = {
            object_name: int((eval_object_idx == idx).sum().item())
            for idx, object_name in enumerate(eval_object_names)
        }

        sim_dt = getattr(eval_ov_env.cfg, "sim_dt", None)
        if sim_dt is None and hasattr(eval_ov_env.cfg, "sim"):
            sim_dt = getattr(eval_ov_env.cfg.sim, "dt", None)
        decimation = getattr(eval_ov_env.cfg, "decimation", 1)
        step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
        hold_steps = 1
        if self.eval_lift_hold_s > 0.0 and step_dt > 0.0:
            hold_steps = max(1, int(math.ceil(self.eval_lift_hold_s / step_dt)))
        print(
            f"[INFO] Eval lift hold gate: {hold_steps} steps "
            f"(~{self.eval_lift_hold_s:.3f}s target, dt={step_dt:.5f}s)",
            flush=True,
        )

        prev_actions = torch.zeros(
            (num_envs, self.num_actions_student), dtype=torch.float32, device=self.device
        )

        was_training = self.student_model.training
        self.student_model.eval()

        max_steps = self.eval_max_steps
        if max_steps is None:
            max_steps = getattr(eval_ov_env, "distill_max_episode_length", None)
        if max_steps is None:
            max_steps = getattr(eval_ov_env, "max_episode_length", None)
        if max_steps is None:
            max_steps = 1000

        success_rates = []
        reward_means = []
        unsafe_rates = []
        total_unsafe_eps = 0
        total_reason_counts = {name: 0 for name in self.unsafe_reason_names}
        per_object_lift_series = {object_name: [] for object_name in eval_object_names}
        per_object_unsafe_rate_series = {object_name: [] for object_name in eval_object_names}
        per_object_reason_counts_total = {
            object_name: {name: 0 for name in self.unsafe_reason_names}
            for object_name in eval_object_names
        }
        per_object_unsafe_eps_total = {object_name: 0 for object_name in eval_object_names}
        total_rollouts = int(num_episodes)
        progress_rollout_interval = max(1, min(50, total_rollouts // 10 if total_rollouts > 10 else 1))
        progress_time_interval_s = _PROGRESS_TIME_INTERVAL_S
        next_rollout_progress = progress_rollout_interval
        loop_start_t = time.time()
        last_progress_t = loop_start_t
        print(
            f"[INFO] Rollout loop started: target_rollouts={total_rollouts}, "
            f"rollout_interval={progress_rollout_interval}, "
            f"time_interval_s={progress_time_interval_s:.1f}",
            flush=True,
        )

        with torch.no_grad():
            for rollout_idx in range(total_rollouts):
                obs = eval_env.reset()[0]
                dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                ever_unsafe_terminated = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
                unsafe_reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
                if self.is_rnn:
                    hidden_states = [s.to(self.device) for s in self.student_model.get_default_rnn_state()]
                else:
                    hidden_states = None
                prev_actions.zero_()
                steps = 0
                reward_sum = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
                lift_hold_counts = torch.zeros((num_envs,), device=self.device, dtype=torch.long)

                while steps < max_steps and not dones.all():
                    actions, hidden_states = self._get_student_actions_eval(obs, prev_actions, hidden_states)
                    obs, reward, out_of_reach, timed_out, info = eval_env.step(actions)
                    dones = out_of_reach | timed_out
                    reason_idx = classify_out_of_reach_reasons(
                        ov_env=eval_ov_env,
                        out_of_reach=out_of_reach,
                        reason_names=self.unsafe_reason_names,
                        reason_to_idx=self.unsafe_reason_to_idx,
                        device=self.device,
                    )
                    ever_unsafe_terminated = ever_unsafe_terminated | out_of_reach
                    classified_out_of_reach = reason_idx >= 0
                    new_reason_mask = (unsafe_reason_idx < 0) & out_of_reach & classified_out_of_reach
                    unsafe_reason_idx[new_reason_mask] = reason_idx[new_reason_mask]
                    prev_actions = actions.detach()
                    if self.is_rnn and dones.any():
                        self._zero_rnn_states(hidden_states, dones.nonzero(as_tuple=False))
                    reward_sum += reward
                    steps += 1

                    table_center_z = eval_ov_env.cfg.table_cfg.init_state.pos[2]
                    table_top_z = table_center_z + 0.5 * eval_ov_env.cfg.table_size_z
                    lift_height_thresh = table_top_z + getattr(eval_ov_env.cfg, "object_height_thresh", 0.0)
                    lift_success = eval_ov_env.object_pos[:, 2] > lift_height_thresh
                    if hasattr(eval_ov_env, "good_grasp_mask") and eval_ov_env.good_grasp_mask is not None:
                        contact_mask = eval_ov_env.good_grasp_mask.to(device=self.device, dtype=torch.bool)
                    elif hasattr(eval_ov_env, "object_contact_counts") and eval_ov_env.object_contact_counts is not None:
                        contact_mask = eval_ov_env.object_contact_counts.to(device=self.device) > 0.0
                    else:
                        contact_mask = torch.ones_like(lift_success, dtype=torch.bool)
                    lift_success = lift_success & contact_mask
                    active_envs = ~dones
                    lift_hold_counts = torch.where(
                        active_envs & lift_success,
                        lift_hold_counts + 1,
                        torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
                    )
                    lift_success = lift_hold_counts >= hold_steps
                    ever_lifted = ever_lifted | lift_success

                # If max-steps cap is hit, mark remaining envs done for aggregation semantics.
                if steps >= max_steps:
                    dones = torch.ones_like(dones)

                success_rates.append(ever_lifted.float().mean().item())
                reward_means.append(reward_sum.mean().item())
                unsafe_rates.append(ever_unsafe_terminated.float().mean().item())
                total_unsafe_eps += int(ever_unsafe_terminated.sum().item())

                reason_counts, _ = _reason_counts_checked(
                    reason_idx=unsafe_reason_idx,
                    unsafe_mask=ever_unsafe_terminated,
                    reason_names=self.unsafe_reason_names,
                    reason_to_idx=self.unsafe_reason_to_idx,
                    warn_label=None,
                )
                for name in self.unsafe_reason_names:
                    total_reason_counts[name] += int(reason_counts[name])

                for obj_idx, object_name in enumerate(eval_object_names):
                    obj_mask = eval_object_idx == obj_idx
                    if not obj_mask.any():
                        continue
                    per_object_lift_series[object_name].append(ever_lifted[obj_mask].float().mean().item())
                    per_object_unsafe_rate_series[object_name].append(
                        ever_unsafe_terminated[obj_mask].float().mean().item()
                    )
                    per_object_unsafe_eps_total[object_name] += int(ever_unsafe_terminated[obj_mask].sum().item())
                    obj_reason_counts, _ = _reason_counts_checked(
                        reason_idx=unsafe_reason_idx[obj_mask],
                        unsafe_mask=ever_unsafe_terminated[obj_mask],
                        reason_names=self.unsafe_reason_names,
                        reason_to_idx=self.unsafe_reason_to_idx,
                        warn_label=None,
                    )
                    for reason_name in self.unsafe_reason_names:
                        per_object_reason_counts_total[object_name][reason_name] += int(
                            obj_reason_counts[reason_name]
                        )

                now_t = time.time()
                rollout_done = rollout_idx + 1
                rollout_trigger = rollout_done >= next_rollout_progress
                time_trigger = (now_t - last_progress_t) >= progress_time_interval_s
                if rollout_trigger or time_trigger:
                    completion = 100.0 * float(rollout_done) / float(max(total_rollouts, 1))
                    avg_success_so_far = float(np.mean(success_rates)) if len(success_rates) > 0 else 0.0
                    avg_unsafe_so_far = float(np.mean(unsafe_rates)) if len(unsafe_rates) > 0 else 0.0
                    print(
                        f"[INFO] Eval progress: {rollout_done}/{total_rollouts} rollouts ({completion:.1f}%) "
                        f"| lift_success={avg_success_so_far:.4f} | unsafe_rate={avg_unsafe_so_far:.4f} "
                        f"| elapsed={now_t - loop_start_t:.1f}s",
                        flush=True,
                    )
                    if rollout_trigger:
                        while (
                            next_rollout_progress <= rollout_done
                            and next_rollout_progress < total_rollouts
                        ):
                            next_rollout_progress += progress_rollout_interval
                    last_progress_t = now_t

        if was_training:
            self.student_model.train()

        total_eval_episodes = max(1, int(num_episodes * num_envs))
        eval_reason_prop = {
            name: float(total_reason_counts.get(name, 0)) / float(total_eval_episodes)
            for name in self.unsafe_reason_names
        }
        eval_reason_pct = unsafe_reason_percentages_from_counts(
            total_reason_counts, total_unsafe_eps, self.unsafe_reason_names
        )

        eval_per_object_metrics = {}
        for object_name in eval_object_names:
            total_obj_episodes = max(1, int(num_episodes * eval_object_env_counts.get(object_name, 0)))
            obj_reason_prop = {
                name: float(per_object_reason_counts_total[object_name].get(name, 0)) / float(total_obj_episodes)
                for name in self.unsafe_reason_names
            }
            obj_reason_pct = unsafe_reason_percentages_from_counts(
                per_object_reason_counts_total[object_name],
                per_object_unsafe_eps_total[object_name],
                self.unsafe_reason_names,
            )
            eval_per_object_metrics[object_name] = {
                "lift_success": (
                    float(np.mean(per_object_lift_series[object_name]))
                    if len(per_object_lift_series[object_name]) > 0
                    else 0.0
                ),
                "unsafe_episode_rate": (
                    float(np.mean(per_object_unsafe_rate_series[object_name]))
                    if len(per_object_unsafe_rate_series[object_name]) > 0
                    else 0.0
                ),
                "unsafe_reason_prop": {
                    name: float(obj_reason_prop.get(name, 0.0)) for name in self.unsafe_reason_names
                },
                "unsafe_reason_pct": {
                    name: float(obj_reason_pct.get(name, 0.0)) for name in self.unsafe_reason_names
                },
            }

        lift_success_out = float(np.mean(success_rates))
        unsafe_rate_out = float(np.mean(unsafe_rates))
        print(
            f"[INFO] Eval complete in {time.time() - eval_start_t:.1f}s: "
            f"episodes={total_eval_episodes}, lift_success={lift_success_out:.4f}, unsafe_rate={unsafe_rate_out:.4f}",
            flush=True,
        )
        return (
            lift_success_out,
            float(np.mean(reward_means)),
            unsafe_rate_out,
            eval_reason_prop,
            eval_reason_pct,
            eval_per_object_metrics,
            total_eval_episodes,
            num_envs,
            total_unsafe_eps,
        )


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env = None
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.objects_dir:
        env_cfg.objects_dir = args_cli.objects_dir
        if (
            hasattr(env_cfg, "valid_objects_dir")
            and isinstance(env_cfg.valid_objects_dir, list)
            and args_cli.objects_dir not in env_cfg.valid_objects_dir
        ):
            env_cfg.valid_objects_dir.append(args_cli.objects_dir)

    try:
        print(
            f"[INFO] Eval start: task={args_cli.task}, objects_dir={args_cli.objects_dir}, "
            f"checkpoint={args_cli.checkpoint}",
            flush=True,
        )
        stage_t = time.time()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        print(
            f"[INFO] Environment ready in {time.time() - stage_t:.1f}s (num_envs={env.env.num_envs}).",
            flush=True,
        )

        parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
        agent_cfg_folder = "dextrah_lab/tasks/tg2_inspirehand/agents"
        if not env.env.simulate_stereo:
            raise ValueError("eval_student only supports stereo transformer policies for distillation_new.")
        student_cfg = os.path.join(parent_path, agent_cfg_folder, "rl_games_ppo_stereo_transformer.yaml")
        with open(student_cfg, "r", encoding="utf-8") as f:
            student_cfg_yaml = yaml.safe_load(f) or {}
        distill_cfg = (
            student_cfg_yaml.get("params", {}).get("distillation", {})
            if isinstance(student_cfg_yaml, dict)
            else {}
        )
        eval_lift_hold_s = (
            float(args_cli.eval_lift_hold_s)
            if args_cli.eval_lift_hold_s is not None
            else float(distill_cfg.get("eval_lift_hold_s", 0.5))
        )
        eval_max_steps = args_cli.eval_max_steps
        print(
            f"[INFO] Eval knobs: num_episodes={args_cli.num_episodes}, "
            f"eval_max_steps={eval_max_steps}, eval_lift_hold_s={eval_lift_hold_s}",
            flush=True,
        )

        model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

        config = {
            "student": {
                "cfg": student_cfg,
                "obs_type": "policy",
            }
        }
        evaluator = StudentEvaluator(
            env=env,
            config=config,
            checkpoint_path=args_cli.checkpoint,
            eval_max_steps=eval_max_steps,
            eval_lift_hold_s=eval_lift_hold_s,
        )

        (
            lift_success,
            avg_reward,
            unsafe_rate,
            reason_prop,
            reason_pct,
            per_object_metrics,
            total_episodes,
            num_envs,
            unsafe_episodes,
        ) = evaluator.evaluate_student(args_cli.num_episodes)
        reason_percentages_full = _reason_percentages_with_defaults(reason_pct)

        print(
            f"eval/lift_success: {lift_success:.4f} | "
            f"eval/unsafe_episode_rate: {unsafe_rate:.4f} | "
            f"eval/out_of_reach_reason_pct: {_format_reason_percentages(reason_percentages_full)} "
            f"(total episodes: {total_episodes}, envs: {num_envs})"
        )

        metrics_payload = {
            "mode": "single_checkpoint",
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "eval_episodes": int(args_cli.num_episodes),
            "metrics": {
                "eval/lift_success": float(lift_success),
                "eval/unsafe_episode_rate": float(unsafe_rate),
                "eval/out_of_reach_reason_pct": reason_percentages_full,
                "eval/avg_reward": float(avg_reward),
                "eval/unsafe_reason_prop": {k: float(v) for k, v in reason_prop.items()},
            },
            "per_object_metrics": per_object_metrics,
            "total_episodes": int(total_episodes),
            "num_envs": int(num_envs),
            "unsafe_episodes": int(unsafe_episodes),
            "eval_max_steps": int(eval_max_steps) if eval_max_steps is not None else None,
            "eval_lift_hold_s": float(eval_lift_hold_s),
        }
        _save_metrics_json(metrics_payload)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
