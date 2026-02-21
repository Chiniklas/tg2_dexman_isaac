"""Offline student checkpoint evaluation over a distillation run folder.

Instructions:
    1) Provide `--policy_run_dir` that contains `nn/*.pth` checkpoints.
    2) Provide `--object_dir` that contains one subfolder per object.
    3) The script evaluates all `*_iters.pth` checkpoints except the final
       (highest-iteration) checkpoint.
    4) For each checkpoint/object, it reports:
       - `eval/lift_success/<object_name>`
       - `eval/unsafe_episode_rate/<object_name>`
       - `eval/out_of_reach_reason_pct/<object_name>`
    5) It also reports checkpoint-level averages:
       - `eval/lift_success_avg`
       - `eval/unsafe_episode_rate_avg`
       - `eval/out_of_reach_reason_pct_avg`
    6) Results are saved to a timestamped `.npy` file.

Example usage:
    python dextrah_lab/distillation_new/eval_student_offline.py \
        --task Dextrah-TG2-InspireHand-Direct-v0 \
        --policy_run_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/runs/dextrah-tg2-inspirehand_13-00-43-46 \
        --object_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/teacher_eval \
        --eval_episodes 10 \
        --deterministic
"""

import argparse
import copy
import os
import pathlib
import re
import shutil
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Offline evaluation for all intermediate student checkpoints in a run.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument(
    "--policy_run_dir",
    type=str,
    required=True,
    help="Distillation run folder containing nn/ checkpoints (e.g., runs/dextrah-tg2-inspirehand_13-00-43-46).",
)
parser.add_argument(
    "--object_dir",
    type=str,
    required=True,
    help="Object root directory containing one subfolder per object.",
)
parser.add_argument(
    "--student_cfg",
    type=str,
    default=None,
    help=(
        "Optional student network cfg YAML. "
        "Default: dextrah_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_stereo_transformer.yaml"
    ),
)
parser.add_argument("--eval_episodes", type=int, default=10, help="Number of eval episodes.")
parser.add_argument(
    "--eval_max_steps",
    type=int,
    default=None,
    help="Max steps per episode (defaults to env max if not provided).",
)
parser.add_argument(
    "--eval_lift_hold_s",
    type=float,
    default=0.5,
    help="Lift hold time gate in seconds.",
)
parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (policy mean).")
parser.add_argument(
    "--metrics_output_npy",
    type=str,
    default=None,
    help=(
        "Optional output path for metrics .npy. "
        "Timestamp is always appended to the filename."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import dextrah_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401
from dextrah_lab.distillation_new.a2c_stereo_transformer import (
    A2CBuilder as A2CStereoTransformerBuilder,
)
from eval_utils import (
    UNSAFE_REASON_NAMES,
    as_bool_mask,
    classify_out_of_reach_reasons,
    unsafe_reason_percentages_from_counts,
)

UNSAFE_REASON_TO_IDX = {name: idx for idx, name in enumerate(UNSAFE_REASON_NAMES)}


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    """Adjust checkpoint keys to match current model keys."""
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


def _list_named_subdirs(path: pathlib.Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return sorted([p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _prepare_single_object_assets(object_dir: str, object_name: str) -> tuple[str, pathlib.Path]:
    root_path = pathlib.Path(__file__).resolve().parents[1]
    assets_dir = root_path / "assets"
    source_object_dir = pathlib.Path(object_dir).expanduser().resolve() / object_name
    if not source_object_dir.is_dir():
        raise FileNotFoundError(f"Object folder missing for '{object_name}': {source_object_dir}")

    target_dir_name = f"__offline_eval_single_{object_name}"
    target_root = assets_dir / target_dir_name
    if target_root.exists():
        shutil.rmtree(target_root)

    target_usd_dir = target_root / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)
    link_path = target_usd_dir / object_name
    link_path.symlink_to(source_object_dir, target_is_directory=True)
    return target_dir_name, target_root


def _discover_intermediate_checkpoints(policy_run_dir: str) -> tuple[list[tuple[int, pathlib.Path]], pathlib.Path]:
    run_path = pathlib.Path(policy_run_dir).expanduser().resolve()
    nn_dir = run_path / "nn"
    if not nn_dir.is_dir():
        raise FileNotFoundError(f"Missing nn directory under run folder: {nn_dir}")

    pattern = re.compile(r".*_(\d+)_iters\.pth$")
    checkpoint_entries: list[tuple[int, pathlib.Path]] = []
    for path in sorted(nn_dir.glob("*.pth")):
        match = pattern.match(path.name)
        if match is None:
            continue
        checkpoint_entries.append((int(match.group(1)), path))

    if len(checkpoint_entries) < 2:
        raise ValueError(
            "Need at least two *_iters checkpoints in run/nn to exclude the final one. "
            f"Found {len(checkpoint_entries)} in {nn_dir}."
        )

    checkpoint_entries.sort(key=lambda x: (x[0], x[1].name))
    excluded_final = checkpoint_entries[-1][1]
    return checkpoint_entries[:-1], excluded_final


def _resolve_student_cfg_path() -> str:
    if args_cli.student_cfg is not None:
        cfg_path = pathlib.Path(args_cli.student_cfg).expanduser().resolve()
    else:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "dextrah_lab" / "tasks" / "tg2_inspirehand" / "agents" / "rl_games_ppo_stereo_transformer.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Student cfg file not found: {cfg_path}")
    return str(cfg_path)


def _compute_lift_success(ov_env) -> torch.Tensor:
    table_center_z = ov_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * ov_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(ov_env.cfg, "object_height_thresh", 0.0)
    lift_success = ov_env.object_pos[:, 2] > lift_height_thresh
    if hasattr(ov_env, "good_grasp_mask") and ov_env.good_grasp_mask is not None:
        contact_mask = ov_env.good_grasp_mask.to(device=lift_success.device, dtype=torch.bool)
    elif hasattr(ov_env, "object_contact_counts") and ov_env.object_contact_counts is not None:
        contact_mask = ov_env.object_contact_counts.to(device=lift_success.device) > 0.0
    else:
        contact_mask = torch.ones_like(lift_success, dtype=torch.bool)
    return lift_success & contact_mask


def _resolve_eval_max_steps(ov_env) -> int:
    if args_cli.eval_max_steps is not None:
        return int(args_cli.eval_max_steps)
    max_steps = getattr(ov_env, "distill_max_episode_length", None)
    if max_steps is None:
        max_steps = getattr(ov_env, "max_episode_length", None)
    if max_steps is None:
        max_steps = 1000
    return int(max_steps)


def _resolve_hold_steps(ov_env) -> tuple[int, float]:
    sim_dt = getattr(ov_env.cfg, "sim_dt", None)
    if sim_dt is None and hasattr(ov_env.cfg, "sim"):
        sim_dt = getattr(ov_env.cfg.sim, "dt", None)
    decimation = getattr(ov_env.cfg, "decimation", 1)
    step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
    hold_steps = 1
    if args_cli.eval_lift_hold_s > 0.0 and step_dt > 0.0:
        hold_steps = max(1, int(np.ceil(args_cli.eval_lift_hold_s / step_dt)))
    return hold_steps, step_dt


def _format_reason_percentages(reason_percentages: dict[str, float]) -> str:
    ordered = [(name, float(reason_percentages.get(name, 0.0))) for name in UNSAFE_REASON_NAMES]
    ordered.sort(key=lambda x: (-x[1], x[0]))
    return ", ".join([f"{name}={value:.1f}%" for name, value in ordered])


def _reason_counts_checked(
    reason_idx: torch.Tensor,
    unsafe_mask: torch.Tensor,
    warn_label: str | None = None,
) -> tuple[dict[str, int], int]:
    unsafe_mask = unsafe_mask.to(dtype=torch.bool)
    counts = {
        name: int(((reason_idx == idx) & unsafe_mask).sum().item())
        for name, idx in UNSAFE_REASON_TO_IDX.items()
    }
    total_unsafe = int(unsafe_mask.sum().item())
    classified_total = int(sum(counts.values()))
    unknown_count = max(0, total_unsafe - classified_total)
    if unknown_count > 0:
        label = warn_label if warn_label is not None else "offline unsafe reason classification"
        raise RuntimeError(
            f"{label}: found {unknown_count} unclassified unsafe episodes "
            f"(unsafe_total={total_unsafe}, classified_total={classified_total}). "
            "Fail-fast mode is enabled; no fallback mapping is allowed."
        )
    return counts, total_unsafe


def _save_metrics_npy(payload: dict, policy_run_dir: str) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.metrics_output_npy is not None:
        raw_path = pathlib.Path(args_cli.metrics_output_npy).expanduser().resolve()
        output_path = raw_path.with_name(f"{raw_path.stem}_{timestamp}.npy")
    else:
        run_path = pathlib.Path(policy_run_dir).expanduser().resolve()
        output_path = run_path / f"offline_eval_metrics_{timestamp}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)
    print(f"[INFO] Saved offline evaluation metrics to: {output_path}", flush=True)
    return output_path


class PolicyEvaluator:
    def __init__(self, env, student_cfg_path: str, checkpoint_path: str):
        self.env = env
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        if hasattr(args_cli, "device"):
            self.device = torch.device(args_cli.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(student_cfg_path, "r") as f:
            network_params = yaml.safe_load(f)["params"]

        self.student_obs_type = "policy"
        self.normalize_input = network_params["config"]["normalize_input"]
        self.model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": 1,
            "normalize_value": network_params["config"]["normalize_value"],
            "normalize_input": self.normalize_input,
        }

        builder = ModelBuilder().load(network_params)
        self.model = builder.build(self.model_config).to(self.device)
        self._load_checkpoint(checkpoint_path)

        self.is_rnn = self.model.is_rnn()
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.env_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        if self.is_rnn:
            self.hidden_states = [s.to(self.device) for s in self.model.get_default_rnn_state()]
        else:
            self.hidden_states = None

        self.rgb_buffers_left = None
        self.rgb_buffers_right = None
        self.rgb_buffers = None
        if getattr(self.ov_env, "simulate_stereo", False):
            self.rgb_buffers_left = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width),
                dtype=torch.float32,
                device=self.device,
            )
            self.rgb_buffers_right = torch.zeros_like(self.rgb_buffers_left)
        elif hasattr(self.ov_env.cfg, "img_height") and hasattr(self.ov_env.cfg, "img_width"):
            self.rgb_buffers = torch.zeros(
                (self.num_envs, 3, self.ov_env.cfg.img_height, self.ov_env.cfg.img_width),
                dtype=torch.float32,
                device=self.device,
            )

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = adjust_state_dict_keys(state_dict, self.model.state_dict())
        self.model.load_state_dict(state_dict)
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def _reset_policy_state(self):
        self.prev_actions.zero_()
        self.env_counter.zero_()
        if self.is_rnn:
            self.hidden_states = [s.to(self.device) for s in self.model.get_default_rnn_state()]
        if self.rgb_buffers is not None:
            self.rgb_buffers.zero_()
        if self.rgb_buffers_left is not None:
            self.rgb_buffers_left.zero_()
        if self.rgb_buffers_right is not None:
            self.rgb_buffers_right.zero_()

    def _prepare_obs(self, obs: dict) -> dict:
        obs = dict(obs)
        even_indices = torch.where(self.env_counter % 2 == 0)[0]
        if self.rgb_buffers_left is not None and "img_left" in obs and "img_right" in obs:
            self.rgb_buffers_left[even_indices] = obs["img_left"][even_indices]
            self.rgb_buffers_right[even_indices] = obs["img_right"][even_indices]
            obs["img_left"] = self.rgb_buffers_left
            obs["img_right"] = self.rgb_buffers_right
        elif self.rgb_buffers is not None and "rgb" in obs:
            self.rgb_buffers[even_indices] = obs["rgb"][even_indices]
            obs["rgb"] = self.rgb_buffers
        return obs

    def get_actions(self, obs: dict, deterministic: bool) -> torch.Tensor:
        obs = self._prepare_obs(obs)
        batch_dict = {
            "is_train": False,
            "obs": obs[self.student_obs_type],
            "prev_actions": self.prev_actions,
            "finetune_backbone": False,
        }
        if "img" in obs:
            batch_dict["img"] = obs["img"]
            batch_dict["rgb_data"] = obs["rgb"]
            batch_dict["rgb"] = obs["rgb"]
        if "img_left" in obs:
            batch_dict["img_left"] = obs["img_left"]
            batch_dict["img_right"] = obs["img_right"]
        if self.is_rnn:
            batch_dict["rnn_states"] = self.hidden_states
            batch_dict["seq_length"] = 1
            batch_dict["rnn_masks"] = None

        res_dict = self.model(batch_dict)
        if self.is_rnn:
            self.hidden_states = [s.detach() for s in res_dict["rnn_states"]]

        mus = res_dict["mus"]
        if deterministic:
            actions = mus
        else:
            sigmas = res_dict["sigmas"]
            actions = torch.distributions.Normal(mus, sigmas, validate_args=False).sample()
        actions = torch.clamp(actions, -1.0, 1.0)
        self.prev_actions = actions.detach()
        return actions.detach()

    def evaluate(self, num_episodes: int, max_steps: int, hold_steps: int, deterministic: bool) -> dict:
        self.model.eval()

        total_done = 0
        total_lift_success = 0
        total_unsafe = 0
        reason_counts: dict[str, int] = {name: 0 for name in UNSAFE_REASON_NAMES}

        with torch.no_grad():
            for _ in range(num_episodes):
                obs = self.env.reset()[0]
                self._reset_policy_state()
                dones = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                ever_lifted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                ever_unsafe = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                unsafe_reason_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
                lift_hold_counts = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
                steps = 0

                while steps < max_steps and not dones.all():
                    actions = self.get_actions(obs, deterministic=deterministic)
                    obs, _, out_of_reach, timed_out, _ = self.env.step(actions)

                    out_of_reach = as_bool_mask(out_of_reach, self.num_envs, self.device)
                    timed_out = as_bool_mask(timed_out, self.num_envs, self.device)
                    dones = out_of_reach | timed_out

                    reason_idx = classify_out_of_reach_reasons(
                        ov_env=self.ov_env,
                        out_of_reach=out_of_reach,
                        reason_names=UNSAFE_REASON_NAMES,
                        reason_to_idx=UNSAFE_REASON_TO_IDX,
                        device=self.device,
                    )
                    ever_unsafe = ever_unsafe | out_of_reach
                    classified_unsafe = reason_idx >= 0
                    new_reason = (unsafe_reason_idx < 0) & out_of_reach & classified_unsafe
                    unsafe_reason_idx[new_reason] = reason_idx[new_reason]

                    step_lift_success = _compute_lift_success(self.ov_env)
                    active_envs = ~dones
                    lift_hold_counts = torch.where(
                        active_envs & step_lift_success,
                        lift_hold_counts + 1,
                        torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
                    )
                    ever_lifted = ever_lifted | (lift_hold_counts >= hold_steps)

                    done_indices = dones.nonzero(as_tuple=False).flatten()
                    if self.is_rnn and len(done_indices) > 0:
                        for state in self.hidden_states:
                            state[:, done_indices, ...] = 0.0
                    if len(done_indices) > 0:
                        self.prev_actions[done_indices] = 0.0
                        self.env_counter[done_indices] = 0
                    self.env_counter += 1
                    steps += 1

                total_done += self.num_envs
                total_lift_success += int(ever_lifted.sum().item())
                ep_reason_counts, ep_unsafe_total = _reason_counts_checked(
                    reason_idx=unsafe_reason_idx,
                    unsafe_mask=ever_unsafe,
                    warn_label=None,
                )
                total_unsafe += int(ep_unsafe_total)
                for reason_name in UNSAFE_REASON_NAMES:
                    reason_counts[reason_name] += int(ep_reason_counts[reason_name])

        lift_success_rate = float(total_lift_success) / float(max(total_done, 1))
        unsafe_episode_rate = float(total_unsafe) / float(max(total_done, 1))
        reason_pct = unsafe_reason_percentages_from_counts(reason_counts, total_unsafe, UNSAFE_REASON_NAMES)
        return {
            "lift_success_rate": lift_success_rate,
            "unsafe_episode_rate": unsafe_episode_rate,
            "unsafe_reason_pct": {name: float(reason_pct.get(name, 0.0)) for name in UNSAFE_REASON_NAMES},
            "reason_counts": {name: int(reason_counts[name]) for name in UNSAFE_REASON_NAMES},
            "unsafe_episodes": int(total_unsafe),
            "total_episodes": int(total_done),
        }


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict):
    if args_cli.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be > 0.")

    policy_run_dir = str(pathlib.Path(args_cli.policy_run_dir).expanduser().resolve())
    object_dir = str(pathlib.Path(args_cli.object_dir).expanduser().resolve())
    if not pathlib.Path(policy_run_dir).is_dir():
        raise FileNotFoundError(f"Policy run folder not found: {policy_run_dir}")
    if not pathlib.Path(object_dir).is_dir():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)

    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

    student_cfg_path = _resolve_student_cfg_path()
    object_names = _list_named_subdirs(pathlib.Path(object_dir))
    if len(object_names) == 0:
        raise ValueError(f"No object subfolders found under: {object_dir}")

    checkpoints_to_eval, excluded_final_checkpoint = _discover_intermediate_checkpoints(policy_run_dir)
    print(
        f"[INFO] Evaluating {len(checkpoints_to_eval)} checkpoints from {policy_run_dir} "
        f"(excluded final: {excluded_final_checkpoint.name})",
        flush=True,
    )
    print(f"[INFO] Object set ({len(object_names)}): {object_names}", flush=True)

    checkpoint_results = []
    for iter_num, checkpoint_path in checkpoints_to_eval:
        per_object_metrics = {}
        per_object_lift = []
        per_object_unsafe = []
        aggregate_reason_counts = {name: 0 for name in UNSAFE_REASON_NAMES}
        aggregate_unsafe_eps = 0

        print(f"[INFO] Evaluating checkpoint: {checkpoint_path.name}", flush=True)
        for object_name in object_names:
            objects_dir_name, temp_root = _prepare_single_object_assets(object_dir, object_name)
            try:
                eval_cfg = copy.deepcopy(env_cfg)
                eval_cfg.objects_dir = objects_dir_name
                if hasattr(eval_cfg, "valid_objects_dir") and eval_cfg.objects_dir not in eval_cfg.valid_objects_dir:
                    eval_cfg.valid_objects_dir.append(eval_cfg.objects_dir)

                env = gym.make(args_cli.task, cfg=eval_cfg)
                try:
                    evaluator = PolicyEvaluator(env, student_cfg_path, str(checkpoint_path))
                    max_steps = _resolve_eval_max_steps(evaluator.ov_env)
                    hold_steps, step_dt = _resolve_hold_steps(evaluator.ov_env)
                    metrics = evaluator.evaluate(
                        num_episodes=int(args_cli.eval_episodes),
                        max_steps=max_steps,
                        hold_steps=hold_steps,
                        deterministic=bool(args_cli.deterministic),
                    )
                finally:
                    env.close()
            finally:
                if temp_root.exists():
                    shutil.rmtree(temp_root)

            per_object_metrics[object_name] = {
                "eval/lift_success": float(metrics["lift_success_rate"]),
                "eval/unsafe_episode_rate": float(metrics["unsafe_episode_rate"]),
                "eval/out_of_reach_reason_pct": dict(metrics["unsafe_reason_pct"]),
                "unsafe_episodes": int(metrics["unsafe_episodes"]),
                "total_episodes": int(metrics["total_episodes"]),
            }
            per_object_lift.append(float(metrics["lift_success_rate"]))
            per_object_unsafe.append(float(metrics["unsafe_episode_rate"]))
            aggregate_unsafe_eps += int(metrics["unsafe_episodes"])
            for reason_name in UNSAFE_REASON_NAMES:
                aggregate_reason_counts[reason_name] += int(metrics["reason_counts"][reason_name])

            print(
                f"iter={iter_num} | object={object_name} | "
                f"eval/lift_success/{object_name}: {metrics['lift_success_rate']:.4f} | "
                f"eval/unsafe_episode_rate/{object_name}: {metrics['unsafe_episode_rate']:.4f} | "
                f"eval/out_of_reach_reason_pct/{object_name}: {_format_reason_percentages(metrics['unsafe_reason_pct'])}",
                flush=True,
            )

        avg_lift = float(np.mean(per_object_lift)) if len(per_object_lift) > 0 else 0.0
        avg_unsafe = float(np.mean(per_object_unsafe)) if len(per_object_unsafe) > 0 else 0.0
        avg_reason_pct = unsafe_reason_percentages_from_counts(
            aggregate_reason_counts, aggregate_unsafe_eps, UNSAFE_REASON_NAMES
        )
        avg_reason_pct = {name: float(avg_reason_pct.get(name, 0.0)) for name in UNSAFE_REASON_NAMES}
        print(
            f"iter={iter_num} | eval/lift_success_avg: {avg_lift:.4f} | "
            f"eval/unsafe_episode_rate_avg: {avg_unsafe:.4f} | "
            f"eval/out_of_reach_reason_pct_avg: {_format_reason_percentages(avg_reason_pct)}",
            flush=True,
        )

        flat = {
            **{f"eval/lift_success/{name}": per_object_metrics[name]["eval/lift_success"] for name in object_names},
            **{
                f"eval/unsafe_episode_rate/{name}": per_object_metrics[name]["eval/unsafe_episode_rate"]
                for name in object_names
            },
            **{
                f"eval/out_of_reach_reason_pct/{name}": per_object_metrics[name]["eval/out_of_reach_reason_pct"]
                for name in object_names
            },
            "eval/lift_success_avg": avg_lift,
            "eval/unsafe_episode_rate_avg": avg_unsafe,
            "eval/out_of_reach_reason_pct_avg": avg_reason_pct,
        }
        checkpoint_results.append(
            {
                "iteration": int(iter_num),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "objects": per_object_metrics,
                "averages": {
                    "eval/lift_success_avg": avg_lift,
                    "eval/unsafe_episode_rate_avg": avg_unsafe,
                    "eval/out_of_reach_reason_pct_avg": avg_reason_pct,
                },
                "flat": flat,
            }
        )

    payload = {
        "mode": "offline_student_run_folder",
        "task": args_cli.task,
        "policy_run_dir": policy_run_dir,
        "object_dir": object_dir,
        "student_cfg": student_cfg_path,
        "eval_episodes": int(args_cli.eval_episodes),
        "eval_max_steps": int(args_cli.eval_max_steps) if args_cli.eval_max_steps is not None else None,
        "eval_lift_hold_s": float(args_cli.eval_lift_hold_s),
        "deterministic": bool(args_cli.deterministic),
        "excluded_final_checkpoint": str(excluded_final_checkpoint),
        "checkpoints": checkpoint_results,
    }
    _save_metrics_npy(payload, policy_run_dir=policy_run_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
