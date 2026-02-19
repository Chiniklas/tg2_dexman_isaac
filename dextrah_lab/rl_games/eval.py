"""Evaluate a checkpoint of an RL-Games agent and report lift success/unsafe rate.

Example usage:
    python dextrah_lab/rl_games/eval.py \
        --task Dextrah-TG2-InspireHand-Direct-v0 \
        --eval_episodes 10 \
        --checkpoint /path/to/checkpoint.pth

    python dextrah_lab/rl_games/eval.py \
        --task Dextrah-TG2-InspireHand-Direct-v0 \
        --eval_episodes 10 \
        --deterministic \
        --teacher_policy_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/pretrained_ckpts/multi_object_distillation \
        --teacher_object_dir /home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/teacher_eval
"""

import argparse

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL-Games agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--teacher_policy_dir",
    type=str,
    default=None,
    help="Directory containing per-object teacher policy subfolders.",
)
parser.add_argument(
    "--teacher_object_dir",
    type=str,
    default=None,
    help="Directory containing per-object assets subfolders to validate against teacher policies.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=10,
    help="Total number of episodes to evaluate across all envs.",
)
parser.add_argument(
    "--eval_max_steps",
    type=int,
    default=None,
    help="Max steps per eval episode (defaults to env limit).",
)
parser.add_argument(
    "--eval_lift_hold_s",
    type=float,
    default=0.5,
    help="Require lift condition to hold this many seconds consecutively before counting success.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use deterministic actions during evaluation.",
)
parser.add_argument(
    "--metrics_output_npy",
    type=str,
    default=None,
    help=(
        "Optional output path for metrics .npy. "
        "Defaults to ./teacher_eval_metrics.npy in teacher-folder mode "
        "or ./eval_metrics.npy in single-checkpoint mode."
    ),
)
# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import numpy as np
import os
import pathlib
import shutil
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import dextrah_lab.tasks.dextrah_kuka_allegro.gym_setup
import dextrah_lab.tasks.dextrah_kuka_inspirehand.gym_setup
import dextrah_lab.tasks.tg2_inspirehand.gym_setup

_ENV_HOLDER = {"env": None}
UNSAFE_REASON_NAMES: tuple[str, ...] = (
    "object_out_of_bound",
    "hand_too_far",
    "harmful_collision",
)


def _compute_lift_success(eval_env):
    table_center_z = eval_env.cfg.table_cfg.init_state.pos[2]
    table_top_z = table_center_z + 0.5 * eval_env.cfg.table_size_z
    lift_height_thresh = table_top_z + getattr(eval_env.cfg, "object_height_thresh", 0.0)
    lift_success = eval_env.object_pos[:, 2] > lift_height_thresh
    if hasattr(eval_env, "good_grasp_mask") and eval_env.good_grasp_mask is not None:
        contact_mask = eval_env.good_grasp_mask.to(device=lift_success.device, dtype=torch.bool)
    elif hasattr(eval_env, "object_contact_counts") and eval_env.object_contact_counts is not None:
        contact_mask = eval_env.object_contact_counts.to(device=lift_success.device) > 0.0
    else:
        contact_mask = torch.ones_like(lift_success, dtype=torch.bool)
    lift_success = lift_success & contact_mask
    return lift_success


def _list_named_subdirs(path: pathlib.Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return sorted([p.name for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _validate_teacher_policy_object_dirs(teacher_policy_dir: str, teacher_object_dir: str) -> list[str]:
    policy_path = pathlib.Path(teacher_policy_dir).expanduser().resolve()
    object_path = pathlib.Path(teacher_object_dir).expanduser().resolve()

    policy_names = _list_named_subdirs(policy_path)
    object_names = _list_named_subdirs(object_path)

    if len(policy_names) != len(object_names):
        raise ValueError(
            "Teacher policy/object folder count mismatch: "
            f"{len(policy_names)} policy folders under {policy_path}, "
            f"{len(object_names)} object folders under {object_path}."
        )

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

    print(
        "[INFO] Teacher policy/object folder names validated: "
        f"{len(policy_names)} matched entries."
    )
    return policy_names


def _resolve_teacher_checkpoint(policy_root_dir: str, object_name: str) -> str:
    policy_object_dir = pathlib.Path(policy_root_dir).expanduser().resolve() / object_name
    if not policy_object_dir.is_dir():
        raise FileNotFoundError(f"Policy folder missing for object '{object_name}': {policy_object_dir}")

    preferred = policy_object_dir / "dextrah_lstm.pth"
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


def _prepare_teacher_single_object_dir(teacher_object_dir: str, object_name: str) -> tuple[str, pathlib.Path]:
    root_path = pathlib.Path(__file__).resolve().parents[1]
    assets_dir = root_path / "assets"
    source_object_dir = pathlib.Path(teacher_object_dir).expanduser().resolve() / object_name
    if not source_object_dir.is_dir():
        raise FileNotFoundError(f"Object folder missing for '{object_name}': {source_object_dir}")

    target_dir_name = f"__teacher_eval_single_{object_name}"
    target_root = assets_dir / target_dir_name
    if target_root.exists():
        shutil.rmtree(target_root)

    target_usd_dir = target_root / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    link_path = target_usd_dir / object_name
    link_path.symlink_to(source_object_dir, target_is_directory=True)
    return target_dir_name, target_root


def _resolve_checkpoint_path(agent_cfg: dict, explicit_checkpoint: str | None = None) -> str:
    if explicit_checkpoint is not None:
        return retrieve_file_path(explicit_checkpoint)

    if args_cli.checkpoint is not None:
        return retrieve_file_path(args_cli.checkpoint)

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
    return get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])


def _resolve_eval_max_steps(eval_env) -> int:
    if args_cli.eval_max_steps is not None:
        return int(args_cli.eval_max_steps)

    sim_dt = getattr(eval_env.cfg, "sim_dt", None)
    if sim_dt is None and hasattr(eval_env.cfg, "sim"):
        sim_dt = getattr(eval_env.cfg.sim, "dt", None)
    decimation = getattr(eval_env.cfg, "decimation", 1)
    if sim_dt is not None and sim_dt > 0:
        return int(max(1, round(4.0 / (sim_dt * decimation))))

    max_steps = getattr(eval_env, "distill_max_episode_length", None)
    if max_steps is None:
        max_steps = getattr(eval_env, "max_episode_length", None)
    if max_steps is None:
        max_steps = 1000
    return int(max_steps)


def _resolve_eval_hold_steps(eval_env) -> tuple[int, float]:
    sim_dt = getattr(eval_env.cfg, "sim_dt", None)
    if sim_dt is None and hasattr(eval_env.cfg, "sim"):
        sim_dt = getattr(eval_env.cfg.sim, "dt", None)
    decimation = getattr(eval_env.cfg, "decimation", 1)
    step_dt = float(sim_dt * decimation) if sim_dt is not None else 0.0
    hold_steps = 1
    if args_cli.eval_lift_hold_s > 0.0 and step_dt > 0.0:
        hold_steps = max(1, int(math.ceil(args_cli.eval_lift_hold_s / step_dt)))
    return hold_steps, step_dt


def _register_rlgames_env() -> None:
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: _ENV_HOLDER["env"]}
    )


def _as_bool_mask(values, num_envs: int, device: torch.device) -> torch.Tensor:
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


def _extract_timeout_mask(info, num_envs: int, device: torch.device) -> torch.Tensor:
    timeout_keys = ("time_outs", "timed_out", "timeouts", "time_out", "truncated", "truncation")
    if isinstance(info, dict):
        for key in timeout_keys:
            if key in info:
                return _as_bool_mask(info[key], num_envs, device)
        return torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if isinstance(info, (list, tuple)) and len(info) == num_envs and num_envs > 0 and isinstance(info[0], dict):
        values = []
        for item in info:
            timeout_flag = False
            for key in timeout_keys:
                if key in item:
                    timeout_flag = bool(torch.as_tensor(item[key]).any().item())
                    break
            values.append(timeout_flag)
        return _as_bool_mask(values, num_envs, device)
    return torch.zeros((num_envs,), dtype=torch.bool, device=device)


def _extract_out_of_reach_mask(eval_env, num_envs: int, device: torch.device) -> torch.Tensor:
    # DirectRLEnv exposes non-timeout terminations in reset_terminated.
    if hasattr(eval_env, "reset_terminated"):
        return _as_bool_mask(eval_env.reset_terminated, num_envs, device)
    return torch.zeros((num_envs,), dtype=torch.bool, device=device)


def _compute_out_of_reach_reason_masks(eval_env, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    # Mirrors tg2_inspirehand _get_dones() reason components where available.
    masks: dict[str, torch.Tensor] = {}
    zeros = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if not hasattr(eval_env, "object_pos"):
        return masks
    object_out_of_bound = zeros.clone()
    hand_too_far = zeros.clone()
    harmful_collision = zeros.clone()

    try:
        object_outside_upper_x = eval_env.object_pos[:, 0] > (eval_env.cfg.x_center + eval_env.cfg.x_width / 2.0)
        object_outside_lower_x = eval_env.object_pos[:, 0] < (eval_env.cfg.x_center - eval_env.cfg.x_width / 2.0)
        object_outside_upper_y = eval_env.object_pos[:, 1] > (eval_env.cfg.y_center + eval_env.cfg.y_width / 2.0)
        object_outside_lower_y = eval_env.object_pos[:, 1] < (eval_env.cfg.y_center - eval_env.cfg.y_width / 2.0)
        object_too_low = eval_env.object_pos[:, 2] < 0.2

        object_out_of_bound = (
            object_outside_upper_x
            | object_outside_lower_x
            | object_outside_upper_y
            | object_outside_lower_y
            | object_too_low
        )
    except Exception:
        pass

    hand_too_close = zeros.clone()
    arm_table_contact = zeros.clone()
    try:
        bbox_margin = getattr(eval_env.cfg, "hand_bbox_margin", 0.0)
        table_half_x = eval_env.cfg.table_size_x * 0.5 + bbox_margin
        table_half_y = eval_env.cfg.table_size_y * 0.5 + bbox_margin

        if hasattr(eval_env, "table_pos"):
            table_pos = eval_env.table_pos
            table_pos_z = eval_env.table_pos[:, 2]
        else:
            table_pos = eval_env.table.data.root_pos_w - eval_env.scene.env_origins
            table_pos_z = table_pos[:, 2]
        table_top_z = table_pos_z + eval_env.cfg.table_size_z * 0.5

        middle_pos = eval_env.robot.data.body_pos_w[:, eval_env.middle_link_0_body_idx] - eval_env.scene.env_origins
        middle_x_out = (middle_pos[:, 0] > (table_pos[:, 0] + table_half_x)) | (
            middle_pos[:, 0] < (table_pos[:, 0] - table_half_x)
        )
        middle_y_out = (middle_pos[:, 1] > (table_pos[:, 1] + table_half_y)) | (
            middle_pos[:, 1] < (table_pos[:, 1] - table_half_y)
        )
        middle_z_out = (middle_pos[:, 2] < table_top_z) | (middle_pos[:, 2] > (table_top_z + 1.0 + bbox_margin))
        hand_too_far = middle_x_out | middle_y_out | middle_z_out

        hand_min_z = eval_env.hand_pos[..., 2].min(dim=1).values
        clearance_thresh = table_top_z + 0.01
        hand_too_close = hand_min_z < clearance_thresh
    except Exception:
        pass

    try:
        arm_table_contact = _as_bool_mask(eval_env.arm_table_contact_mask, num_envs, device)
    except Exception:
        pass

    harmful_collision = hand_too_close | arm_table_contact
    masks["object_out_of_bound"] = object_out_of_bound
    masks["hand_too_far"] = hand_too_far
    masks["harmful_collision"] = harmful_collision

    return masks


def _select_primary_reason(reason_masks: dict[str, torch.Tensor], env_idx: int) -> str | None:
    reason_order = UNSAFE_REASON_NAMES
    for name in reason_order:
        mask = reason_masks.get(name)
        if mask is not None and bool(mask[env_idx].item()):
            return name
    return None


def _format_reason_percentages(reason_percentages: dict[str, float]) -> str:
    if not reason_percentages:
        return "none"
    ordered = sorted(reason_percentages.items(), key=lambda x: (-x[1], x[0]))
    return ", ".join([f"{name}={value:.1f}%" for name, value in ordered])


def _reason_counts_to_percentages(reason_counts: dict[str, int], total_unsafe_eps: int) -> dict[str, float]:
    if total_unsafe_eps <= 0:
        return {}
    return {
        name: 100.0 * float(count) / float(total_unsafe_eps)
        for name, count in reason_counts.items()
    }


def _reason_percentages_with_defaults(reason_percentages: dict[str, float]) -> dict[str, float]:
    return {
        name: float(reason_percentages.get(name, 0.0))
        for name in UNSAFE_REASON_NAMES
    }


def _save_metrics_npy(metrics_payload: dict, teacher_mode: bool) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.metrics_output_npy is not None:
        raw_output_path = pathlib.Path(args_cli.metrics_output_npy).expanduser().resolve()
        output_path = raw_output_path.with_name(f"{raw_output_path.stem}_{timestamp}.npy")
    else:
        default_name = (
            f"teacher_eval_metrics_{timestamp}.npy"
            if teacher_mode
            else f"eval_metrics_{timestamp}.npy"
        )
        output_path = pathlib.Path.cwd() / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, metrics_payload, allow_pickle=True)
    print(f"[INFO] Saved evaluation metrics to: {output_path}")


def _run_eval_for_checkpoint(
    checkpoint_path: str, objects_dir_override: str | None = None
) -> tuple[float, float, int, int, dict[str, int], int]:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if objects_dir_override is not None:
        env_cfg.objects_dir = objects_dir_override
        if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
            env_cfg.valid_objects_dir.append(env_cfg.objects_dir)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    resume_path = _resolve_checkpoint_path(agent_cfg, explicit_checkpoint=checkpoint_path)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    _ENV_HOLDER["env"] = env
    try:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading model checkpoint from: {agent_cfg['params']['load_path']}")

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        eval_env = env.unwrapped
        num_envs = eval_env.num_envs
        max_steps = _resolve_eval_max_steps(eval_env)
        hold_steps, step_dt = _resolve_eval_hold_steps(eval_env)
        print(
            f"[INFO] Eval lift hold gate: {hold_steps} steps "
            f"(~{args_cli.eval_lift_hold_s:.3f}s target, dt={step_dt:.5f}s)"
        )

        total_target = int(args_cli.eval_episodes)
        total_done = 0
        total_success = 0.0
        total_unsafe = 0.0
        unsafe_reason_counts: dict[str, int] = {}
        steps_per_env = torch.zeros((num_envs,), dtype=torch.long, device=args_cli.device)
        dones = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
        ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
        lift_hold_counts = torch.zeros((num_envs,), dtype=torch.long, device=args_cli.device)

        while total_done < total_target:
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_t, is_deterministic=args_cli.deterministic)
                obs, _, env_dones, _ = env.step(actions)
            steps_per_env += 1
            capped_timed_out = steps_per_env >= max_steps
            env_dones = _as_bool_mask(env_dones, num_envs, args_cli.device)
            dones = env_dones | capped_timed_out
            out_of_reach = _extract_out_of_reach_mask(eval_env, num_envs, args_cli.device)
            reason_masks = _compute_out_of_reach_reason_masks(eval_env, num_envs, args_cli.device)
            step_lift_success = _compute_lift_success(eval_env)
            active_envs = ~dones
            lift_hold_counts = torch.where(
                active_envs & step_lift_success,
                lift_hold_counts + 1,
                torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
            )
            hold_lift_success = lift_hold_counts >= hold_steps
            ever_lifted = ever_lifted | hold_lift_success

            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).flatten()
                for idx in done_indices.tolist():
                    if total_done >= total_target:
                        break
                    total_success += float(ever_lifted[idx].item())
                    unsafe_flag = False
                    if bool(out_of_reach[idx].item()):
                        reason = _select_primary_reason(reason_masks, idx)
                        if reason is not None:
                            unsafe_reason_counts[reason] = unsafe_reason_counts.get(reason, 0) + 1
                            unsafe_flag = True
                    total_unsafe += float(unsafe_flag)
                    total_done += 1
                    steps_per_env[idx] = 0
                    ever_lifted[idx] = False
                    lift_hold_counts[idx] = 0
                if agent.is_rnn and agent.states is not None:
                    new_states = []
                    for s in agent.states:
                        s_clone = s.clone()
                        s_clone[:, done_indices, :] = 0.0
                        new_states.append(s_clone)
                    agent.states = new_states
                dones = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)

        avg_success = total_success / max(total_done, 1)
        unsafe_episode_rate = total_unsafe / max(total_done, 1)
        total_unsafe_eps = int(round(total_unsafe))
        return avg_success, unsafe_episode_rate, total_done, num_envs, unsafe_reason_counts, total_unsafe_eps
    finally:
        env.close()
        _ENV_HOLDER["env"] = None


def main():
    if args_cli.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be > 0 for evaluation.")
    if (args_cli.teacher_policy_dir is None) != (args_cli.teacher_object_dir is None):
        raise ValueError("Provide both --teacher_policy_dir and --teacher_object_dir together.")
    _register_rlgames_env()

    if args_cli.teacher_policy_dir is not None and args_cli.teacher_object_dir is not None:
        if args_cli.checkpoint is not None:
            raise ValueError("Do not pass --checkpoint when using teacher folder evaluation mode.")

        object_names = _validate_teacher_policy_object_dirs(args_cli.teacher_policy_dir, args_cli.teacher_object_dir)
        per_object_lift_results = []
        per_object_unsafe_rates = []
        per_object_metrics: dict[str, dict] = {}
        flat_metrics: dict[str, object] = {}
        aggregate_reason_counts: dict[str, int] = {}
        aggregate_unsafe_eps = 0
        for object_name in object_names:
            checkpoint_path = _resolve_teacher_checkpoint(args_cli.teacher_policy_dir, object_name)
            objects_dir_name, temp_root = _prepare_teacher_single_object_dir(args_cli.teacher_object_dir, object_name)
            try:
                obj_success, obj_unsafe, total_done, num_envs, reason_counts, obj_unsafe_eps = _run_eval_for_checkpoint(
                    checkpoint_path, objects_dir_override=objects_dir_name
                )
            finally:
                if temp_root.exists():
                    shutil.rmtree(temp_root)

            per_object_lift_results.append(obj_success)
            per_object_unsafe_rates.append(obj_unsafe)
            aggregate_unsafe_eps += obj_unsafe_eps
            for reason_name, count in reason_counts.items():
                aggregate_reason_counts[reason_name] = aggregate_reason_counts.get(reason_name, 0) + count
            reason_percentages = _reason_counts_to_percentages(reason_counts, obj_unsafe_eps)
            reason_percentages_full = _reason_percentages_with_defaults(reason_percentages)
            per_object_metrics[object_name] = {
                "eval/lift_success": float(obj_success),
                "eval/unsafe_episode_rate": float(obj_unsafe),
                "eval/out_of_reach_reason_pct": reason_percentages_full,
                "total_episodes": int(total_done),
                "num_envs": int(num_envs),
                "unsafe_episodes": int(obj_unsafe_eps),
            }
            flat_metrics[f"eval/lift_success/{object_name}"] = float(obj_success)
            flat_metrics[f"eval/unsafe_episode_rate/{object_name}"] = float(obj_unsafe)
            flat_metrics[f"eval/out_of_reach_reason_pct/{object_name}"] = reason_percentages_full
            print(
                f"eval/lift_success/{object_name}: {obj_success:.4f} "
                f"| eval/unsafe_episode_rate/{object_name}: {obj_unsafe:.4f} "
                f"| eval/out_of_reach_reason_pct/{object_name}: {_format_reason_percentages(reason_percentages)} "
                f"(total episodes: {total_done}, envs: {num_envs})"
            )

        avg_success = sum(per_object_lift_results) / max(len(per_object_lift_results), 1)
        avg_unsafe = sum(per_object_unsafe_rates) / max(len(per_object_unsafe_rates), 1)
        avg_reason_percentages = _reason_counts_to_percentages(aggregate_reason_counts, aggregate_unsafe_eps)
        avg_reason_percentages_full = _reason_percentages_with_defaults(avg_reason_percentages)
        print(f"eval/lift_success_avg: {avg_success:.4f} (objects: {len(per_object_lift_results)})")
        print(f"eval/unsafe_episode_rate_avg: {avg_unsafe:.4f} (objects: {len(per_object_unsafe_rates)})")
        print(f"eval/out_of_reach_reason_pct_avg: {_format_reason_percentages(avg_reason_percentages)}")
        flat_metrics["eval/lift_success_avg"] = float(avg_success)
        flat_metrics["eval/unsafe_episode_rate_avg"] = float(avg_unsafe)
        flat_metrics["eval/out_of_reach_reason_pct_avg"] = avg_reason_percentages_full
        metrics_payload = {
            "mode": "teacher_folder",
            "task": args_cli.task,
            "teacher_policy_dir": str(pathlib.Path(args_cli.teacher_policy_dir).expanduser().resolve()),
            "teacher_object_dir": str(pathlib.Path(args_cli.teacher_object_dir).expanduser().resolve()),
            "eval_episodes": int(args_cli.eval_episodes),
            "deterministic": bool(args_cli.deterministic),
            "objects": per_object_metrics,
            "averages": {
                "eval/lift_success_avg": float(avg_success),
                "eval/unsafe_episode_rate_avg": float(avg_unsafe),
                "eval/out_of_reach_reason_pct_avg": avg_reason_percentages_full,
            },
            "flat": flat_metrics,
        }
        _save_metrics_npy(metrics_payload, teacher_mode=True)
        return

    avg_success, unsafe_episode_rate, total_done, num_envs, reason_counts, unsafe_eps = _run_eval_for_checkpoint(
        checkpoint_path=args_cli.checkpoint
    )
    reason_percentages = _reason_counts_to_percentages(reason_counts, unsafe_eps)
    print(
        f"eval/lift_success: {avg_success:.4f} | "
        f"eval/unsafe_episode_rate: {unsafe_episode_rate:.4f} | "
        f"eval/out_of_reach_reason_pct: {_format_reason_percentages(reason_percentages)} "
        f"(total episodes: {total_done}, envs: {num_envs})"
    )
    metrics_payload = {
        "mode": "single_checkpoint",
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "eval_episodes": int(args_cli.eval_episodes),
        "deterministic": bool(args_cli.deterministic),
        "metrics": {
            "eval/lift_success": float(avg_success),
            "eval/unsafe_episode_rate": float(unsafe_episode_rate),
            "eval/out_of_reach_reason_pct": _reason_percentages_with_defaults(reason_percentages),
        },
        "total_episodes": int(total_done),
        "num_envs": int(num_envs),
        "unsafe_episodes": int(unsafe_eps),
    }
    _save_metrics_npy(metrics_payload, teacher_mode=False)


if __name__ == "__main__":
    main()
    simulation_app.close()
