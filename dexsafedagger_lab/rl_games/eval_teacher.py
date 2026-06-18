"""Evaluate teacher policy checkpoints and report lift success/unsafe rate.

Example usage:
    python dexsafedagger_lab/rl_games/eval_teacher.py \
        --task DexSafeDagger-TG2-InspireHand-Direct-v0 \
        --eval_episodes 10 \
        --checkpoint /path/to/checkpoint.pth

    python dexsafedagger_lab/rl_games/eval_teacher.py \
        --task DexSafeDagger-TG2-InspireHand-Direct-v0 \
        --eval_episodes 10 \
        --teacher_policy_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/pretrained_ckpts/multi_object_distillation \
        --teacher_object_dir /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/assets/teacher_eval
"""

import argparse
import copy
import json

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Evaluate teacher policies (single checkpoint or teacher folder).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to single teacher model checkpoint.")
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
    help="Number of eval rollouts; each rollout evaluates all env slots (matches student eval semantics).",
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
    "--metrics_output_npy",
    type=str,
    default=None,
    help=(
        "Optional output path for metrics JSON file (legacy flag name kept for compatibility). "
        "Defaults to the TensorBoard run directory under ./logs. "
        "If a relative path is provided, it is resolved under the same run directory."
    ),
)
parser.add_argument(
    "--file_name_head",
    type=str,
    default=None,
    help=(
        "Optional output JSON filename prefix override. "
        "Example: --file_name_head student_eval_metrics "
        "produces <prefix>_<timestamp>.json. "
        "If unset, defaults remain teacher_eval_metrics / eval_metrics."
    ),
)
parser.add_argument(
    "--tb_logdir",
    type=str,
    default=None,
    help=(
        "Optional TensorBoard log directory. "
        "Defaults to ./logs/teacher_eval_tb_<timestamp> in teacher-folder mode "
        "or ./logs/eval_tb_<timestamp> in single-checkpoint mode."
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
import pathlib
import shutil
import time
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter

from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import dexsafedagger_lab.tasks.dexsafedagger_kuka_allegro.gym_setup
import dexsafedagger_lab.tasks.dexsafedagger_kuka_inspirehand.gym_setup
import dexsafedagger_lab.tasks.tg2_inspirehand.gym_setup
from dexsafedagger_lab.distillation_new.eval_utils import (
    UNSAFE_REASON_NAMES,
    classify_out_of_reach_reasons,
    unsafe_reason_percentages_from_counts,
)

_ENV_HOLDER = {"env": None}
_PROGRESS_TIME_INTERVAL_S = 20.0
_TB_LOGDIR_HOLDER = {"path": None}
_TB_STATE = {"disabled": False}


def _default_logs_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "logs"


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


def _prepare_teacher_multi_object_dir(
    teacher_object_dir: str, object_names: list[str]
) -> tuple[str, pathlib.Path]:
    root_path = pathlib.Path(__file__).resolve().parents[1]
    assets_dir = root_path / "assets"
    source_root = pathlib.Path(teacher_object_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Teacher object directory missing: {source_root}")

    target_dir_name = "__teacher_eval_multi"
    target_root = assets_dir / target_dir_name
    if target_root.exists():
        shutil.rmtree(target_root)

    target_usd_dir = target_root / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    for object_name in object_names:
        source_object_dir = source_root / object_name
        if not source_object_dir.is_dir():
            raise FileNotFoundError(f"Object folder missing for '{object_name}': {source_object_dir}")
        link_path = target_usd_dir / object_name
        link_path.symlink_to(source_object_dir, target_is_directory=True)

    return target_dir_name, target_root


def _resolve_checkpoint_path(agent_cfg: dict, explicit_checkpoint: str | None = None) -> str:
    if explicit_checkpoint is not None:
        return retrieve_file_path(explicit_checkpoint)

    if args_cli.checkpoint is None:
        raise ValueError(
            "Single teacher checkpoint mode requires --checkpoint."
        )
    return retrieve_file_path(args_cli.checkpoint)


def _resolve_eval_max_steps(eval_env) -> int:
    if args_cli.eval_max_steps is not None:
        return int(args_cli.eval_max_steps)

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


def _reason_counts_from_episode(
    unsafe_reason_idx: torch.Tensor,
    unsafe_mask: torch.Tensor,
    scope_label: str,
) -> dict[str, int]:
    counts = {
        name: int((unsafe_reason_idx == idx).sum().item())
        for idx, name in enumerate(UNSAFE_REASON_NAMES)
    }
    total_unsafe = int(unsafe_mask.sum().item())
    classified_total = int(sum(counts.values()))
    unknown_count = max(0, total_unsafe - classified_total)
    if unknown_count > 0:
        raise RuntimeError(
            f"{scope_label}: found {unknown_count} unclassified unsafe episodes "
            f"(unsafe_total={total_unsafe}, classified_total={classified_total}). "
            "Fail-fast mode is enabled; no fallback mapping is allowed."
        )
    return counts


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


def _save_metrics_json(metrics_payload: dict, teacher_mode: bool) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_head = None
    if isinstance(args_cli.file_name_head, str):
        candidate = args_cli.file_name_head.strip()
        if len(candidate) > 0:
            custom_head = candidate
    if args_cli.metrics_output_npy is not None:
        raw_output_path = pathlib.Path(args_cli.metrics_output_npy).expanduser()
        if not raw_output_path.is_absolute():
            if _TB_LOGDIR_HOLDER["path"] is not None:
                raw_output_path = pathlib.Path(_TB_LOGDIR_HOLDER["path"]) / raw_output_path
            else:
                raw_output_path = _default_logs_root() / raw_output_path
        raw_output_path = raw_output_path.resolve()
        stem = custom_head if custom_head is not None else raw_output_path.stem
        output_path = raw_output_path.with_name(f"{stem}_{timestamp}.json")
    else:
        if custom_head is not None:
            default_name = f"{custom_head}_{timestamp}.json"
        else:
            default_name = (
                f"teacher_eval_metrics_{timestamp}.json"
                if teacher_mode
                else f"eval_metrics_{timestamp}.json"
            )
        if _TB_LOGDIR_HOLDER["path"] is not None:
            output_path = pathlib.Path(_TB_LOGDIR_HOLDER["path"]) / default_name
        else:
            output_path = _default_logs_root() / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics_payload), f, indent=2, sort_keys=True)
    print(f"[INFO] Saved evaluation metrics JSON to: {output_path}")


def _create_tb_writer(teacher_mode: bool) -> tuple[SummaryWriter, pathlib.Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _TB_STATE["disabled"] = False
    if args_cli.tb_logdir is not None:
        tb_path = pathlib.Path(args_cli.tb_logdir).expanduser().resolve()
    else:
        default_name = (
            f"teacher_eval_tb_{timestamp}"
            if teacher_mode
            else f"eval_tb_{timestamp}"
        )
        tb_path = _default_logs_root() / default_name
    tb_path.mkdir(parents=True, exist_ok=True)
    _TB_LOGDIR_HOLDER["path"] = str(tb_path)
    writer = SummaryWriter(str(tb_path))
    print(f"[INFO] TensorBoard logdir: {tb_path}", flush=True)
    return writer, tb_path


def _tb_log_reason_percentages(writer: SummaryWriter, prefix: str, reason_pct: dict[str, float], step: int) -> None:
    if writer is None or _TB_STATE["disabled"]:
        return
    for reason_name in UNSAFE_REASON_NAMES:
        _tb_add_scalar(
            writer,
            f"{prefix}/unsafe_reason_pct/{reason_name}",
            float(reason_pct.get(reason_name, 0.0)),
            step,
        )


def _tb_add_scalar(writer: SummaryWriter | None, tag: str, value: float, step: int) -> None:
    if writer is None or _TB_STATE["disabled"]:
        return
    try:
        writer.add_scalar(tag, value, step)
    except Exception as exc:
        _TB_STATE["disabled"] = True
        print(
            f"[WARN] TensorBoard write failed and will be disabled for this run: {exc}",
            flush=True,
        )


def _resolve_eval_object_names_and_idx(
    eval_env, num_envs: int, device: torch.device, fallback_names: list[str] | None = None
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


def _run_eval_for_checkpoint(
    checkpoint_path: str, objects_dir_override: str | None = None
) -> tuple[float, float, int, int, dict[str, float], int]:
    eval_start_t = time.time()
    print(
        f"[INFO] Eval start: task={args_cli.task}, objects_dir={objects_dir_override}, "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )

    stage_t = time.time()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[INFO] Parsed env cfg in {time.time() - stage_t:.1f}s", flush=True)
    if objects_dir_override is not None:
        env_cfg.objects_dir = objects_dir_override
        if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
            env_cfg.valid_objects_dir.append(env_cfg.objects_dir)

    stage_t = time.time()
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    resume_path = _resolve_checkpoint_path(agent_cfg, explicit_checkpoint=checkpoint_path)
    print(
        f"[INFO] Loaded agent cfg + resolved checkpoint in {time.time() - stage_t:.1f}s",
        flush=True,
    )

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    stage_t = time.time()
    print("[INFO] Creating evaluation environment...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    print(
        f"[INFO] Environment ready in {time.time() - stage_t:.1f}s "
        f"(num_envs={env.unwrapped.num_envs}).",
        flush=True,
    )
    _ENV_HOLDER["env"] = env
    try:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO] Loading model checkpoint from: {agent_cfg['params']['load_path']}", flush=True)

        stage_t = time.time()
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
        print(f"[INFO] Player initialized in {time.time() - stage_t:.1f}s", flush=True)

        stage_t = time.time()
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
        print(f"[INFO] First reset complete in {time.time() - stage_t:.1f}s", flush=True)

        eval_env = env.unwrapped
        num_envs = eval_env.num_envs
        max_steps = _resolve_eval_max_steps(eval_env)
        hold_steps, step_dt = _resolve_eval_hold_steps(eval_env)
        print(
            f"[INFO] Eval lift hold gate: {hold_steps} steps "
            f"(~{args_cli.eval_lift_hold_s:.3f}s target, dt={step_dt:.5f}s)",
            flush=True,
        )

        total_rollouts = int(args_cli.eval_episodes)
        # Automatic progress defaults: about 10 rollout-based updates plus a time heartbeat.
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
        success_rates: list[float] = []
        unsafe_rates: list[float] = []
        total_reason_counts = {name: 0 for name in UNSAFE_REASON_NAMES}
        reason_to_idx = {name: idx for idx, name in enumerate(UNSAFE_REASON_NAMES)}
        total_unsafe_eps = 0
        for rollout_idx in range(total_rollouts):
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            if agent.is_rnn:
                agent.init_rnn()
            dones = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            ever_unsafe_terminated = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            unsafe_reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=args_cli.device)
            steps = 0
            lift_hold_counts = torch.zeros((num_envs,), dtype=torch.long, device=args_cli.device)

            while steps < max_steps and not bool(dones.all().item()):
                # Use no_grad (not inference_mode) so env tensors remain mutable across reset calls.
                with torch.no_grad():
                    obs_t = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs_t, is_deterministic=False)
                    obs, _, env_dones, info = env.step(actions)
                env_dones = _as_bool_mask(env_dones, num_envs, args_cli.device)
                out_of_reach = _extract_out_of_reach_mask(eval_env, num_envs, args_cli.device)
                timed_out = _extract_timeout_mask(info, num_envs, args_cli.device)
                # If timeout signals are unavailable, treat done-but-not-out_of_reach as timeout.
                timed_out = timed_out | (env_dones & (~out_of_reach))
                dones = out_of_reach | timed_out
                reason_idx = classify_out_of_reach_reasons(
                    ov_env=eval_env,
                    out_of_reach=out_of_reach,
                    reason_names=UNSAFE_REASON_NAMES,
                    reason_to_idx=reason_to_idx,
                    device=args_cli.device,
                )
                ever_unsafe_terminated = ever_unsafe_terminated | out_of_reach
                classified_out_of_reach = reason_idx >= 0
                new_reason_mask = (unsafe_reason_idx < 0) & out_of_reach & classified_out_of_reach
                unsafe_reason_idx[new_reason_mask] = reason_idx[new_reason_mask]

                if agent.is_rnn and agent.states is not None and bool(dones.any().item()):
                    done_indices = dones.nonzero(as_tuple=False).flatten()
                    new_states = []
                    for s in agent.states:
                        s_clone = s.clone()
                        s_clone[:, done_indices, :] = 0.0
                        new_states.append(s_clone)
                    agent.states = new_states

                step_lift_success = _compute_lift_success(eval_env)
                active_envs = ~dones
                lift_hold_counts = torch.where(
                    active_envs & step_lift_success,
                    lift_hold_counts + 1,
                    torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
                )
                hold_lift_success = lift_hold_counts >= hold_steps
                ever_lifted = ever_lifted | hold_lift_success
                steps += 1

            if steps >= max_steps:
                dones = torch.ones_like(dones)

            success_rates.append(ever_lifted.float().mean().item())
            unsafe_rates.append(ever_unsafe_terminated.float().mean().item())
            unsafe_count = int(ever_unsafe_terminated.sum().item())
            total_unsafe_eps += unsafe_count
            rollout_reason_counts = _reason_counts_from_episode(
                unsafe_reason_idx=unsafe_reason_idx,
                unsafe_mask=ever_unsafe_terminated,
                scope_label=f"single-checkpoint rollout {rollout_idx + 1}",
            )
            for name in UNSAFE_REASON_NAMES:
                total_reason_counts[name] += rollout_reason_counts[name]

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

        avg_success = float(np.mean(success_rates)) if len(success_rates) > 0 else 0.0
        unsafe_episode_rate = float(np.mean(unsafe_rates)) if len(unsafe_rates) > 0 else 0.0
        eval_reason_pct = unsafe_reason_percentages_from_counts(
            total_reason_counts,
            total_unsafe_eps,
            UNSAFE_REASON_NAMES,
        )
        total_done = int(total_rollouts * num_envs)
        print(
            f"[INFO] Eval complete in {time.time() - eval_start_t:.1f}s: "
            f"episodes={total_done}, lift_success={avg_success:.4f}, unsafe_rate={unsafe_episode_rate:.4f}",
            flush=True,
        )
        return avg_success, unsafe_episode_rate, total_done, num_envs, eval_reason_pct, total_unsafe_eps
    finally:
        env.close()
        _ENV_HOLDER["env"] = None


def _run_eval_for_teacher_pool(
    teacher_policy_dir: str,
    object_names: list[str],
    objects_dir_override: str,
) -> tuple[float, float, int, int, dict[str, float], int, dict[str, dict]]:
    eval_start_t = time.time()
    print(
        f"[INFO] Eval start (multi-teacher): task={args_cli.task}, "
        f"objects_dir={objects_dir_override}, num_objects={len(object_names)}",
        flush=True,
    )

    stage_t = time.time()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.objects_dir = objects_dir_override
    if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
        env_cfg.valid_objects_dir.append(env_cfg.objects_dir)
    # Teacher standalone eval needs per-env multi-object spawning while keeping
    # teacher observations (distillation=False).
    env_cfg.multi_object_eval = True
    print(f"[INFO] Parsed env cfg in {time.time() - stage_t:.1f}s", flush=True)

    stage_t = time.time()
    agent_cfg_template = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    checkpoint_map = {
        object_name: _resolve_teacher_checkpoint(teacher_policy_dir, object_name)
        for object_name in object_names
    }
    print(
        f"[INFO] Loaded agent cfg + resolved {len(checkpoint_map)} checkpoints in {time.time() - stage_t:.1f}s",
        flush=True,
    )

    rl_device = agent_cfg_template["params"]["config"]["device"]
    clip_obs = agent_cfg_template["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg_template["params"]["env"].get("clip_actions", math.inf)

    stage_t = time.time()
    print("[INFO] Creating evaluation environment...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    print(
        f"[INFO] Environment ready in {time.time() - stage_t:.1f}s "
        f"(num_envs={env.unwrapped.num_envs}).",
        flush=True,
    )
    _ENV_HOLDER["env"] = env
    try:
        eval_env = env.unwrapped
        num_envs = eval_env.num_envs
        max_steps = _resolve_eval_max_steps(eval_env)
        hold_steps, step_dt = _resolve_eval_hold_steps(eval_env)
        eval_object_names, _ = _resolve_eval_object_names_and_idx(
            eval_env, num_envs, args_cli.device, fallback_names=object_names
        )
        print(
            f"[INFO] Teacher pool object names from env: {eval_object_names} "
            f"(requested={len(object_names)})",
            flush=True,
        )
        missing_policy = [name for name in eval_object_names if name not in checkpoint_map]
        if len(missing_policy) > 0:
            raise ValueError(
                "Missing teacher checkpoints for object names exposed by eval env: "
                f"{missing_policy}"
            )

        players_by_name: dict[str, BasePlayer] = {}
        for object_name in eval_object_names:
            resume_path = retrieve_file_path(checkpoint_map[object_name])
            cfg = copy.deepcopy(agent_cfg_template)
            cfg["params"]["load_checkpoint"] = True
            cfg["params"]["load_path"] = resume_path
            cfg["params"]["config"]["num_actors"] = num_envs
            runner = Runner()
            runner.load(cfg)
            player: BasePlayer = runner.create_player()
            player.restore(resume_path)
            player.reset()
            players_by_name[object_name] = player
            print(f"[INFO] Loaded teacher player for {object_name}: {resume_path}", flush=True)

        stage_t = time.time()
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        for player in players_by_name.values():
            _ = player.get_batch_size(obs, 1)
            if player.is_rnn:
                player.init_rnn()
        print(f"[INFO] First reset complete in {time.time() - stage_t:.1f}s", flush=True)
        print(
            f"[INFO] Eval lift hold gate: {hold_steps} steps "
            f"(~{args_cli.eval_lift_hold_s:.3f}s target, dt={step_dt:.5f}s)",
            flush=True,
        )

        total_rollouts = int(args_cli.eval_episodes)
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

        success_rates: list[float] = []
        unsafe_rates: list[float] = []
        total_reason_counts = {name: 0 for name in UNSAFE_REASON_NAMES}
        per_object_lift_series = {
            object_name: [] for object_name in eval_object_names
        }
        per_object_unsafe_rate_series = {
            object_name: [] for object_name in eval_object_names
        }
        per_object_reason_counts_total = {
            object_name: {name: 0 for name in UNSAFE_REASON_NAMES}
            for object_name in eval_object_names
        }
        per_object_unsafe_total = {
            object_name: 0
            for object_name in eval_object_names
        }
        reason_to_idx = {name: idx for idx, name in enumerate(UNSAFE_REASON_NAMES)}
        total_unsafe_eps = 0

        for rollout_idx in range(total_rollouts):
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            eval_object_names, eval_object_idx = _resolve_eval_object_names_and_idx(
                eval_env, num_envs, args_cli.device, fallback_names=object_names
            )
            for player in players_by_name.values():
                if player.is_rnn:
                    player.init_rnn()

            dones = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            ever_lifted = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            ever_unsafe_terminated = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
            unsafe_reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=args_cli.device)
            steps = 0
            lift_hold_counts = torch.zeros((num_envs,), dtype=torch.long, device=args_cli.device)

            while steps < max_steps and not bool(dones.all().item()):
                # Use no_grad (not inference_mode) so env tensors remain mutable across reset calls.
                with torch.no_grad():
                    actions = None
                    for obj_idx, object_name in enumerate(eval_object_names):
                        obj_mask = eval_object_idx == obj_idx
                        if not bool(obj_mask.any().item()):
                            continue
                        player = players_by_name.get(object_name, None)
                        if player is None:
                            raise RuntimeError(
                                f"No loaded teacher player for object '{object_name}' "
                                f"(object index {obj_idx})."
                            )
                        obs_t = player.obs_to_torch(obs)
                        action_by_obj = player.get_action(obs_t, is_deterministic=False)
                        action_by_obj = torch.as_tensor(action_by_obj, device=args_cli.device)
                        if actions is None:
                            actions = torch.zeros_like(action_by_obj)
                        actions[obj_mask] = action_by_obj[obj_mask]
                    if actions is None:
                        raise RuntimeError(
                            "No teacher actions were produced. Check object-name alignment between "
                            "env object_names and teacher policy folders."
                        )
                    obs, _, env_dones, info = env.step(actions)

                env_dones = _as_bool_mask(env_dones, num_envs, args_cli.device)
                out_of_reach = _extract_out_of_reach_mask(eval_env, num_envs, args_cli.device)
                timed_out = _extract_timeout_mask(info, num_envs, args_cli.device)
                timed_out = timed_out | (env_dones & (~out_of_reach))
                dones = out_of_reach | timed_out
                reason_idx = classify_out_of_reach_reasons(
                    ov_env=eval_env,
                    out_of_reach=out_of_reach,
                    reason_names=UNSAFE_REASON_NAMES,
                    reason_to_idx=reason_to_idx,
                    device=args_cli.device,
                )
                ever_unsafe_terminated = ever_unsafe_terminated | out_of_reach
                classified_out_of_reach = reason_idx >= 0
                new_reason_mask = (unsafe_reason_idx < 0) & out_of_reach & classified_out_of_reach
                unsafe_reason_idx[new_reason_mask] = reason_idx[new_reason_mask]

                if bool(dones.any().item()):
                    done_indices = dones.nonzero(as_tuple=False).flatten()
                    for player in players_by_name.values():
                        if player.is_rnn and player.states is not None:
                            new_states = []
                            for s in player.states:
                                s_clone = s.clone()
                                s_clone[:, done_indices, :] = 0.0
                                new_states.append(s_clone)
                            player.states = new_states

                step_lift_success = _compute_lift_success(eval_env)
                active_envs = ~dones
                lift_hold_counts = torch.where(
                    active_envs & step_lift_success,
                    lift_hold_counts + 1,
                    torch.where(active_envs, torch.zeros_like(lift_hold_counts), lift_hold_counts),
                )
                hold_lift_success = lift_hold_counts >= hold_steps
                ever_lifted = ever_lifted | hold_lift_success
                steps += 1

            if steps >= max_steps:
                dones = torch.ones_like(dones)

            success_rates.append(ever_lifted.float().mean().item())
            unsafe_rates.append(ever_unsafe_terminated.float().mean().item())
            unsafe_count = int(ever_unsafe_terminated.sum().item())
            total_unsafe_eps += unsafe_count
            rollout_reason_counts = _reason_counts_from_episode(
                unsafe_reason_idx=unsafe_reason_idx,
                unsafe_mask=ever_unsafe_terminated,
                scope_label=f"teacher-pool rollout {rollout_idx + 1}",
            )
            for name in UNSAFE_REASON_NAMES:
                total_reason_counts[name] += rollout_reason_counts[name]

            for obj_idx, object_name in enumerate(eval_object_names):
                obj_mask = eval_object_idx == obj_idx
                if not bool(obj_mask.any().item()):
                    continue
                per_object_lift_series[object_name].append(
                    ever_lifted[obj_mask].float().mean().item()
                )
                per_object_unsafe_rate_series[object_name].append(
                    ever_unsafe_terminated[obj_mask].float().mean().item()
                )
                obj_unsafe_count = int(ever_unsafe_terminated[obj_mask].sum().item())
                per_object_unsafe_total[object_name] += obj_unsafe_count
                obj_reason_counts = _reason_counts_from_episode(
                    unsafe_reason_idx=unsafe_reason_idx[obj_mask],
                    unsafe_mask=ever_unsafe_terminated[obj_mask],
                    scope_label=(
                        f"teacher-pool rollout {rollout_idx + 1} object {object_name}"
                    ),
                )
                for reason_name in UNSAFE_REASON_NAMES:
                    per_object_reason_counts_total[object_name][reason_name] += obj_reason_counts[reason_name]

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

        avg_success = float(np.mean(success_rates)) if len(success_rates) > 0 else 0.0
        unsafe_episode_rate = float(np.mean(unsafe_rates)) if len(unsafe_rates) > 0 else 0.0
        eval_reason_pct = unsafe_reason_percentages_from_counts(
            total_reason_counts,
            total_unsafe_eps,
            UNSAFE_REASON_NAMES,
        )
        eval_per_object_metrics = {}
        for object_name in eval_object_names:
            obj_reason_pct = unsafe_reason_percentages_from_counts(
                per_object_reason_counts_total[object_name],
                int(per_object_unsafe_total[object_name]),
                UNSAFE_REASON_NAMES,
            )
            eval_per_object_metrics[object_name] = {
                "eval/lift_success": (
                    float(np.mean(per_object_lift_series[object_name]))
                    if len(per_object_lift_series[object_name]) > 0
                    else 0.0
                ),
                "eval/unsafe_episode_rate": (
                    float(np.mean(per_object_unsafe_rate_series[object_name]))
                    if len(per_object_unsafe_rate_series[object_name]) > 0
                    else 0.0
                ),
                "eval/out_of_reach_reason_pct": {
                    name: float(obj_reason_pct.get(name, 0.0))
                    for name in UNSAFE_REASON_NAMES
                },
            }
        total_done = int(total_rollouts * num_envs)
        print(
            f"[INFO] Eval complete in {time.time() - eval_start_t:.1f}s: "
            f"episodes={total_done}, lift_success={avg_success:.4f}, unsafe_rate={unsafe_episode_rate:.4f}",
            flush=True,
        )
        return (
            avg_success,
            unsafe_episode_rate,
            total_done,
            num_envs,
            eval_reason_pct,
            total_unsafe_eps,
            eval_per_object_metrics,
        )
    finally:
        env.close()
        _ENV_HOLDER["env"] = None


def main():
    if args_cli.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be > 0 for evaluation.")
    if (args_cli.teacher_policy_dir is None) != (args_cli.teacher_object_dir is None):
        raise ValueError("Provide both --teacher_policy_dir and --teacher_object_dir together.")
    if args_cli.teacher_policy_dir is not None and args_cli.checkpoint is not None:
        raise ValueError(
            "Choose one teacher eval mode: either --checkpoint (single) "
            "or --teacher_policy_dir/--teacher_object_dir (batch)."
        )
    if args_cli.teacher_policy_dir is None and args_cli.checkpoint is None:
        raise ValueError(
            "Teacher eval requires either --checkpoint (single mode) "
            "or --teacher_policy_dir and --teacher_object_dir (batch mode)."
        )
    _register_rlgames_env()
    tb_writer = None

    try:
        if args_cli.teacher_policy_dir is not None and args_cli.teacher_object_dir is not None:
            if args_cli.checkpoint is not None:
                raise ValueError("Do not pass --checkpoint when using teacher folder evaluation mode.")

            tb_writer, _ = _create_tb_writer(teacher_mode=True)
            object_names = _validate_teacher_policy_object_dirs(args_cli.teacher_policy_dir, args_cli.teacher_object_dir)
            teacher_start_t = time.time()
            print(
                f"[INFO] Teacher-folder evaluation started for {len(object_names)} objects in a single multi-object run.",
                flush=True,
            )
            objects_dir_name, temp_root = _prepare_teacher_multi_object_dir(
                args_cli.teacher_object_dir, object_names
            )
            print(
                f"[INFO] [TeacherEval] Prepared temporary multi-object override: {objects_dir_name}",
                flush=True,
            )
            try:
                (
                    avg_success,
                    avg_unsafe,
                    total_done,
                    num_envs,
                    avg_reason_percentages,
                    unsafe_eps,
                    per_object_metrics,
                ) = _run_eval_for_teacher_pool(
                    teacher_policy_dir=args_cli.teacher_policy_dir,
                    object_names=object_names,
                    objects_dir_override=objects_dir_name,
                )
            finally:
                if temp_root.exists():
                    shutil.rmtree(temp_root)
                    print(
                        f"[INFO] [TeacherEval] Removed temporary directory: {temp_root}",
                        flush=True,
                    )

            avg_reason_percentages_full = _reason_percentages_with_defaults(avg_reason_percentages)
            missing_metric_objects = [
                object_name for object_name in object_names
                if object_name not in per_object_metrics
            ]
            if len(missing_metric_objects) > 0:
                raise RuntimeError(
                    "Teacher pool eval did not produce metrics for all requested objects: "
                    f"{missing_metric_objects}"
                )
            flat_metrics: dict[str, object] = {}
            for object_idx, object_name in enumerate(object_names, start=1):
                object_metrics = per_object_metrics.get(object_name, {})
                reason_pct = _reason_percentages_with_defaults(
                    object_metrics.get("eval/out_of_reach_reason_pct", {})
                )
                lift_val = float(object_metrics.get("eval/lift_success", 0.0))
                unsafe_val = float(object_metrics.get("eval/unsafe_episode_rate", 0.0))
                flat_metrics[f"eval/lift_success/{object_name}"] = lift_val
                flat_metrics[f"eval/unsafe_episode_rate/{object_name}"] = unsafe_val
                flat_metrics[f"eval/out_of_reach_reason_pct/{object_name}"] = reason_pct
                tb_object_name = str(object_name).replace("/", "_")
                _tb_add_scalar(tb_writer, f"eval/{tb_object_name}/lift_success", lift_val, object_idx)
                _tb_add_scalar(tb_writer, f"eval/{tb_object_name}/unsafe_episode_rate", unsafe_val, object_idx)
                _tb_log_reason_percentages(
                    tb_writer,
                    f"eval/{tb_object_name}",
                    reason_pct,
                    object_idx,
                )
                print(
                    f"eval/lift_success/{object_name}: {lift_val:.4f} "
                    f"| eval/unsafe_episode_rate/{object_name}: {unsafe_val:.4f} "
                    f"| eval/out_of_reach_reason_pct/{object_name}: {_format_reason_percentages(reason_pct)}",
                    flush=True,
                )

            print(f"eval/lift_success_avg: {avg_success:.4f} (objects: {len(object_names)})")
            print(f"eval/unsafe_episode_rate_avg: {avg_unsafe:.4f} (objects: {len(object_names)})")
            print(f"eval/out_of_reach_reason_pct_avg: {_format_reason_percentages(avg_reason_percentages_full)}")
            flat_metrics["eval/lift_success_avg"] = float(avg_success)
            flat_metrics["eval/unsafe_episode_rate_avg"] = float(avg_unsafe)
            flat_metrics["eval/out_of_reach_reason_pct_avg"] = avg_reason_percentages_full
            _tb_add_scalar(tb_writer, "eval/avg/lift_success", float(avg_success), len(object_names))
            _tb_add_scalar(tb_writer, "eval/avg/unsafe_episode_rate", float(avg_unsafe), len(object_names))
            _tb_log_reason_percentages(
                tb_writer,
                "eval/avg",
                avg_reason_percentages_full,
                len(object_names),
            )
            metrics_payload = {
                "mode": "teacher_folder",
                "task": args_cli.task,
                "teacher_policy_dir": str(pathlib.Path(args_cli.teacher_policy_dir).expanduser().resolve()),
                "teacher_object_dir": str(pathlib.Path(args_cli.teacher_object_dir).expanduser().resolve()),
                "eval_episodes": int(args_cli.eval_episodes),
                "objects": per_object_metrics,
                "averages": {
                    "eval/lift_success_avg": float(avg_success),
                    "eval/unsafe_episode_rate_avg": float(avg_unsafe),
                    "eval/out_of_reach_reason_pct_avg": avg_reason_percentages_full,
                },
                "flat": flat_metrics,
                "total_episodes": int(total_done),
                "num_envs": int(num_envs),
                "unsafe_episodes": int(unsafe_eps),
            }
            _save_metrics_json(metrics_payload, teacher_mode=True)
            print(
                f"[INFO] Teacher-folder evaluation finished in {time.time() - teacher_start_t:.1f}s",
                flush=True,
            )
            return

        tb_writer, _ = _create_tb_writer(teacher_mode=False)
        avg_success, unsafe_episode_rate, total_done, num_envs, reason_percentages, unsafe_eps = _run_eval_for_checkpoint(
            checkpoint_path=args_cli.checkpoint
        )
        reason_percentages_full = _reason_percentages_with_defaults(reason_percentages)
        print(
            f"eval/lift_success: {avg_success:.4f} | "
            f"eval/unsafe_episode_rate: {unsafe_episode_rate:.4f} | "
            f"eval/out_of_reach_reason_pct: {_format_reason_percentages(reason_percentages_full)} "
            f"(total episodes: {total_done}, envs: {num_envs})"
        )
        _tb_add_scalar(tb_writer, "eval/lift_success", float(avg_success), 0)
        _tb_add_scalar(tb_writer, "eval/unsafe_episode_rate", float(unsafe_episode_rate), 0)
        _tb_log_reason_percentages(
            tb_writer,
            "eval",
            reason_percentages_full,
            0,
        )
        metrics_payload = {
            "mode": "single_checkpoint",
            "task": args_cli.task,
            "checkpoint": args_cli.checkpoint,
            "eval_episodes": int(args_cli.eval_episodes),
            "metrics": {
                "eval/lift_success": float(avg_success),
                "eval/unsafe_episode_rate": float(unsafe_episode_rate),
                "eval/out_of_reach_reason_pct": reason_percentages_full,
            },
            "total_episodes": int(total_done),
            "num_envs": int(num_envs),
            "unsafe_episodes": int(unsafe_eps),
        }
        _save_metrics_json(metrics_payload, teacher_mode=False)
    finally:
        if tb_writer is not None:
            try:
                if not _TB_STATE["disabled"]:
                    tb_writer.flush()
            except Exception as exc:
                print(f"[WARN] TensorBoard flush failed: {exc}", flush=True)
            try:
                tb_writer.close()
            except Exception as exc:
                print(f"[WARN] TensorBoard close failed: {exc}", flush=True)
            print("[INFO] TensorBoard writer closed.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
