"""Script to perform student-teacher distillation"""

import argparse
import copy
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

DISTILLATION_VARIANTS = {
    "vanilla_dagger": {
        "pipeline": "safedagger",
        "unsafe_mode": "none",
        "description": "Vanilla DAgger: student rollout with teacher labels.",
    },
    "vanilla_safedagger": {
        "pipeline": "safedagger",
        "unsafe_mode": "l2",
        "description": "Vanilla SafeDAgger: teacher takeover on action disagreement.",
    },
    "dexsafedagger": {
        "pipeline": "both",
        "unsafe_mode": "failure_predictor",
        "description": "DexSafeDagger: warm-start plus disagreement/success-value intervention.",
    },
    "vc_dexsafedagger": {
        "pipeline": "both",
        "unsafe_mode": "failure_predictor",
        "description": (
            "VC-DexSafeDagger: SafeDAgger disagreement plus success-value critic "
            "and VLM threshold tuner."
        ),
    },
    "dexsafedaggerUltra": {
        "pipeline": "both",
        "unsafe_mode": "failure_predictor",
        "description": (
            "Legacy alias for VC-DexSafeDagger with the VLM threshold tuner."
        ),
        "experimental": True,
    },
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=100000, help="Total distillation iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")
parser.add_argument(
    "--teacher_object_map",
    type=str,
    default=None,
    help=(
        "Optional JSON/YAML object->teacher mapping (file path or inline JSON). "
        "Map values can be subdir names under --teacher root or explicit .pth/.pt paths."
    ),
)
parser.add_argument("--student", type=str, default=None, help="Student checkpoint to use")
parser.add_argument("--play_policy", type=bool, default=False, help="Play a distilled policy.")
parser.add_argument(
    "--variant",
    type=str,
    default="dexsafedagger",
    choices=sorted(DISTILLATION_VARIANTS.keys()),
    help=(
        "Distillation variant: vanilla_dagger, vanilla_safedagger, "
        "dexsafedagger, vc_dexsafedagger, or the legacy dexsafedaggerUltra alias."
    ),
)
parser.add_argument(
    "--pipeline",
    type=str,
    default=None,
    choices=["warmstart", "safedagger", "both"],
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--warm_start_collect_steps",
    type=int,
    default=None,
    help="Override warm-start rollout collection steps.",
)
parser.add_argument(
    "--warm_start_predictor_train_steps",
    type=int,
    default=None,
    help="Override warm-start failure-predictor fit steps.",
)
parser.add_argument(
    "--warm_start_no_match_predictor_buffer_size",
    action="store_true",
    default=False,
    help=(
        "Do not increase warm-start collect_steps to fill the failure predictor replay buffer. "
        "Useful for small smoke runs."
    ),
)
parser.add_argument("--data_aug", action="store_true", default=False, help="Whether to use data augmentation for student")
parser.add_argument(
    "--eval_every",
    type=int,
    default=2500,
    help="Run student-only eval every N iterations (0 disables).",
)
parser.add_argument(
    "--eval_num_episodes",
    type=int,
    default=3,
    help="Number of episodes per evaluation run.",
)
parser.add_argument(
    "--eval_num_envs",
    type=int,
    default=None,
    help="Number of environments for evaluation (defaults to training num_envs).",
)
parser.add_argument(
    "--eval_max_steps",
    type=int,
    default=None,
    help="Max steps per eval episode before moving on (default: env limit).",
)
parser.add_argument(
    "--eval_lift_hold_s",
    type=float,
    default=None,
    help=(
        "Inline eval lift gate hold duration in seconds. "
        "Object must stay above lift threshold (with grasp/contact) for this long."
    ),
)
parser.add_argument(
    "--eval_objects_dir",
    type=str,
    default=None,
    help=(
        "Object assets folder for inline evaluation. "
        "If unset, eval uses the training env/object folder."
    ),
)
parser.add_argument(
    "--unsafe_mode",
    type=str,
    default=None,
    choices=["none", "l2", "failure_predictor"],
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--unsafe_l2_threshold",
    type=float,
    default=None,
    help=(
        "Override the student-teacher weighted-L2 threshold used by l2 SafeDagger "
        "intervention."
    ),
)
parser.add_argument(
    "--failure_predictor_warm_start_model_path",
    type=str,
    default=None,
    help=(
        "Optional checkpoint path for failure predictor warm-start model. "
        "Warmstart/both save predictor here; safedagger mode loads predictor from here."
    ),
)
parser.add_argument(
    "--switch_back_min_teacher_steps",
    type=int,
    default=None,
    help=(
        "Minimum teacher takeover steps after an unsafe trigger before switch-back checks. "
        "Set 0 to disable hold."
    ),
)
parser.add_argument(
    "--vlm_threshold_advisor",
    action="store_true",
    default=False,
    help="Enable VLM threshold advisor for L2/success thresholds.",
)
parser.add_argument(
    "--vlm_threshold_advisor_mode",
    choices=["shadow", "active"],
    default=None,
    help="VLM threshold advisor mode: shadow logs recommendations; active applies them.",
)
parser.add_argument("--vlm_threshold_update_interval_steps", type=int, default=None)
parser.add_argument("--vlm_threshold_warmup_steps", type=int, default=None)
parser.add_argument("--vlm_threshold_min_samples", type=int, default=None)
parser.add_argument("--vlm_threshold_model", type=str, default=None)
parser.add_argument("--vlm_threshold_base_url", type=str, default=None)
parser.add_argument("--vlm_threshold_api_key_env", type=str, default=None)
parser.add_argument("--vlm_threshold_max_tokens", type=int, default=None)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

variant_cfg = DISTILLATION_VARIANTS[args_cli.variant]
if args_cli.pipeline is None:
    args_cli.pipeline = variant_cfg["pipeline"]
if args_cli.unsafe_mode is None:
    args_cli.unsafe_mode = variant_cfg["unsafe_mode"]

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
import os
import yaml
from datetime import datetime
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


from dexsafedagger_lab.distillation.core.distillation_safedagger import SafeDagger
import dexsafedagger_lab.tasks.tg2_inspirehand.gym_setup

from dexsafedagger_lab.distillation.models.a2c_stereo_transformer import (
    A2CBuilder as A2CStereoTransformerBuilder,
)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _load_teacher_object_map(raw_value: str | None):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if len(text) == 0:
        return None

    loaded = None
    candidate_path = pathlib.Path(text).expanduser()
    if candidate_path.is_file():
        with candidate_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
    else:
        try:
            loaded = json.loads(text)
        except Exception as exc:
            raise ValueError(
                "--teacher_object_map must be a valid JSON string or an existing JSON/YAML file path."
            ) from exc

    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        raise ValueError("--teacher_object_map must parse to a dictionary.")

    normalized = {}
    for key, value in loaded.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized if len(normalized) > 0 else None


def _reason_prop_to_pct(reason_prop: dict, unsafe_rate: float, reason_names: list[str]) -> dict[str, float]:
    if unsafe_rate <= 0.0:
        return {name: 0.0 for name in reason_names}
    return {
        name: (100.0 * float(reason_prop.get(name, 0.0)) / float(unsafe_rate))
        for name in reason_names
    }


def _save_final_eval_json(
    output_path: pathlib.Path,
    *,
    task: str,
    distillation_variant: str,
    final_checkpoint: str,
    final_eval_episodes: int,
    eval_max_steps,
    eval_lift_hold_s: float,
    metrics: dict,
) -> None:
    payload = {
        "mode": "runtime_final_eval",
        "task": task,
        "distillation_variant": distillation_variant,
        "final_checkpoint": final_checkpoint,
        "final_eval_episodes": int(final_eval_episodes),
        "eval_max_steps": int(eval_max_steps) if eval_max_steps is not None else None,
        "eval_lift_hold_s": float(eval_lift_hold_s),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)
    print(f"[INFO] Saved final runtime eval JSON to: {output_path}", flush=True)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """ Performs distillation. """
    print(
        f"CLI eval_every={args_cli.eval_every}, eval_num_episodes={args_cli.eval_num_episodes}",
        flush=True,
    )
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env = None
    eval_env = None
    dagger = None
    try:
        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        ov_env = env.env

        if args_cli.eval_every is not None and args_cli.eval_every > 0:
            if args_cli.eval_num_envs is not None and args_cli.eval_num_envs != env_cfg.scene.num_envs:
                print(
                    "Inline eval uses the training environment; ignoring --eval_num_envs "
                    f"({args_cli.eval_num_envs} != {env_cfg.scene.num_envs})."
                )
            if args_cli.eval_objects_dir:
                eval_env_cfg = copy.deepcopy(env_cfg)
                eval_env_cfg.scene.num_envs = env_cfg.scene.num_envs
                eval_env_cfg.objects_dir = args_cli.eval_objects_dir
                if (
                    hasattr(eval_env_cfg, "valid_objects_dir")
                    and isinstance(eval_env_cfg.valid_objects_dir, list)
                    and args_cli.eval_objects_dir not in eval_env_cfg.valid_objects_dir
                ):
                    eval_env_cfg.valid_objects_dir.append(args_cli.eval_objects_dir)
                eval_env = gym.make(args_cli.task, cfg=eval_env_cfg, render_mode=None)
                print(
                    f"Inline eval objects_dir={args_cli.eval_objects_dir} "
                    f"(train objects_dir={env_cfg.objects_dir})",
                    flush=True,
                )

        distillation_dir = pathlib.Path(__file__).resolve().parents[1]
        parent_path = str(REPO_ROOT)
        agent_cfg_folder = "dexsafedagger_lab/tasks/tg2_inspirehand/agents"

        if not ov_env.simulate_stereo:
            raise ValueError("distillation only supports stereo transformer policies.")
        student_cfg = os.path.join(
            parent_path,
            agent_cfg_folder,
            "rl_games_ppo_stereo_transformer.yaml",
        )

        teacher_cfg = os.path.join(
            parent_path,
            agent_cfg_folder,
            "rl_games_ppo_lstm_cfg.yaml"
        )

        num_student_obs = ov_env.num_observations
        num_teacher_obs = ov_env.num_teacher_observations
        num_actions = ov_env.num_actions
        # Determine checkpoints
        teacher_ckpt = None
        if not args_cli.play_policy:
            if args_cli.teacher is not None:
                teacher_ckpt = os.path.join(parent_path, "pretrained_ckpts", args_cli.teacher)
            else:
                teacher_ckpt = os.path.join(parent_path, "pretrained_ckpts/new_teacher.pth")
        student_ckpt = None
        if args_cli.student is not None:
            student_ckpt = args_cli.student
            if not os.path.isabs(student_ckpt):
                student_ckpt = os.path.join(parent_path, "pretrained_ckpts", student_ckpt)
        teacher_object_map = _load_teacher_object_map(args_cli.teacher_object_map)

        train_dir = str(distillation_dir / "runs")
        variant_tag = str(args_cli.variant).lower()
        experiment_name = (
            f"dexsafedagger-tg2-inspirehand-{variant_tag}"
            + datetime.now().strftime("_%d-%H-%M-%S")
        )
        experiment_dir = os.path.join(train_dir, experiment_name)
        nn_dir = os.path.join(experiment_dir, "nn")
        summaries_dir = os.path.join(experiment_dir, "summaries")
        params_dir = os.path.join(experiment_dir, "params")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(params_dir, exist_ok=True)
        print(f"[INFO] Distillation logs in directory: {os.path.abspath(experiment_dir)}")

        with open(student_cfg, "r") as f:
            student_cfg_yaml = yaml.safe_load(f) or {}
        with open(teacher_cfg, "r") as f:
            teacher_cfg_yaml = yaml.safe_load(f) or {}
        distill_cfg = (
            student_cfg_yaml.get("params", {}).get("distillation", {}) if isinstance(student_cfg_yaml, dict) else {}
        )

        dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            "obs_type": "policy",
            "data_aug": args_cli.data_aug,
        },
        "teacher": {
            "cfg": teacher_cfg,
            "ckpt": teacher_ckpt,
            "obs_type": "expert_policy",
            "object_ckpt_map": teacher_object_map,
        },
        "imitation_loss_type": distill_cfg.get("imitation_loss_type", "l2"),
        "unsafe_mode": distill_cfg.get("unsafe_mode", "l2"),
        "unsafe_l2_threshold": distill_cfg.get("unsafe_l2_threshold", 0.5),
        "switch_back_min_teacher_steps": distill_cfg.get("switch_back_min_teacher_steps", 10),
        "failure_predictor": {
            "enabled": False,
            "type": "critic",
            "obs_key": "predictor_transition",
            "hidden_sizes": [256, 128],
            "lr": 1e-3,
            "dropout": 0.0,
            "buffer_size": 100_000,
            "batch_size": 128,
            "min_samples": 10_000,
            "update_interval": 1_000,
            "online_train_step_calls": 1,
            "unsafe_enable_after_steps": 0,
            "horizon_steps": 10,
            "success_threshold": 0.5,
            "success_key": "lift_success",
            "output_temperature": 2.0,
            "include_current_step": False,
            "pos_weight": None,
            "pos_fraction": 0.1,
            "device": "cpu",
            "warm_start_model_path": None,
        },
        "vlm_threshold_advisor": {
            **{
                "enabled": False,
                "mode": "shadow",  # shadow logs recommendations; active applies clamped/smoothed thresholds
                "base_url": None,
                "model": None,
                "api_key_env": None,
                "update_interval_steps": 10000,
                "warmup_steps": 2000,
                "min_samples": 1,
                "smoothing": 0.1,
                "l2_min_scale": 0.5,
                "l2_max_scale": 1.5,
                "success_min_scale": 0.5,
                "success_max_scale": 1.5,
                "max_tokens": 1024,
                "timeout": 90.0,
                "visual_buffer_enabled": True,
                "visual_buffer_size": 64,
                "visual_capture_interval_steps": 20,
                "visual_captures_per_step": 2,
                "visual_samples_per_update": 8,
                "visual_image_key": "img_left",
                "visual_max_edge": 256,
                "visual_jpeg_quality": 70,
                "visual_detail": "low",
                "report_enabled": True,
                "report_dir": None,
            },
            **(
                dict(distill_cfg.get("vlm_threshold_advisor", {}))
                if isinstance(distill_cfg.get("vlm_threshold_advisor", None), dict)
                else {}
            ),
        },
        "warm_start": (
            dict(distill_cfg.get("warm_start", {}))
            if isinstance(distill_cfg.get("warm_start", None), dict)
            else {}
        ),
        "play_policy": args_cli.play_policy,
        "eval_every": args_cli.eval_every,
        "eval_num_episodes": args_cli.eval_num_episodes,
        "eval_max_steps": args_cli.eval_max_steps,
        "eval_lift_hold_s": distill_cfg.get("eval_lift_hold_s", 0.5),
        "num_iters": args_cli.max_iterations,
    }
        if isinstance(distill_cfg.get("failure_predictor", None), dict):
            dagger_config["failure_predictor"].update(distill_cfg["failure_predictor"])
        if args_cli.vlm_threshold_advisor:
            dagger_config["vlm_threshold_advisor"]["enabled"] = True
        if args_cli.variant in {"vc_dexsafedagger", "dexsafedaggerUltra"}:
            dagger_config["vlm_threshold_advisor"]["enabled"] = True
            dagger_config["vlm_threshold_advisor"]["mode"] = "active"
        if args_cli.vlm_threshold_advisor_mode is not None:
            dagger_config["vlm_threshold_advisor"]["mode"] = args_cli.vlm_threshold_advisor_mode
        if args_cli.vlm_threshold_update_interval_steps is not None:
            dagger_config["vlm_threshold_advisor"]["update_interval_steps"] = (
                args_cli.vlm_threshold_update_interval_steps
            )
        if args_cli.vlm_threshold_warmup_steps is not None:
            dagger_config["vlm_threshold_advisor"]["warmup_steps"] = args_cli.vlm_threshold_warmup_steps
        if args_cli.vlm_threshold_min_samples is not None:
            dagger_config["vlm_threshold_advisor"]["min_samples"] = args_cli.vlm_threshold_min_samples
        if args_cli.vlm_threshold_model is not None:
            dagger_config["vlm_threshold_advisor"]["model"] = args_cli.vlm_threshold_model
        if args_cli.vlm_threshold_base_url is not None:
            dagger_config["vlm_threshold_advisor"]["base_url"] = args_cli.vlm_threshold_base_url
        if args_cli.vlm_threshold_api_key_env is not None:
            dagger_config["vlm_threshold_advisor"]["api_key_env"] = args_cli.vlm_threshold_api_key_env
        if args_cli.vlm_threshold_max_tokens is not None:
            dagger_config["vlm_threshold_advisor"]["max_tokens"] = args_cli.vlm_threshold_max_tokens
        if args_cli.failure_predictor_warm_start_model_path is not None:
            dagger_config["failure_predictor"]["warm_start_model_path"] = (
                args_cli.failure_predictor_warm_start_model_path
            )
        if args_cli.unsafe_mode is not None:
            dagger_config["unsafe_mode"] = args_cli.unsafe_mode
        if args_cli.unsafe_l2_threshold is not None:
            dagger_config["unsafe_l2_threshold"] = float(args_cli.unsafe_l2_threshold)
        mode = dagger_config["unsafe_mode"]
        vlm_advisor_enabled = bool(dagger_config["vlm_threshold_advisor"].get("enabled", False))
        if mode == "failure_predictor" or vlm_advisor_enabled:
            dagger_config["failure_predictor"]["enabled"] = True
        elif mode in {"none", "l2"}:
            dagger_config["failure_predictor"]["enabled"] = False
        if args_cli.switch_back_min_teacher_steps is not None:
            dagger_config["switch_back_min_teacher_steps"] = int(args_cli.switch_back_min_teacher_steps)
        if args_cli.eval_lift_hold_s is not None:
            dagger_config["eval_lift_hold_s"] = args_cli.eval_lift_hold_s
        if args_cli.warm_start_collect_steps is not None:
            dagger_config["warm_start"]["collect_steps"] = int(args_cli.warm_start_collect_steps)
        if args_cli.warm_start_predictor_train_steps is not None:
            dagger_config["warm_start"]["predictor_train_steps"] = int(args_cli.warm_start_predictor_train_steps)
        if args_cli.warm_start_no_match_predictor_buffer_size:
            dagger_config["warm_start"]["match_predictor_buffer_size"] = False

    # Save run metadata/config snapshots in the same style as rl_games train logs.
        run_meta = {
        "timestamp": datetime.now().isoformat(),
        "entrypoint": os.path.abspath(__file__),
        "cwd": os.getcwd(),
        "task": args_cli.task,
        "distillation_variant": args_cli.variant,
        "variant_description": DISTILLATION_VARIANTS[args_cli.variant]["description"],
        "resolved_pipeline": args_cli.pipeline,
        "resolved_unsafe_mode": args_cli.unsafe_mode,
        "experiment_dir": os.path.abspath(experiment_dir),
        "student_cfg_path": student_cfg,
        "teacher_cfg_path": teacher_cfg,
        "args_cli": vars(args_cli),
        "hydra_args": hydra_args,
    }
        dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
        dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)
        dump_yaml(os.path.join(params_dir, "student_agent.yaml"), student_cfg_yaml)
        dump_pickle(os.path.join(params_dir, "student_agent.pkl"), student_cfg_yaml)
        dump_yaml(os.path.join(params_dir, "teacher_agent.yaml"), teacher_cfg_yaml)
        dump_pickle(os.path.join(params_dir, "teacher_agent.pkl"), teacher_cfg_yaml)
        dump_yaml(os.path.join(params_dir, "dagger.yaml"), dagger_config)
        dump_pickle(os.path.join(params_dir, "dagger.pkl"), dagger_config)
        dump_yaml(os.path.join(params_dir, "run_meta.yaml"), run_meta)
        dump_pickle(os.path.join(params_dir, "run_meta.pkl"), run_meta)
        with open(os.path.join(params_dir, "hydra_overrides.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(hydra_args))

        model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

        dagger = SafeDagger(
            env,
            dagger_config,
            summaries_dir=summaries_dir,
            nn_dir=nn_dir,
            eval_env=eval_env,
        )
        reached_iters = dagger.run_pipeline(args_cli.pipeline)
        if args_cli.pipeline == "warmstart":
            final_ckpt = os.path.join(dagger.nn_dir, "dexsafedagger_student_after_warmstart.pth")
        else:
            final_ckpt = os.path.join(dagger.nn_dir, "dexsafedagger_student_safe_dagger.pth")
        if getattr(dagger, "rank", 0) == 0:
            dagger.save(final_ckpt)
            print(
                f"[INFO] Variant '{args_cli.variant}' finished at iter {reached_iters} / {dagger.num_iters}.",
                flush=True,
            )
            last_eval_snapshot = getattr(dagger, "last_eval_snapshot", None)
            if isinstance(last_eval_snapshot, dict):
                print(
                    "[INFO] Exporting final eval JSON from last TensorBoard eval point (no extra eval run).",
                    flush=True,
                )
                metrics = dict(last_eval_snapshot.get("metrics", {}))
                reason_prop = dict(metrics.get("eval/unsafe_reason_prop", {}))
                reason_names = [str(name) for name in reason_prop.keys()]
                unsafe_rate = float(metrics.get("eval/unsafe_episode_rate", 0.0))
                metrics["eval/out_of_reach_reason_pct"] = _reason_prop_to_pct(
                    reason_prop, unsafe_rate, reason_names
                )
                per_object_metrics_raw = dict(last_eval_snapshot.get("per_object_metrics", {}))
                per_object_metrics_with_pct = {}
                for object_name, object_metrics in per_object_metrics_raw.items():
                    obj_metrics = dict(object_metrics)
                    obj_unsafe_rate = float(obj_metrics.get("unsafe_episode_rate", 0.0))
                    obj_reason_prop = dict(obj_metrics.get("unsafe_reason_prop", {}))
                    obj_reason_pct = _reason_prop_to_pct(obj_reason_prop, obj_unsafe_rate, reason_names)
                    obj_metrics["unsafe_reason_pct"] = obj_reason_pct
                    per_object_metrics_with_pct[str(object_name)] = obj_metrics
                metrics["per_object_metrics"] = per_object_metrics_with_pct

                final_eval_json_path = pathlib.Path(final_ckpt).with_name(
                    f"{pathlib.Path(final_ckpt).stem}_final_eval.json"
                )
                _save_final_eval_json(
                    final_eval_json_path,
                    task=str(args_cli.task),
                    distillation_variant=str(args_cli.variant),
                    final_checkpoint=str(final_ckpt),
                    final_eval_episodes=int(last_eval_snapshot.get("eval_num_episodes", 0)),
                    eval_max_steps=last_eval_snapshot.get("eval_max_steps", None),
                    eval_lift_hold_s=float(last_eval_snapshot.get("eval_lift_hold_s", 0.0)),
                    metrics=metrics,
                )
            else:
                print(
                    "[INFO] No eval point was logged to TensorBoard; skipping final eval JSON export.",
                    flush=True,
                )
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
