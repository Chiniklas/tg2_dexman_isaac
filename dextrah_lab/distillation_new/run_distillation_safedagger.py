"""Script to perform student-teacher distillation"""

import argparse
import copy
import sys

from isaaclab.app import AppLauncher

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
parser.add_argument("--student", type=str, default=None, help="Student checkpoint to use")
parser.add_argument("--play_policy", type=bool, default=False, help="Play a distilled policy.")
parser.add_argument(
    "--pipeline",
    type=str,
    default="safedagger",
    choices=["warmstart", "safedagger", "both"],
    help=(
        "Training pipeline stage to run: "
        "warmstart = offline bootstrap only, "
        "safedagger = teacher-intervention training only, "
        "both = warmstart + safedagger."
    ),
)
parser.add_argument("--data_aug", action="store_true", default=False, help="Whether to use data augmentation for student")
parser.add_argument(
    "--eval_every",
    type=int,
    default=0,
    help="Run student-only eval every N iterations (0 disables).",
)
parser.add_argument(
    "--eval_num_episodes",
    type=int,
    default=5,
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
    choices=["none", "l2", "ood", "failure_predictor"],
    help="Unsafe gating mode override.",
)
parser.add_argument(
    "--ood_type",
    type=str,
    default=None,
    choices=["gaussian", "pca", "mlp"],
    help="OOD classifier type override (used when unsafe_mode=ood).",
)
parser.add_argument(
    "--failure_predictor_type",
    type=str,
    default=None,
    choices=["critic", "legacy"],
    help="Failure predictor type override (used when unsafe_mode=failure_predictor).",
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


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
import pathlib

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


from distillation_safedagger import SafeDagger
import dextrah_lab.tasks.tg2_inspirehand.gym_setup

from dextrah_lab.distillation_new.a2c_stereo_transformer import (
    A2CBuilder as A2CStereoTransformerBuilder,
)


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

        parent_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
        agent_cfg_folder = "dextrah_lab/tasks/tg2_inspirehand/agents"

        if not ov_env.simulate_stereo:
            raise ValueError("distillation_new only supports stereo transformer policies.")
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

        train_dir = "runs"
        pipeline_tag = str(args_cli.pipeline).lower()
        experiment_name = (
            f"dextrah-tg2-inspirehand-{pipeline_tag}"
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
        },
        "imitation_loss_type": distill_cfg.get("imitation_loss_type", "l2"),
        "unsafe_mode": distill_cfg.get("unsafe_mode", "l2"),
        "unsafe_l2_threshold": distill_cfg.get("unsafe_l2_threshold", 0.5),
        "switch_back_min_teacher_steps": distill_cfg.get("switch_back_min_teacher_steps", 10),
        "failure_predictor": {
            "enabled": False,
            "obs_key": "ood_policy_embed",
            "hidden_sizes": [256, 128],
            "lr": 1e-3,
            "dropout": 0.0,
            "buffer_size": 100_000,
            "batch_size": 128,
            "min_samples": 10_000,
            "update_interval": 1_000,
            "train_steps": 1,
            "horizon_steps": 10,
            "failure_threshold": 0.5,
            "success_threshold": 0.5,
            "success_key": "lift_success",
            "include_current_step": False,
            "pos_weight": None,
            "pos_fraction": 0.1,
            "device": "cpu",
            "warm_start_model_path": None,
        },
        "ood": {
            "enabled": False,
            "type": "gaussian", # gaussian # pca # mlp
            "obs_key": "ood_policy_embed",
            "min_samples": 10_000,
            "update_interval": 500,
            "threshold_quantile": 0.80,
            "diag_eps": 1e-4,
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
        if isinstance(distill_cfg.get("ood", None), dict):
            dagger_config["ood"].update(distill_cfg["ood"])
        if args_cli.ood_type is not None:
            dagger_config["ood"]["type"] = args_cli.ood_type
        if args_cli.failure_predictor_type is not None:
            dagger_config["failure_predictor"]["type"] = args_cli.failure_predictor_type
        if args_cli.failure_predictor_warm_start_model_path is not None:
            dagger_config["failure_predictor"]["warm_start_model_path"] = (
                args_cli.failure_predictor_warm_start_model_path
            )
        if args_cli.unsafe_mode is not None:
            dagger_config["unsafe_mode"] = args_cli.unsafe_mode
        mode = dagger_config["unsafe_mode"]
        if mode == "ood":
            dagger_config["ood"]["enabled"] = True
            dagger_config["failure_predictor"]["enabled"] = False
        elif mode == "failure_predictor":
            dagger_config["failure_predictor"]["enabled"] = True
            dagger_config["ood"]["enabled"] = False
        elif mode in {"none", "l2"}:
            dagger_config["ood"]["enabled"] = False
            dagger_config["failure_predictor"]["enabled"] = False
        if args_cli.switch_back_min_teacher_steps is not None:
            dagger_config["switch_back_min_teacher_steps"] = int(args_cli.switch_back_min_teacher_steps)
        if args_cli.eval_lift_hold_s is not None:
            dagger_config["eval_lift_hold_s"] = args_cli.eval_lift_hold_s

    # Save run metadata/config snapshots in the same style as rl_games train logs.
        run_meta = {
        "timestamp": datetime.now().isoformat(),
        "entrypoint": os.path.abspath(__file__),
        "cwd": os.getcwd(),
        "task": args_cli.task,
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

        dagger = SafeDagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir, eval_env=eval_env)
        reached_iters = dagger.run_pipeline(args_cli.pipeline)
        if args_cli.pipeline == "warmstart":
            final_ckpt = os.path.join(dagger.nn_dir, "dextrah_student_after_warmstart.pth")
        else:
            final_ckpt = os.path.join(dagger.nn_dir, "dextrah_student_safe_dagger.pth")
        if getattr(dagger, "rank", 0) == 0:
            dagger.save(final_ckpt)
            print(
                f"[INFO] Pipeline '{args_cli.pipeline}' finished at iter {reached_iters} / {dagger.num_iters}.",
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
