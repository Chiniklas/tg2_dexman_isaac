"""Script to perform student-teacher distillation"""

import argparse
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
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--teacher", type=str, default=None, help="Teacher checkpoint to use")
parser.add_argument("--student", type=str, default=None, help="Student checkpoint to use")
parser.add_argument("--play_policy", type=bool, default=False, help="Play a distilled policy.")
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
    "--imitation_target",
    type=str,
    default=None,
    choices=["action_distribution", "sampled_action"],
    help="Imitation target type (overrides imitation_loss_type when paired with --loss_type).",
)
parser.add_argument(
    "--loss_type",
    type=str,
    default=None,
    choices=["kl", "nll", "l2", "mse"],
    help="Imitation loss type (used with --imitation_target).",
)
parser.add_argument(
    "--unsafe_mode",
    type=str,
    default=None,
    choices=["none", "l2", "ood"],
    help="Unsafe gating mode override.",
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env

    eval_env = None
    if args_cli.eval_every is not None and args_cli.eval_every > 0:
        if args_cli.eval_num_envs is not None and args_cli.eval_num_envs != env_cfg.scene.num_envs:
            print(
                "Inline eval uses the training environment; ignoring --eval_num_envs "
                f"({args_cli.eval_num_envs} != {env_cfg.scene.num_envs})."
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
    experiment_name = (
        "dextrah-tg2-inspirehand"
        + datetime.now().strftime("_%d-%H-%M-%S")
    )
    experiment_dir = os.path.join(train_dir, experiment_name)
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    with open(student_cfg, "r") as f:
        student_cfg_yaml = yaml.safe_load(f) or {}
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
        "imitation_target": distill_cfg.get("imitation_target", None),
        "loss_type": distill_cfg.get("loss_type", None),
        "unsafe_mode": distill_cfg.get("unsafe_mode", "l2"),
        "unsafe_l2_threshold": distill_cfg.get("unsafe_l2_threshold", 0.5),
        "ood": {
            "enabled": False,
            "type": "gaussian",
            "obs_key": "ood_policy_embed",
            "min_samples": 5_000,
            "update_interval": 1_000,
            "threshold_quantile": 0.99,
            "diag_eps": 1e-4,
        },
        "play_policy": args_cli.play_policy,
        "eval_every": args_cli.eval_every,
        "eval_num_episodes": args_cli.eval_num_episodes,
        "eval_max_steps": args_cli.eval_max_steps,
    }
    if isinstance(distill_cfg.get("ood", None), dict):
        dagger_config["ood"].update(distill_cfg["ood"])
    if args_cli.imitation_target is not None:
        dagger_config["imitation_target"] = args_cli.imitation_target
    if args_cli.loss_type is not None:
        dagger_config["loss_type"] = args_cli.loss_type
    if args_cli.unsafe_mode is not None:
        dagger_config["unsafe_mode"] = args_cli.unsafe_mode
        if args_cli.unsafe_mode == "ood":
            dagger_config["ood"]["enabled"] = True
        elif args_cli.unsafe_mode in {"none", "l2"}:
            dagger_config["ood"]["enabled"] = False

    model_builder.register_network("a2c_stereo_transformer", A2CStereoTransformerBuilder)

    dagger = SafeDagger(env, dagger_config, summaries_dir=summaries_dir, nn_dir=nn_dir, eval_env=eval_env)
    dagger.distill()
    final_ckpt = os.path.join(dagger.nn_dir, "dextrah_student_safe_dagger.pth")
    if getattr(dagger, "rank", 0) == 0:
        dagger.save(final_ckpt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
