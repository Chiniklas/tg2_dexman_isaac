"""Train the SimToolReal TG2 task with the vendored SAPO RL-Games fork."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import pathlib
import sys
import time
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEXTRAH_ROOT = pathlib.Path(__file__).resolve().parent
TASK_ROOT = DEXTRAH_ROOT / "tasks" / "simtoolreal_tg2"
VENDORED_RL_GAMES = DEXTRAH_ROOT / "rl_games"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the SimToolReal TG2 task with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="simtoolreal_tg2", help="Gym task id.")
parser.add_argument("--seed", type=int, default=None, help="Environment/agent seed.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="Initial policy standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum RL-Games epochs.")
parser.add_argument("--agent_cfg", type=str, default=None, help="Path or agents/ filename for an RL-Games YAML recipe.")
parser.add_argument(
    "--visualize_keypoints",
    "--visualize-keypoints",
    dest="visualize_keypoints",
    action="store_true",
    default=False,
    help="Draw task keypoints in the Isaac viewer.",
)
parser.add_argument(
    "--visualize_grasp_bounding_box",
    "--visualize-grasp-bounding-box",
    "--visualize_grasp_bouding_box",
    "--visualize-grasp-bouding-box",
    dest="visualize_grasp_bounding_box",
    action="store_true",
    default=False,
    help="Draw grasp bounding box debug geometry.",
)
parser.add_argument(
    "--visualize_fingertips",
    "--visualize-fingertips",
    dest="visualize_fingertips",
    action="store_true",
    default=False,
    help="Draw fingertip debug points in the Isaac viewer.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if not any(arg.startswith("hydra.run.dir=") for arg in hydra_args):
    hydra_args.append(f"hydra.run.dir={TASK_ROOT / 'outputs'}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if VENDORED_RL_GAMES.is_dir():
    sys.path.insert(0, str(VENDORED_RL_GAMES))

import gymnasium as gym
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import dextrah_lab.tasks.simtoolreal_tg2.gym_setup  # noqa: F401


def _load_agent_cfg_override(agent_cfg_path: str) -> dict:
    path = pathlib.Path(agent_cfg_path).expanduser()
    if not path.exists():
        path = TASK_ROOT / "agents" / agent_cfg_path
    if not path.exists():
        raise FileNotFoundError(f"Agent cfg not found: {agent_cfg_path}")
    with path.open(encoding="utf-8") as f:
        loaded_cfg = yaml.safe_load(f)
    if not isinstance(loaded_cfg, dict) or "params" not in loaded_cfg:
        raise ValueError(f"Agent cfg must be a YAML mapping with a top-level 'params' key: {path}")
    print(f"[INFO]: Loading agent cfg from: {path}")
    return loaded_cfg


def _apply_object_selection(env_cfg) -> None:
    cfg_module = importlib.import_module(env_cfg.__class__.__module__)
    apply_selection = getattr(cfg_module, "apply_object_selection", None)
    if apply_selection is not None:
        apply_selection(env_cfg)


def _set_cfg_value(cfg, key_path: str, value) -> None:
    target = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise AttributeError(f"Unknown env cfg key '{key_path}': missing '{key}'.")
        target = getattr(target, key)
    final_key = keys[-1]
    if not hasattr(target, final_key):
        raise AttributeError(f"Unknown env cfg key '{key_path}': missing '{final_key}'.")
    setattr(target, final_key, value)


def _set_dict_value(mapping: dict, key_path: str, value) -> None:
    target = mapping
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise KeyError(f"Unknown agent cfg key '{key_path}': missing '{key}'.")
        target = target[key]
    final_key = keys[-1]
    if final_key not in target:
        raise KeyError(f"Unknown agent cfg key '{key_path}': missing '{final_key}'.")
    target[final_key] = value


def _parse_cli_override(arg: str) -> tuple[str, object] | None:
    if "=" not in arg:
        return None
    key, raw_value = arg.split("=", 1)
    if not key.startswith(("env.", "agent.")):
        return None
    return key, yaml.safe_load(raw_value)


def _apply_agent_env_cfg(env_cfg, agent_cfg: dict) -> None:
    env_overrides = agent_cfg.get("env_cfg", {})
    for key_path, value in env_overrides.items():
        _set_cfg_value(env_cfg, key_path, value)

    if "sim_dt" in env_overrides:
        env_cfg.sim.dt = env_cfg.sim_dt
    if "decimation" in env_overrides:
        env_cfg.sim.render_interval = env_cfg.decimation


def _apply_cli_config_overrides(env_cfg, agent_cfg: dict) -> None:
    env_timing_changed = False
    for arg in hydra_args:
        parsed = _parse_cli_override(arg)
        if parsed is None:
            continue
        key_path, value = parsed
        if key_path.startswith("env."):
            env_key = key_path.removeprefix("env.")
            _set_cfg_value(env_cfg, env_key, value)
            env_timing_changed = env_timing_changed or env_key in {"sim_dt", "decimation"}
        elif key_path.startswith("agent."):
            _set_dict_value(agent_cfg, key_path.removeprefix("agent."), value)

    if env_timing_changed:
        env_cfg.sim.dt = env_cfg.sim_dt
        env_cfg.sim.render_interval = env_cfg.decimation


class SimToolRealRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    def set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.unwrapped, "set_train_info"):
            self.unwrapped.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        if hasattr(self.unwrapped, "get_env_state"):
            return self.unwrapped.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.unwrapped, "set_env_state"):
            self.unwrapped.set_env_state(env_state)


class SimToolRealRlGamesGpuEnv(RlGamesGpuEnv):
    def set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    if args_cli.seed == -1:
        args_cli.seed = int(time.time())

    if args_cli.agent_cfg is not None:
        agent_cfg = _load_agent_cfg_override(args_cli.agent_cfg)

    _apply_agent_env_cfg(env_cfg, agent_cfg)
    _apply_cli_config_overrides(env_cfg, agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.visualize_keypoints and hasattr(env_cfg, "debug_keypoints"):
        env_cfg.debug_keypoints = True
    if args_cli.visualize_grasp_bounding_box and hasattr(env_cfg, "debug_grasp_bounding_box"):
        env_cfg.debug_grasp_bounding_box = True
    if args_cli.visualize_fingertips and hasattr(env_cfg, "debug_fingertips"):
        env_cfg.debug_fingertips = True
    _apply_object_selection(env_cfg)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    expl_type = agent_cfg["params"]["config"].get("expl_type", "none")
    continuous_space_cfg = agent_cfg["params"]["network"]["space"]["continuous"]
    if not str(expl_type).startswith("mixed_expl") and continuous_space_cfg.get("fixed_sigma") == "coef_cond":
        print("[INFO]: Setting fixed_sigma='fixed' because coef_cond requires mixed_expl coefficient IDs.")
        continuous_space_cfg["fixed_sigma"] = "fixed"

    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]
    log_root_path = str((TASK_ROOT / "logs").resolve())
    log_dir = agent_cfg["params"]["config"].get(
        "full_experiment_name", f"0_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SimToolRealRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: SimToolRealRlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    batch_size = env.unwrapped.num_envs * agent_cfg["params"]["config"]["horizon_length"]
    if agent_cfg["params"]["config"]["minibatch_size"] > batch_size:
        print(f"[INFO]: Setting minibatch_size={batch_size} for batch_size={batch_size}.")
        agent_cfg["params"]["config"]["minibatch_size"] = batch_size
    central_value_config = agent_cfg["params"]["config"].get("central_value_config")
    if central_value_config is not None and central_value_config.get("minibatch_size", batch_size) > batch_size:
        minibatch_size = min(agent_cfg["params"]["config"]["minibatch_size"], batch_size)
        print(
            f"[INFO]: Setting central_value_config.minibatch_size={minibatch_size} "
            f"for batch_size={batch_size}."
        )
        central_value_config["minibatch_size"] = minibatch_size

    print(f"[INFO] Using rl_games from: {sys.modules['rl_games'].__file__}")
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
