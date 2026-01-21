"""Script to play a checkpoint of an RL agent from RL-Games (test wrapper)."""

import argparse

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games (test).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--objects_dir",
    type=str,
    default="visdex_objects",
    help="Name of the objects directory under assets/ to load (e.g., visdex_objects).",
)
parser.add_argument(
    "--object_name",
    type=str,
    default=None,
    help="Optional: pick a single object (folder name inside <objects_dir>/USD).",
)
parser.add_argument(
    "--max_pose_angle",
    type=float,
    default=45.0,
    help="Max palm pose angle (degrees); must be positive.",
)
# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import os
import pathlib
import shutil
import torch

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

def _prepare_single_object_dir(base_objects_dir: str, object_name: str) -> str:
    """Create a single-object view while preserving the original one-hot length.

    We mirror the full list of object folder names but point every USD file to the chosen object.
    This keeps num_unique_objects (and obs size) unchanged so checkpoints still load.
    """
    # repo_root/dextrah_lab
    root_path = pathlib.Path(__file__).resolve().parents[1]
    assets_dir = root_path / "assets"
    source_usd_dir = assets_dir / base_objects_dir / "USD"

    src_object_dir = source_usd_dir / object_name
    if not src_object_dir.is_dir():
        raise FileNotFoundError(f"Object '{object_name}' not found under {source_usd_dir}")

    # Gather full object list to preserve one-hot length
    sub_dirs = sorted([p.name for p in source_usd_dir.iterdir() if p.is_dir()])

    target_dir_name = f"{base_objects_dir}_single_pick"
    target_usd_dir = assets_dir / target_dir_name / "USD"
    if target_usd_dir.exists():
        shutil.rmtree(target_usd_dir)
    target_usd_dir = assets_dir / target_dir_name / "USD"
    target_usd_dir.mkdir(parents=True, exist_ok=True)

    chosen_usd = src_object_dir / f"{object_name}.usd"
    if not chosen_usd.is_file():
        raise FileNotFoundError(f"USD file not found for object '{object_name}' at {chosen_usd}")

    for name in sub_dirs:
        obj_dir = target_usd_dir / name
        obj_dir.mkdir(parents=True, exist_ok=True)
        link_path = obj_dir / f"{name}.usd"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(chosen_usd)

    return target_dir_name


def main():
    """Play with RL-Games agent."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Optionally force a single object by creating a temporary filtered objects_dir.
    if args_cli.object_name:
        env_cfg.objects_dir = _prepare_single_object_dir(args_cli.objects_dir, args_cli.object_name)
        # Ensure the env allows this temp dir.
        if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
            env_cfg.valid_objects_dir.append(env_cfg.objects_dir)
        print(f"[INFO] Using single object '{args_cli.object_name}' via objects_dir='{env_cfg.objects_dir}'.")
    else:
        env_cfg.objects_dir = args_cli.objects_dir
    env_cfg.max_pose_angle = args_cli.max_pose_angle
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

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

    # Pause to allow recording setup before starting simulation
    input("Ready to start simulation. Press Enter to begin...")

    count = 0
    num_evals = 5000
    sr = torch.zeros(num_evals + 1, device=args_cli.device, dtype=torch.float32)

    while simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=False)
            obs, _, dones, _ = env.step(actions)
            success_rate = env.unwrapped.in_success_region.float().mean()
            print("count", count, "sr: ", success_rate)
            sr[count] = success_rate
            count += 1

            if len(dones) > 0:
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            if count > num_evals:
                break

    env.close()
    print("final sr: ", sr[num_evals - 100:].mean())


if __name__ == "__main__":
    main()
    simulation_app.close()
