"""Spawn and inspect the SimToolReal TG2 Isaac Lab scene.

Visual scene check:
python tg2_lab/tasks/simtoolreal_tg2/tests/test_scene_spawning.py --num-envs 1

Headless smoke check:
python tg2_lab/tasks/simtoolreal_tg2/tests/test_scene_spawning.py --headless --num-envs 1 --steps 2
"""

from __future__ import annotations

import argparse
import os
import traceback


def _spawn_label(spawn_cfg) -> str:
    return getattr(spawn_cfg, "usd_path", type(spawn_cfg).__name__)


def _make_env_cfg(
    *,
    num_envs: int,
    device: str | None,
    debug_keypoints: bool,
    debug_grasp_bounding_box: bool,
    debug_fingertips: bool,
):
    """Build SimToolRealTg2EnvCfg after Kit/SimulationApp has started."""
    from tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env_cfg import SimToolRealTg2EnvCfg

    cfg = SimToolRealTg2EnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device or os.environ.get("SIMTOOLREAL_TG2_TEST_DEVICE", "cuda:0")
    cfg.debug_keypoints = debug_keypoints
    cfg.debug_grasp_bounding_box = debug_grasp_bounding_box
    cfg.debug_fingertips = debug_fingertips

    return cfg


def _spawn_env(cfg):
    """Construct and reset the current task env."""
    from tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env import SimToolRealTg2Env

    env = SimToolRealTg2Env(cfg)
    obs, _ = env.reset()
    _assert_scene(env, cfg, obs)
    _print_scene_summary(env, cfg, obs)
    return env


def _assert_scene(env, cfg, obs) -> None:
    unwrapped = env.unwrapped

    assert unwrapped.num_envs == cfg.scene.num_envs
    assert set(unwrapped.scene.articulations) == {"robot"}
    assert set(unwrapped.scene.rigid_objects) == {"table", "object", "goal_object"}
    assert obs["policy"].shape == (cfg.scene.num_envs, cfg.num_observations)
    assert obs["critic"].shape == (cfg.scene.num_envs, cfg.num_states)
    assert unwrapped.num_actions == cfg.num_actions

    for name in ("robot",):
        assert name in unwrapped.scene.articulations
    for name in ("table", "object", "goal_object"):
        assert name in unwrapped.scene.rigid_objects


def _print_scene_summary(env, cfg, obs) -> None:
    unwrapped = env.unwrapped
    robot = unwrapped.scene.articulations["robot"]
    table = unwrapped.scene.rigid_objects["table"]
    obj = unwrapped.scene.rigid_objects["object"]
    goal = unwrapped.scene.rigid_objects["goal_object"]

    print("simtoolreal_tg2 scene spawned", flush=True)
    print(f"  num_envs: {unwrapped.num_envs}", flush=True)
    print(f"  device: {unwrapped.device}", flush=True)
    print(f"  actions: {unwrapped.num_actions}", flush=True)
    print(f"  obs policy: {tuple(obs['policy'].shape)}", flush=True)
    print(f"  obs critic: {tuple(obs['critic'].shape)}", flush=True)
    print(f"  robot spawn: {_spawn_label(cfg.robot_cfg.spawn)}", flush=True)
    print(f"  table spawn: {_spawn_label(cfg.table_cfg.spawn)}", flush=True)
    print(f"  object spawn: {_spawn_label(cfg.object_cfg.spawn)}", flush=True)
    print(f"  goal spawn: {_spawn_label(cfg.goal_object_cfg.spawn)}", flush=True)
    print(f"  robot init pose: {cfg.robot_cfg.init_state.pos} {cfg.robot_cfg.init_state.rot}", flush=True)
    print(f"  table init pose: {cfg.table_cfg.init_state.pos} {cfg.table_cfg.init_state.rot}", flush=True)
    print(
        f"  object spawn center/range: {cfg.object_spawn_center} +/-{cfg.object_spawn_xy_range}m",
        flush=True,
    )
    print(f"  goal height range: {cfg.goal_height_above_object_range}", flush=True)
    print(f"  robot root: {_tensor_row(robot.data.root_pos_w)}", flush=True)
    print(f"  table root: {_tensor_row(table.data.root_pos_w)}", flush=True)
    print(f"  object root: {_tensor_row(obj.data.root_pos_w)}", flush=True)
    print(f"  goal root: {_tensor_row(goal.data.root_pos_w)}", flush=True)


def _tensor_row(tensor) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in tensor[0].detach().cpu().tolist())


def _make_zero_actions(env):
    import torch

    return torch.zeros((env.num_envs, env.num_actions), device=env.device)


def _step_env(env, actions):
    obs, rewards, terminated, truncated, _ = env.step(actions)
    return obs, rewards, terminated, truncated


def _run_headless(env, steps: int, print_every: int) -> None:
    actions = _make_zero_actions(env)
    for step_idx in range(1, steps + 1):
        _obs, rewards, terminated, truncated = _step_env(env, actions)
        if print_every > 0 and (step_idx == 1 or step_idx % print_every == 0):
            print(
                f"step={step_idx} reward_mean={float(rewards.mean()):.4f} "
                f"terminated={int(terminated.sum())} truncated={int(truncated.sum())}",
                flush=True,
            )
    print("Headless smoke check complete.", flush=True)


def _run_visual_forever(env, print_every: int) -> None:
    actions = _make_zero_actions(env)
    print("Visual scene is live. Press Ctrl-C in this terminal to stop.", flush=True)
    step_idx = 0
    try:
        while True:
            step_idx += 1
            _obs, rewards, terminated, truncated = _step_env(env, actions)
            if print_every > 0 and step_idx % print_every == 0:
                print(
                    f"step={step_idx} reward_mean={float(rewards.mean()):.4f} "
                    f"terminated={int(terminated.sum())} truncated={int(truncated.sum())}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Stopping visual scene.", flush=True)


def _run_with_app(args_cli) -> None:
    from isaaclab.app import AppLauncher

    print("Launching Isaac Sim app...", flush=True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("Isaac Sim app launched. Building simtoolreal_tg2 env cfg...", flush=True)
    env = None
    try:
        cfg = _make_env_cfg(
            num_envs=args_cli.num_envs,
            device=getattr(args_cli, "device", None),
            debug_keypoints=args_cli.debug_keypoints,
            debug_grasp_bounding_box=args_cli.debug_grasp_bounding_box,
            debug_fingertips=args_cli.debug_fingertips,
        )
        print("Env cfg built. Spawning simtoolreal_tg2 env...", flush=True)
        env = _spawn_env(cfg)
        print("Env spawned and reset.", flush=True)
        if args_cli.headless:
            _run_headless(env, args_cli.steps, args_cli.print_every)
        else:
            _run_visual_forever(env, args_cli.print_every)
    except BaseException as exc:
        print(f"Scene spawning failed with {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        if isinstance(exc, SystemExit):
            raise RuntimeError(f"Unexpected SystemExit while spawning simtoolreal_tg2 scene: {exc}") from exc
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def test_simtoolreal_tg2_scene_spawns_from_env_config():
    import pytest

    isaaclab_app = pytest.importorskip("isaaclab.app", reason="Isaac Lab is required to load this scene.")
    simulation_app = isaaclab_app.AppLauncher(headless=True).app
    env = None
    try:
        cfg = _make_env_cfg(
            num_envs=1,
            device=os.environ.get("SIMTOOLREAL_TG2_TEST_DEVICE", "cuda:0"),
            debug_keypoints=False,
            debug_grasp_bounding_box=False,
            debug_fingertips=False,
        )
        env = _spawn_env(cfg)
        _run_headless(env, steps=1, print_every=0)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Spawn and visualize the simtoolreal_tg2 scene.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2, help="Number of env steps for --headless.")
    parser.add_argument("--print-every", type=int, default=120, help="Print step summary every N steps; 0 disables.")
    parser.add_argument("--debug-keypoints", action="store_true", help="Draw object/goal keypoints.")
    parser.add_argument("--debug-grasp-bounding-box", action="store_true", help="Draw object/goal grasp bounding boxes.")
    parser.add_argument("--debug-fingertips", action="store_true", help="Draw fingertip debug points.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    _run_with_app(args_cli)


if __name__ == "__main__":
    main()
