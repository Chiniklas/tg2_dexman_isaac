"""Spawn the TG2-Inspirehand env, hold zero actions, and enable stereo cameras."""

import argparse

from isaaclab.app import AppLauncher


def _describe_tensor(tensor) -> str:
    if tensor is None:
        return "None"
    shape = tuple(tensor.shape)
    return f"{shape}, dtype={tensor.dtype}"


def main() -> None:
    parser = argparse.ArgumentParser(description="TG2-Inspirehand camera setup test.")
    parser.add_argument(
        "--task",
        type=str,
        default="dextrah_tg2_inspirehand",
        help="Gym task name to load.",
    )
    parser.add_argument(
        "--objects_dir",
        type=str,
        default="test_object",
        help="Objects directory under assets/ to load (e.g., test_object, visdex_objects).",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--print_camera_every",
        type=int,
        default=60,
        help="Print camera output shapes every N steps.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import dextrah_lab.tasks.tg2_inspirehand.gym_setup  # noqa: F401

    device = getattr(args_cli, "device", "cuda:0")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.objects_dir = args_cli.objects_dir
    if env_cfg.objects_dir not in env_cfg.valid_objects_dir:
        env_cfg.valid_objects_dir.append(env_cfg.objects_dir)
    env_cfg.distillation = True
    env_cfg.simulate_stereo = True
    env_cfg.action_mode = "absolute"

    env = gym.make(args_cli.task, cfg=env_cfg)
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        _ = reset_out[0]

    robot = env.unwrapped.robot
    sim = env.unwrapped.sim
    scene = env.unwrapped.scene
    hold_pos = env.unwrapped.robot_start_joint_pos.clone()
    hold_vel = torch.zeros_like(hold_pos)

    print(
        "[INFO] Running with"
        f" num_envs={env.unwrapped.num_envs},"
        f" num_actions={env.unwrapped.num_actions} (holding initial pose)."
    )

    step_count = 0
    print_every = max(1, args_cli.print_camera_every)

    while simulation_app.is_running():
        # Hard-clamp the joints to the initial pose each frame to eliminate any drift/twitching.
        robot.write_joint_state_to_sim(hold_pos, hold_vel)
        robot.set_joint_position_target(hold_pos)
        robot.set_joint_velocity_target(hold_vel)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt=env.unwrapped.physics_dt)

        if step_count % print_every == 0:
            left_cam = env.unwrapped.scene.sensors.get("tiled_camera_left")
            right_cam = env.unwrapped.scene.sensors.get("tiled_camera_right")
            if left_cam is None:
                print("[WARN] tiled_camera_left not found.")
            else:
                left_rgb = left_cam.data.output.get("rgb")
                left_depth = left_cam.data.output.get("depth")
                print(
                    "[CAMERA] left"
                    f" rgb={_describe_tensor(left_rgb)}"
                    f" depth={_describe_tensor(left_depth)}"
                )
            if right_cam is None:
                print("[WARN] tiled_camera_right not found.")
            else:
                right_rgb = right_cam.data.output.get("rgb")
                right_depth = right_cam.data.output.get("depth")
                print(
                    "[CAMERA] right"
                    f" rgb={_describe_tensor(right_rgb)}"
                    f" depth={_describe_tensor(right_depth)}"
                )

        step_count += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
