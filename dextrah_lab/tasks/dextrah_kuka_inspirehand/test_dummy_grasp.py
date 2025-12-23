"""Dummy grasp sanity check for the Kuka-Inspirehand.

Boot Isaac Sim/Kit first via AppLauncher so carb/USD are available.
Spawns the arm+hand with ground plane and dome light, then holds a nominal pose.
"""

import argparse

from isaaclab.app import AppLauncher


def main():
    """Spawn the robot in a simple scene and hold a nominal joint configuration."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Kit before importing anything that needs carb/USD.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from dextrah_lab.assets.kuka_inspirehand.kuka_inspirehand import KUKA_INSPIREHAND_CFG

    # Nominal joint angles for a reachable tabletop pose; fingers open.
    nominal_joint_pos = {
        "iiwa7_joint_1": -0.85,
        "iiwa7_joint_2": 0.0,
        "iiwa7_joint_3": 0.76,
        "iiwa7_joint_4": 1.25,
        "iiwa7_joint_5": -1.76,
        "iiwa7_joint_6": 0.90,
        "iiwa7_joint_7": 0.64,
        "index_joint_0": 0.0,
        "middle_joint_0": 0.0,
        "ring_joint_0": 0.0,
        "little_joint_0": 0.0,
        "thumb_joint_0": 0.0,
        "thumb_joint_1": 0.0,
    }

    @configclass
    class DummyGraspSceneCfg(InteractiveSceneCfg):
        """Simple scene with ground, lighting, and the robot."""

        # Ground plane and lighting
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=4000.0, color=(0.9, 0.9, 0.9)),
        )

        # Robot at nominal pose
        robot = KUKA_INSPIREHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=KUKA_INSPIREHAND_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=nominal_joint_pos,
            ),
        )

    # Build the scene
    scene_cfg = DummyGraspSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    device = getattr(args_cli, "device", "cuda:0")
    sim_dt = 1.0 / 120.0

    # Spin up simulation context before building the scene so SimulationContext.instance() is valid.
    with build_simulation_context(device=device, dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)

        # Start physics (this triggers asset initialization/actuators), then reset scene.
        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.joint_vel)
        for idx, name in enumerate(robot.joint_names):
            if name in nominal_joint_pos:
                joint_pos[:, idx] = nominal_joint_pos[name]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)

        print("[INFO] Dummy grasp scene running. Close the window to exit.")
        while simulation_app.is_running():
            # Hold the nominal pose; adjust targets here for simple grasp tests.
            robot.set_joint_position_target(joint_pos)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)


if __name__ == "__main__":
    main()
