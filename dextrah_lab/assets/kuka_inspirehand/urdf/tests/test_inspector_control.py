"""Spawn the Kuka Inspirehand and leave it free for Physics Inspector control."""

import argparse

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1, help="Number of robot instances.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch Kit before importing heavy Omni/Isaac modules.
    app = AppLauncher(args)
    simulation_app = app.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim import build_simulation_context, SimulationCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from dextrah_lab.assets.kuka_inspirehand.kuka_inspirehand import KUKA_INSPIREHAND_CFG

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        )
        robot = KUKA_INSPIREHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene_cfg = SceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=False)

    # Use CPU PhysX to keep Physics Inspector joint drives happy.
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device="cpu")

    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        # Disable PD stiffness/damping so inspector can drive freely.
        # for act in robot.actuators.values():
        #     if hasattr(act, "stiffness"):
        #         act.stiffness[:] = 0.0
        #     if hasattr(act, "damping"):
        #         act.damping[:] = 0.0

        sim_dt = sim_cfg.dt
        input("Robot spawned. Open Physics Inspector, then press Enter to start stepping...")

        while simulation_app.is_running():
            # scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
