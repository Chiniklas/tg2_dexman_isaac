"""Minimal contact-sensor test using the Kuka Inspirehand USD.

Boot Isaac Sim/Kit first via AppLauncher so carb/USD are available.
"""

import argparse

from isaaclab.app import AppLauncher


def main():
    """Spawn the robot and cube, and print contact forces from the palm sensor."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Kit before importing anything that needs carb/USD.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass

    from dextrah_lab.assets.kuka_inspirehand.kuka_inspirehand import KUKA_INSPIREHAND_CFG

    @configclass
    class ContactSensorSceneCfg(InteractiveSceneCfg):
        """Design the scene with sensors on the robot."""

        # ground plane
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        # lights
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # robot -- enable contact reporters on all links so contact sensor works
        robot = KUKA_INSPIREHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=KUKA_INSPIREHAND_CFG.spawn.replace(activate_contact_sensors=True),
        )

        # Rigid Object to collide with
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.05)),
        )

        # Contact sensor on the Inspirehand palm.
        contact_palm = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/palm",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        )

    # Build the scene
    scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    # Spin up simulation context before building the scene so SimulationContext.instance() is valid.
    device = getattr(args_cli, "device", "cuda:0")
    sim_dt = 1.0 / 120.0
    with build_simulation_context(device=device, dt=sim_dt, add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)
        # Create contact sensor before resets so it initializes with the scene.
        contact_sensor = ContactSensor(scene_cfg.contact_palm)
        scene.sensors["contact_palm"] = contact_sensor

        # Start physics (this triggers asset initialization/actuators/sensors), then reset scene.
        sim.reset()
        scene.reset()

        resolved_paths = contact_sensor.body_physx_view.prim_paths[: contact_sensor.num_bodies]
        print(f"[INFO] Contact sensor prim path: {contact_sensor.cfg.prim_path}")
        if len(resolved_paths) > 0:
            print(f"[INFO] First resolved body prim: {resolved_paths[0]}")

        while simulation_app.is_running():
            # Write any staged actions/commands, step physics, then update sensors/buffers.
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)

            # Contact sensor outputs
            data = contact_sensor.data
            if data is not None and data.net_forces_w is not None:
                print("-------------------------------")
                print("palm sensor:", contact_sensor)
                print("force_matrix_w:", data.force_matrix_w)
                print("net_forces_w:", data.net_forces_w)

    simulation_app.close()


if __name__ == "__main__":
    main()
