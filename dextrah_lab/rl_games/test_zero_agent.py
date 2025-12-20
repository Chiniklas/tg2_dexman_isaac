"""Minimal contact-sensor test using the Kuka Inspirehand USD and a USD object."""

import argparse
import pathlib

from isaaclab.app import AppLauncher


def main():
    """Spawn the robot, table, and a USD object, then run a simple contact-sensor loop."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch Kit before importing anything that needs carb/USD.
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import math

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass
    from isaacsim.core.utils.prims import set_prim_attribute_value

    from dextrah_lab.assets.kuka_inspirehand.kuka_inspirehand import KUKA_INSPIREHAND_CFG

    start_joint_pos_map = {
        "iiwa7_joint_1": -0.85,
        "iiwa7_joint_2": -0.0,
        "iiwa7_joint_3": 0.76,
        "iiwa7_joint_4": 1.25,
        "iiwa7_joint_5": -1.76,
        "iiwa7_joint_6": 0.90,
        "iiwa7_joint_7": 0.64,
        "index_joint_0": 0.55,
        "little_joint_0": 0.55,
        "middle_joint_0": 0.55,
        "ring_joint_0": 0.55,
        "thumb_joint_0": 0.3,
        "index_joint_1": 0.85,
        "little_joint_1": 0.85,
        "middle_joint_1": 0.85,
        "ring_joint_1": 0.85,
        "thumb_joint_1": 0.25,
        "thumb_joint_2": 0.25,
        "thumb_joint_3": 0.6,
    }

    @configclass
    class ContactSensorSceneCfg(InteractiveSceneCfg):
        """Design the scene with sensors on the robot."""

        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        robot = KUKA_INSPIREHAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=KUKA_INSPIREHAND_CFG.spawn.replace(activate_contact_sensors=True),
        ).replace(
            init_state=KUKA_INSPIREHAND_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=start_joint_pos_map,
            ),
        )

        table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/chizhang/projects/DEXTRAH/dextrah_lab/assets/scene_objects/table.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.21 - 0.725 / 2, 0.668 - 1.16 / 2, 0.25 - 0.03 / 2),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        obj = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/chizhang/projects/DEXTRAH/dextrah_lab/assets/visdex_objects/USD/1b0kr9wf/1b0kr9wf.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1000.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        contact_palm = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/iiwa7_link_[1-7]",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/table/box",
            ],
        )

    scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    sim_dt = 1.0 / 120.0
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device="cpu")

    with build_simulation_context(sim_cfg=sim_cfg, device="cpu", add_ground_plane=False, add_lighting=False) as sim:
        scene = InteractiveScene(scene_cfg)
        contact_sensor = ContactSensor(scene_cfg.contact_palm)
        scene.sensors["contact_palm"] = contact_sensor

        # Disable articulation root on the spawned object before physics init.
        set_prim_attribute_value(
            prim_path="/World/envs/env_0/object/baseLink",
            attribute_name="physxArticulation:articulationEnabled",
            value=False,
        )

        sim.reset()
        scene.reset()

        robot = scene.articulations["robot"]
        # # Disable PD control so the robot is free for manual inspector control.
        # for act in robot.actuators.values():
        #     if hasattr(act, "stiffness"):
        #         act.stiffness[:] = 0.0
        #     if hasattr(act, "damping"):
        #         act.damping[:] = 0.0

        # start_joint_pos = robot.data.joint_pos.clone()
        # for idx, name in enumerate(robot.joint_names):
        #     if name in start_joint_pos_map:
        #         start_joint_pos[:, idx] = start_joint_pos_map[name]
        # robot.write_joint_state_to_sim(start_joint_pos, torch.zeros_like(start_joint_pos))

        resolved_paths = contact_sensor.body_physx_view.prim_paths[: contact_sensor.num_bodies]
        print(f"[INFO] Contact sensor prim path: {contact_sensor.cfg.prim_path}")
        if len(resolved_paths) > 0:
            print(f"[INFO] First resolved body prim: {resolved_paths[0]}")

        while simulation_app.is_running():
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
