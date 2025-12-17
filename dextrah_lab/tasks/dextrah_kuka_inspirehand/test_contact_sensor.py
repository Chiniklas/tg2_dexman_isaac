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

    import math

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sim import build_simulation_context
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass

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
        ).replace(
            init_state=KUKA_INSPIREHAND_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=start_joint_pos_map,
            ),
        )

        # Table (USD spawn)
        table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/chizhang/projects/DEXTRAH/dextrah_lab/assets/scene_objects/table.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                scale=(1.0, 1.5, 50.0),  # adjust XYZ scale here
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.21 - 0.725 / 2, 0.668 - 1.16 / 2, 0.25 - 0.03 / 2),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Table (simple rigid cuboid)
        # table = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/table",
        #     spawn=sim_utils.CuboidCfg(
        #         # size=(0.725, 1.16, 0.03),
        #         size=(0.725, 1.0, 1.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.25)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(-0.21 - 0.725 / 2, 0.668 - 1.16 / 2, 0.25 - 0.03 / 2),
        #         rot=(1.0, 0.0, 0.0, 0.0),
        #     ),
        # )

        # Object to collide with (cube)
        obj = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
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

        # # Contact sensor on arm links only (iiwa7_link_1..7).
        # contact_palm = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/iiwa7_link_[1-7]",
        #     update_period=0.0,
        #     history_length=6,
        #     debug_vis=True,
        #     filter_prim_paths_expr=[
        #         "{ENV_REGEX_NS}/table",
        #         "{ENV_REGEX_NS}/table/.*",
        #     ],
        # )

        # to test table usd contact 
        contact_palm = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/iiwa7_link_[1-7]",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/table/box",
            ],
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

        robot = scene.articulations["robot"]
        start_joint_pos = robot.data.joint_pos.clone()
        for idx, name in enumerate(robot.joint_names):
            if name in start_joint_pos_map:
                start_joint_pos[:, idx] = start_joint_pos_map[name]
        robot.write_joint_state_to_sim(start_joint_pos, torch.zeros_like(start_joint_pos))
        robot.set_joint_position_target(start_joint_pos)

        default_joint_pos = start_joint_pos.clone()
        phase = torch.zeros((1, robot.num_joints), device=default_joint_pos.device)
        amp = 0.2
        freq_hz = 0.5

        resolved_paths = contact_sensor.body_physx_view.prim_paths[: contact_sensor.num_bodies]
        print(f"[INFO] Contact sensor prim path: {contact_sensor.cfg.prim_path}")
        if len(resolved_paths) > 0:
            print(f"[INFO] First resolved body prim: {resolved_paths[0]}")

        time_s = 0.0
        while simulation_app.is_running():
            target_pos = default_joint_pos + amp * torch.sin(2.0 * math.pi * freq_hz * time_s + phase)
            robot.set_joint_position_target(target_pos)
            # Write any staged actions/commands, step physics, then update sensors/buffers.
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt=sim_dt)
            time_s += sim_dt

            # Contact sensor outputs
            data = contact_sensor.data
            if data is not None and data.net_forces_w is not None:
                # print("-------------------------------")
                # print("link sensor:", contact_sensor)
                print("force_matrix_w:", data.force_matrix_w)
                print("net_forces_w:", data.net_forces_w)
                if data.force_matrix_w is not None:
                    body_names = getattr(contact_sensor, "body_names", [])
                    filters = contact_sensor.cfg.filter_prim_paths_expr
                    nz = (data.force_matrix_w.abs().sum(-1) > 1e-4).nonzero(as_tuple=False)
                    if len(nz) == 0:
                        print("contact pairs: none")
                    else:
                        pairs = []
                        for env_idx, body_idx, filter_idx in nz.tolist():
                            body_name = body_names[body_idx] if body_idx < len(body_names) else f"body_{body_idx}"
                            filter_name = (
                                filters[filter_idx] if filter_idx < len(filters) else f"filter_{filter_idx}"
                            )
                            pairs.append((env_idx, body_name, filter_name))
                        print("contact pairs:", pairs)
                input("debugging contact sensor")

    simulation_app.close()


if __name__ == "__main__":
    main()
