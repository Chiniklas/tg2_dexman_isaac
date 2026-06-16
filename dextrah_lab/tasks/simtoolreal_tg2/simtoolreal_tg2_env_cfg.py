"""Configuration for the SimToolReal TG2-InspireHand Isaac Lab task."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from dextrah_lab.assets.tg2_inspirehand.tg2_inspirehand import TG2_INSPIREHAND_CFG
from .simtoolreal_tg2_utils import TG2_INSPIREHAND_JOINT_NAMES


DEXTRAH_LAB_DIR = Path(__file__).resolve().parents[2]
SIMTOOLREAL_OBJECT_USD_DIR = DEXTRAH_LAB_DIR / "assets" / "test_object" / "USD"
SIMTOOLREAL_OBJECT_SCALES = {
    "1wdf56lx": (1.0, 1.0, 1.0),
}


def _object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=False,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=100.0,
    )


def _goal_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=100.0,
    )


def _goal_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=False)


def make_cube_object_cfg(mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_cube_goal_object_cfg(mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_simtoolreal_object_cfg(object_name: str, mass: float) -> RigidObjectCfg:
    usd_path = SIMTOOLREAL_OBJECT_USD_DIR / object_name / f"{object_name}.usd"
    if object_name not in SIMTOOLREAL_OBJECT_SCALES:
        known = ", ".join(sorted(SIMTOOLREAL_OBJECT_SCALES))
        raise ValueError(f"Unknown SimToolReal TG2 object '{object_name}'. Known objects: {known}")
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing USD for SimToolReal TG2 object '{object_name}': {usd_path}")

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_simtoolreal_goal_object_cfg(object_name: str, mass: float) -> RigidObjectCfg:
    usd_path = SIMTOOLREAL_OBJECT_USD_DIR / object_name / f"{object_name}.usd"
    if object_name not in SIMTOOLREAL_OBJECT_SCALES:
        known = ", ".join(sorted(SIMTOOLREAL_OBJECT_SCALES))
        raise ValueError(f"Unknown SimToolReal TG2 object '{object_name}'. Known objects: {known}")
    if not usd_path.exists():
        raise FileNotFoundError(f"Missing USD for SimToolReal TG2 object '{object_name}': {usd_path}")

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_multi_simtoolreal_object_cfg(object_names: list[str], mass: float) -> RigidObjectCfg:
    usd_paths: list[str] = []
    for object_name in object_names:
        usd_path = SIMTOOLREAL_OBJECT_USD_DIR / object_name / f"{object_name}.usd"
        if object_name not in SIMTOOLREAL_OBJECT_SCALES:
            known = ", ".join(sorted(SIMTOOLREAL_OBJECT_SCALES))
            raise ValueError(f"Unknown SimToolReal TG2 object '{object_name}'. Known objects: {known}")
        if not usd_path.exists():
            raise FileNotFoundError(f"Missing USD for SimToolReal TG2 object '{object_name}': {usd_path}")
        usd_paths.append(str(usd_path))

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_multi_simtoolreal_goal_object_cfg(object_names: list[str], mass: float) -> RigidObjectCfg:
    usd_paths: list[str] = []
    for object_name in object_names:
        usd_path = SIMTOOLREAL_OBJECT_USD_DIR / object_name / f"{object_name}.usd"
        if object_name not in SIMTOOLREAL_OBJECT_SCALES:
            known = ", ".join(sorted(SIMTOOLREAL_OBJECT_SCALES))
            raise ValueError(f"Unknown SimToolReal TG2 object '{object_name}'. Known objects: {known}")
        if not usd_path.exists():
            raise FileNotFoundError(f"Missing USD for SimToolReal TG2 object '{object_name}': {usd_path}")
        usd_paths.append(str(usd_path))

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def configure_cube_object(cfg: "SimToolRealTg2EnvCfg") -> None:
    cfg.object_name = "cube"
    cfg.object_cfg = make_cube_object_cfg(cfg.object_mass)
    cfg.goal_object_cfg = make_cube_goal_object_cfg(cfg.object_mass)
    cfg.object_scales = (1.0, 1.0, 1.0)
    cfg.scene.replicate_physics = True


def configure_simtoolreal_object(cfg: "SimToolRealTg2EnvCfg", object_name: str) -> None:
    cfg.object_name = object_name
    cfg.object_cfg = make_simtoolreal_object_cfg(object_name, cfg.object_mass)
    cfg.goal_object_cfg = make_simtoolreal_goal_object_cfg(object_name, cfg.object_mass)
    cfg.object_scales = SIMTOOLREAL_OBJECT_SCALES[object_name]


def configure_multi_simtoolreal_objects(cfg: "SimToolRealTg2EnvCfg", object_names: list[str]) -> None:
    cfg.object_name = "multi_simtoolreal"
    cfg.multi_object_names = tuple(object_names)
    cfg.object_cfg = make_multi_simtoolreal_object_cfg(object_names, cfg.object_mass)
    cfg.goal_object_cfg = make_multi_simtoolreal_goal_object_cfg(object_names, cfg.object_mass)
    cfg.object_scales = SIMTOOLREAL_OBJECT_SCALES[object_names[0]]


def apply_object_selection(cfg: "SimToolRealTg2EnvCfg") -> None:
    if cfg.object_name == "cube":
        configure_cube_object(cfg)
    elif cfg.object_name == "multi_simtoolreal":
        cfg.scene.replicate_physics = False
        configure_multi_simtoolreal_objects(cfg, list(cfg.multi_object_names))
    elif cfg.object_name in SIMTOOLREAL_OBJECT_SCALES:
        configure_simtoolreal_object(cfg, cfg.object_name)
    else:
        known = ", ".join(["cube", "multi_simtoolreal", *sorted(SIMTOOLREAL_OBJECT_SCALES)])
        raise ValueError(f"Unknown object_name '{cfg.object_name}'. Known values: {known}")


@configclass
class SimToolRealTg2EnvCfg(DirectRLEnvCfg):
    """Direct RL config mirroring the reference IsaacGym SimToolReal task for TG2."""

    # env timing
    sim_dt = 1.0 / 60.0
    decimation = 1
    episode_length_s = 10.0
    num_actions = 13
    num_observations = 92
    num_states = 114
    observation_space = 92
    state_space = 114
    action_space = 13
    asymmetric_obs = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_collision_stack_size=2**29,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1536, env_spacing=1.2, replicate_physics=True)

    # assets
    object_name = "cube"
    multi_object_names = tuple(sorted(SIMTOOLREAL_OBJECT_SCALES))
    robot_cfg = TG2_INSPIREHAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=TG2_INSPIREHAND_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pitch_r_joint": -1.570796,
                "shoulder_roll_r_joint": -0.523599,
                "shoulder_yaw_r_joint": 1.108284,
                "elbow_pitch_r_joint": -1.275836,
                "elbow_yaw_r_joint": 0.089012,
                "wrist_pitch_r_joint": -0.027925,
                "wrist_roll_r_joint": -0.048869,
                "index_joint_0": 0.0,
                "little_joint_0": 0.0,
                "middle_joint_0": 0.0,
                "ring_joint_0": 0.0,
                "thumb_joint_0": 0.4,
                "index_joint_1": 0.0,
                "little_joint_1": 0.0,
                "middle_joint_1": 0.0,
                "ring_joint_1": 0.0,
                "thumb_joint_1": 0.1,
                "thumb_joint_2": 0.2,
                "thumb_joint_3": 0.4,
            },
        )
    )
    actuated_joint_names = TG2_INSPIREHAND_JOINT_NAMES
    palm_body_name = "palm"
    fingertip_body_names = [
        "index_tip",
        "middle_tip",
        "ring_tip",
        "little_tip",
        "thumb_tip",
    ]
    palm_offset = (0.0, 0.0, 0.0)
    fingertip_offsets = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.475, 0.4, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.56, 0.35)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/table",
        debug_vis=False,
        filter_prim_paths_expr=["/World/envs/env_.*/object"],
    )
    object_mass = 0.05
    object_cfg: RigidObjectCfg = make_cube_object_cfg(object_mass)
    goal_object_cfg: RigidObjectCfg = make_cube_goal_object_cfg(object_mass)

    # reset/control
    clamp_abs_observations = 10.0
    use_relative_control = False
    dof_speed_scale = 1.5
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    reset_position_noise_x = 0.1
    reset_position_noise_y = 0.1
    reset_position_noise_z = 0.02
    reset_dof_pos_noise_fingers = 0.1
    reset_dof_pos_noise_arm = 0.1
    reset_dof_vel_noise = 0.5
    randomize_object_rotation = True
    object_start_pose: tuple[float, float, float, float, float, float, float] | None = None
    goal_object_pose: tuple[float, float, float, float, float, float, float] | None = None
    debug_keypoints = False
    debug_grasp_bounding_box = False
    debug_keypoint_radius = 0.012
    debug_grasp_bounding_box_line_width = 3.0

    # reference task geometry
    table_top_z = 0.53
    table_reset_z_range = 0.01
    table_object_z_offset = 0.25
    object_base_size = 0.04
    object_scales = (1.0, 1.0, 1.0)
    object_scale_noise_multiplier_range = (0.9, 1.1)
    fixed_size_keypoint_reward = True
    fixed_size = (0.141, 0.03025, 0.0271)
    keypoint_scale = 1.5
    target_volume_mins = (-0.35, -0.1, 0.68)
    target_volume_maxs = (0.35, 0.2, 1.05)
    target_volume_region_scale = 1.0
    goal_sampling_type = "delta"
    delta_goal_distance = 0.1
    delta_rotation_degrees = 90.0

    # rewards/resets
    lifting_rew_scale = 20.0
    lifting_bonus = 300.0
    lifting_bonus_threshold = 0.15
    keypoint_rew_scale = 200.0
    distance_delta_rew_scale = 50.0
    reach_goal_bonus = 1000.0
    arm_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003
    fall_distance = 0.24
    fall_penalty = 0.0
    object_lin_vel_penalty_scale = 0.0
    object_ang_vel_penalty_scale = 0.0
    object_z_low_reset_threshold = 0.1
    hand_far_from_object_threshold = 1.5
    with_table_force_sensor = False
    table_force_threshold = 100.0
    reset_when_dropped = True
    success_tolerance = 0.075
    target_success_tolerance = 0.01
    tolerance_curriculum_increment = 0.9
    tolerance_curriculum_interval = 3000
    eval_success_tolerance = None
    success_steps = 10
    max_consecutive_successes = 50
    force_consecutive_near_goal_steps = False

    # sim2real/domain-randomization delays and observation noise
    use_obs_delay = True
    obs_delay_max = 3
    use_action_delay = True
    action_delay_max = 3
    use_object_state_delay_noise = True
    object_state_delay_max = 10
    object_state_xyz_noise_std = 0.01
    object_state_rotation_noise_degrees = 5.0
    joint_velocity_obs_noise_std = 0.01

    # object force/torque disturbances
    force_scale = 2.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08
    force_only_when_lifted = True
    torque_scale = 0.0
    torque_prob_range = (0.001, 0.1)
    torque_decay = 0.99
    torque_decay_interval = 0.08
    torque_only_when_lifted = False

    def __post_init__(self):
        super().__post_init__()
        apply_object_selection(self)
