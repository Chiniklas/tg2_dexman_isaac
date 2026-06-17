"""Tiangong2Pro half-body + InspireHand robot configuration for Isaac Lab."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


MODULE_PATH = os.path.dirname(__file__)
TIANGONG2PRO_HALF_BODY_USD_PATH = os.path.join(
    MODULE_PATH,
    "usd",
    "tiangong2.0_pro_with_hands_half_body",
    "tiangong2.0_pro_with_hands_half_body.usd",
)

TIANGONG2PRO_ARM_JOINT_NAMES = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

# Policy/deployment command space: arm joints plus one command per non-thumb
# finger and two thumb commands. Distal joints are expanded by the task wrapper.
TIANGONG2PRO_POLICY_JOINT_NAMES = [
    *TIANGONG2PRO_ARM_JOINT_NAMES,
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "little_joint_0",
    "thumb_joint_0",
    "thumb_joint_1",
]

TIANGONG2PRO_ACTUATED_JOINT_NAMES = [
    *TIANGONG2PRO_ARM_JOINT_NAMES,
    "index_joint_0",
    "index_joint_1",
    "middle_joint_0",
    "middle_joint_1",
    "ring_joint_0",
    "ring_joint_1",
    "little_joint_0",
    "little_joint_1",
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]

TIANGONG2PRO_MIMIC_JOINT_MAP = {
    "index_joint_1": ("index_joint_0", 1.54545, 0.0),
    "middle_joint_1": ("middle_joint_0", 1.54545, 0.0),
    "ring_joint_1": ("ring_joint_0", 1.54545, 0.0),
    "little_joint_1": ("little_joint_0", 1.54545, 0.0),
    "thumb_joint_2": ("thumb_joint_1", 1.0, 0.0),
    "thumb_joint_3": ("thumb_joint_2", 0.8, 0.0),
}

_SHARPA_STIFFNESS = {
    "shoulder_pitch_r_joint": 600.0,
    "shoulder_roll_r_joint": 600.0,
    "shoulder_yaw_r_joint": 500.0,
    "elbow_pitch_r_joint": 400.0,
    "elbow_yaw_r_joint": 200.0,
    "wrist_pitch_r_joint": 200.0,
    "wrist_roll_r_joint": 200.0,
    "index_joint_0": 4.76,
    "middle_joint_0": 4.76,
    "ring_joint_0": 4.76,
    "little_joint_0": 4.76,
    "index_joint_1": 0.9,
    "middle_joint_1": 0.9,
    "ring_joint_1": 0.9,
    "little_joint_1": 0.9,
    "thumb_joint_0": 6.95,
    "thumb_joint_1": 6.95,
    "thumb_joint_2": 4.76,
    "thumb_joint_3": 0.9,
}

_SHARPA_DAMPING = {
    "shoulder_pitch_r_joint": 27.0,
    "shoulder_roll_r_joint": 27.0,
    "shoulder_yaw_r_joint": 24.7,
    "elbow_pitch_r_joint": 22.1,
    "elbow_yaw_r_joint": 9.2,
    "wrist_pitch_r_joint": 9.2,
    "wrist_roll_r_joint": 9.2,
    "index_joint_0": 0.21,
    "middle_joint_0": 0.21,
    "ring_joint_0": 0.21,
    "little_joint_0": 0.21,
    "index_joint_1": 0.042,
    "middle_joint_1": 0.042,
    "ring_joint_1": 0.042,
    "little_joint_1": 0.042,
    "thumb_joint_0": 0.29,
    "thumb_joint_1": 0.29,
    "thumb_joint_2": 0.21,
    "thumb_joint_3": 0.042,
}

_SHARPA_EFFORT_LIMITS = {
    "shoulder_.*_r_joint": 300.0,
    "elbow_.*_r_joint": 300.0,
    "wrist_.*_r_joint": 300.0,
    "index_joint_0": 1.864,
    "middle_joint_0": 1.864,
    "ring_joint_0": 1.864,
    "little_joint_0": 1.864,
    "index_joint_1": 0.638,
    "middle_joint_1": 0.638,
    "ring_joint_1": 0.638,
    "little_joint_1": 0.638,
    "thumb_joint_0": 3.3,
    "thumb_joint_1": 3.3,
    "thumb_joint_2": 1.864,
    "thumb_joint_3": 0.638,
}

_SHARPA_VELOCITY_LIMITS = {
    "shoulder_.*_r_joint": 10.0,
    "elbow_.*_r_joint": 10.0,
    "wrist_.*_r_joint": 10.0,
    "index_joint_0": 16.08,
    "middle_joint_0": 16.08,
    "ring_joint_0": 16.08,
    "little_joint_0": 16.08,
    "index_joint_1": 11.62,
    "middle_joint_1": 11.62,
    "ring_joint_1": 11.62,
    "little_joint_1": 11.62,
    "thumb_joint_0": 11.84,
    "thumb_joint_1": 11.84,
    "thumb_joint_2": 16.08,
    "thumb_joint_3": 11.62,
}

_SHARPA_ARMATURE = {
    "index_joint_0": 0.00265,
    "middle_joint_0": 0.00265,
    "ring_joint_0": 0.00265,
    "little_joint_0": 0.00265,
    "index_joint_1": 0.0006,
    "middle_joint_1": 0.0006,
    "ring_joint_1": 0.0006,
    "little_joint_1": 0.0006,
    "thumb_joint_0": 0.0032,
    "thumb_joint_1": 0.0032,
    "thumb_joint_2": 0.00265,
    "thumb_joint_3": 0.0006,
}

_SHARPA_FRICTION = {
    "index_joint_0": 0.07456,
    "middle_joint_0": 0.07456,
    "ring_joint_0": 0.07456,
    "little_joint_0": 0.07456,
    "index_joint_1": 0.01276,
    "middle_joint_1": 0.01276,
    "ring_joint_1": 0.01276,
    "little_joint_1": 0.01276,
    "thumb_joint_0": 0.132,
    "thumb_joint_1": 0.132,
    "thumb_joint_2": 0.07456,
    "thumb_joint_3": 0.01276,
}


TIANGONG2PRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TIANGONG2PRO_HALF_BODY_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=500.0,
            max_depenetration_velocity=30.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
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
            "index_joint_1": 0.0,
            "middle_joint_0": 0.0,
            "middle_joint_1": 0.0,
            "ring_joint_0": 0.0,
            "ring_joint_1": 0.0,
            "little_joint_0": 0.0,
            "little_joint_1": 0.0,
            "thumb_joint_0": 0.4,
            "thumb_joint_1": 0.1,
            "thumb_joint_2": 0.1,
            "thumb_joint_3": 0.08,
        },
    ),
    actuators={
        "tiangong2pro_actuators": ImplicitActuatorCfg(
            joint_names_expr=TIANGONG2PRO_ACTUATED_JOINT_NAMES,
            effort_limit_sim=_SHARPA_EFFORT_LIMITS,
            velocity_limit_sim=_SHARPA_VELOCITY_LIMITS,
            stiffness=_SHARPA_STIFFNESS,
            damping=_SHARPA_DAMPING,
            armature=_SHARPA_ARMATURE,
            friction=_SHARPA_FRICTION,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Tiangong2Pro half-body InspireHand robot."""
