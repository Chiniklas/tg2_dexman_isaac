# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the TG2-Inspirehand robot configuration for simulation with Isaac Sim.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

module_path = os.path.dirname(__file__)
root_path = os.path.dirname(module_path)
# Update the USD path to point to the TG2-Inspirehand USD file
# tg2_inspirehand_usd_path = os.path.join(root_path, "tg2_inspirehand.usd")
# tg2_inspirehand_usd_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/tg2_inspirehand/tg2_inspirehand.usd"
tg2_inspirehand_usd_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/tg2_inspirehand/tg2_inspirehand_no_leg.usd"
TG2_INSPIREHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=tg2_inspirehand_usd_path,
        activate_contact_sensors=True,
        # activate_contact_sensors = False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0, # default 1000
            max_angular_velocity=500.0, # default 1000
            max_depenetration_velocity=5.0, # default 1000
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=False,
        #     solver_position_iteration_count=16,
        #     solver_velocity_iteration_count=2,
        #     sleep_threshold=0.005,
        #     stabilization_threshold=0.0005,
        # ),
        # Use position drive and mirror the primary actuator gains for faster, consistent response in GUI/Inspector.
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_(pitch|roll|yaw)_r_joint": 0.,
            "elbow_(pitch|yaw)_r_joint": 0.,
            "wrist_(pitch|roll)_r_joint": 0.,
            "index_joint_(0|1)": 0.,
            "middle_joint_(0|1)": 0.,
            "ring_joint_(0|1)": 0.,
            "little_joint_(0|1)": 0.,
            "thumb_joint_0": 0.5,
            "thumb_joint_(1|2|3)": 0.
        },
    ),
    actuators={
        "tg2_inspirehand_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_r_joint",
                "shoulder_roll_r_joint",
                "shoulder_yaw_r_joint",
                "elbow_pitch_r_joint",
                "elbow_yaw_r_joint",
                "wrist_pitch_r_joint",
                "wrist_roll_r_joint",
                "index_joint_0",
                "middle_joint_0",
                "ring_joint_0",
                "little_joint_0",
                "thumb_joint_0",
                "thumb_joint_1",
            ],
            effort_limit_sim={
                "shoulder_pitch_r_joint": 250.,
                "shoulder_roll_r_joint": 200.,
                "shoulder_yaw_r_joint": 120.,
                "elbow_pitch_r_joint": 120.,
                "elbow_yaw_r_joint": 120.,
                "wrist_pitch_r_joint": 50.,
                "wrist_roll_r_joint": 50.,
                "index_joint_0": 3.0,
                "middle_joint_0": 3.0,
                "ring_joint_0": 3.0,
                "little_joint_0": 3.0,
                "thumb_joint_0": 3.0,
                "thumb_joint_1": 3.0,
            },

            stiffness = {
                "shoulder_pitch_r_joint": 1200,
                "shoulder_roll_r_joint": 300,
                "shoulder_yaw_r_joint": 300,
                "elbow_pitch_r_joint": 300,
                "elbow_yaw_r_joint": 300,
                "wrist_pitch_r_joint": 200,
                "wrist_roll_r_joint": 200,
                "index_joint_0": 30,
                "middle_joint_0": 30,
                "ring_joint_0": 30,
                "little_joint_0": 30,
                "thumb_joint_0": 30,
                "thumb_joint_1": 30,
            },
            damping={
                "shoulder_pitch_r_joint": 60,
                "shoulder_roll_r_joint": 20,
                "shoulder_yaw_r_joint": 20,
                "elbow_pitch_r_joint": 15,
                "elbow_yaw_r_joint": 15,
                "wrist_pitch_r_joint": 8,
                "wrist_roll_r_joint": 8,
                "index_joint_0": 1.0,
                "middle_joint_0": 1.0,
                "ring_joint_0": 1.0,
                "little_joint_0": 1.0,
                "thumb_joint_0": 1.0,
                "thumb_joint_1": 1.0,
            },
        ),
    },
    # actuators = {
    #     "tg2_inspirehand_actuators": ImplicitActuatorCfg(
    #         joint_names_expr = [".*"],
    #         damping = 30,
    #         stiffness = 300,
    #         effort_limit_sim = 10,
    #     )
    # },

    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of TG2 Inspirehand robot."""
