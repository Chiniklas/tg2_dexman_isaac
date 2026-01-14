# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Defines the Kuka-Inspirehand robot configuration for simulation with Isaac Sim.
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
# Update the USD path to point to the Inspirehand USD file
# kuka_inspirehand_usd_path = os.path.join(root_path, "kuka_inspirehand.usd")
# kuka_inspirehand_usd_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/kuka_inspirehand/kuka_inspirehand.usd"
# kuka_inspirehand_usd_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/kuka_inspirehand/kuka_inspirehand_legacy.usd"
kuka_inspirehand_usd_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/kuka_inspirehand/kuka_inspirehand_adjusted_thumb.usd"
KUKA_INSPIREHAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=kuka_inspirehand_usd_path,
        activate_contact_sensors=True,
        # activate_contact_sensors = False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0, # default 1000
            max_angular_velocity=500.0, # default 1000
            max_depenetration_velocity=30.0, # default 1000
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
            "iiwa7_joint_(1|2|3|4|5|6|7)": 0.,
            "index_joint_(0|1)": 0.,
            "middle_joint_(0|1)": 0.,
            "ring_joint_(0|1)": 0.,
            "little_joint_(0|1)": 0.,
            "thumb_joint_0": 0.5,
            "thumb_joint_(1|2|3)": 0.
        },
    ),
    actuators={
        "kuka_inspirehand_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_1",
                "iiwa7_joint_2",
                "iiwa7_joint_3",
                "iiwa7_joint_4",
                "iiwa7_joint_5",
                "iiwa7_joint_6",
                "iiwa7_joint_7",
                "index_joint_0",
                "middle_joint_0",
                "ring_joint_0",
                "little_joint_0",
                "thumb_joint_0",
                "thumb_joint_1",
            ],
            effort_limit_sim={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 300.,
                "index_joint_0": 1.0, # default 0.5
                "middle_joint_0": 1.0,
                "ring_joint_0": 1.0,
                "little_joint_0": 1.0,
                "thumb_joint_(0|1)": 1.0,
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4)": 300.,
                "iiwa7_joint_5": 100.,
                "iiwa7_joint_6": 50.,
                "iiwa7_joint_7": 25.,
                "index_joint_0": 300, #default 3 
                "middle_joint_0": 300,
                "ring_joint_0": 300,
                "little_joint_0": 300,
                "thumb_joint_(0|1)": 300,
            },
            damping={
                "iiwa7_joint_(1|2|3|4)": 45.,
                "iiwa7_joint_5": 20.,
                "iiwa7_joint_6": 15.,
                "iiwa7_joint_7": 15.,
                "index_joint_0": 10, # default 0.1
                "middle_joint_0": 10,
                "ring_joint_0": 10,
                "little_joint_0": 10,
                "thumb_joint_(0|1)": 10,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Kuka Inspirehand robot."""
