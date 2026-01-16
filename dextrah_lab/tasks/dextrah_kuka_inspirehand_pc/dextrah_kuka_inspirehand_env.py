# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os
import pickle
import re

import numpy as np
import torch
from colorsys import hsv_to_rgb
import glob
import torch.distributed as dist
import torch.nn.functional as F
from collections.abc import Sequence
from scipy.spatial.transform import Rotation as R
import random
from pxr import Gf, UsdGeom, UsdShade, Sdf, Vt

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera, ContactSensor, ContactSensorCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaacsim.core.utils.prims import set_prim_attribute_value

from .dextrah_kuka_inspirehand_env_cfg import DextrahKukaInspirehandEnvCfg
from .dextrah_kuka_inspirehand_utils import (
    assert_equals,
    scale,
    compute_absolute_action,
    to_torch,
    quat_to_rotmat,
    print_prim_tree_once,
)

# ADR imports
from .dextrah_adr import DextrahADR

from .dextrah_kuka_inspirehand_constants import (
    NUM_XYZ,
    NUM_RPY,
    NUM_QUAT,
    NUM_HAND_PCA,
    HAND_PCA_MINS,
    HAND_PCA_MAXS,
    PALM_POSE_MINS_FUNC,
    PALM_POSE_MAXS_FUNC,
#    TABLE_LENGTH_X,
#    TABLE_LENGTH_Y,
#    TABLE_LENGTH_Z,
)

# this is for calculating the forward kinematics on the hand points
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

## TODO:
# define a palm direction vector
# add a palm direction penalty
# keep the palm to be always facing down

class _TrimeshCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "trimesh.caching" and name == "TrackedArray":
            return np.ndarray
        return super().find_class(module, name)

class DextrahKukaInspirehandEnv(DirectRLEnv):
    cfg: DextrahKukaInspirehandEnvCfg

    def __init__(self, cfg: DextrahKukaInspirehandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robot_dofs = self.robot.num_joints
        # Track whether any arm link is in contact with the table (per-env mask).
        self.arm_table_contact_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Track hand-object contact counts per env.
        self.object_contact_counts = torch.zeros(self.num_envs, device=self.device)
        # Track whether each env has made hand-object contact since reset.
        self.had_object_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.num_observations = (
            self.cfg.num_student_observations if self.cfg.distillation
            else self.cfg.num_teacher_observations
        )
        self.num_teacher_observations = self.cfg.num_teacher_observations
        self.use_camera = self.cfg.distillation
        self.simulate_stereo = self.use_camera and self.cfg.simulate_stereo
        self.stereo_baseline = self.cfg.stereo_baseline
        # Number of env steps to hold zero actions after reset so the object can settle.
        step_dt = self.cfg.sim_dt * self.cfg.decimation
        # self.action_freeze_steps = int(max(0, round(self.cfg.action_freeze_duration_s / step_dt)))
        self.action_freeze_steps = 6000
        self.no_contact_timeout_s = getattr(self.cfg, "no_contact_timeout_s", 3.0)
        self.no_contact_max_steps = max(1, int(np.ceil(self.no_contact_timeout_s / step_dt)))

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))

        # actions are 1:1 with actuated joints
        self.cfg.num_actions = len(self.actuated_dof_indices)
        self.num_actions = self.cfg.num_actions
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.action_delta = torch.zeros_like(self.actions)

        # Debug joint mapping to ensure orders align with USD
        print("[DEBUG] Robot joint order (USD):", self.robot.joint_names)
        debug_joint_map = list(zip(cfg.actuated_joint_names, self.actuated_dof_indices))
        print("[DEBUG] Actuated joints -> indices:", debug_joint_map)
        print("[DEBUG] Body names:", self.robot.body_names)
        # input("Debugging")

        # buffers for position targets
        self.robot_dof_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.dof_pos_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.dof_vel_targets =\
            torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)

        # finger bodies
        self.hand_bodies = list()
        for body_name in self.cfg.hand_body_names:
            self.hand_bodies.append(self.robot.body_names.index(body_name))
        self.hand_bodies.sort()
        self.num_hand_bodies = len(self.hand_bodies)
        # arm bodies (everything not in hand_bodies)
        self.arm_bodies = [i for i, name in enumerate(self.robot.body_names) if i not in self.hand_bodies]

        # Bodies used for hand-object distance calculation.
        self.hand_object_distance_bodies = []
        for body_name in self.cfg.hand_object_distance_body_names:
            self.hand_object_distance_bodies.append(self.robot.body_names.index(body_name))
        self.hand_object_distance_bodies.sort()
        self.num_hand_object_distance_bodies = len(self.hand_object_distance_bodies)
        self.encode_hand_object_dist = getattr(self.cfg, "encode_hand_object_dist", False)
        self._pc_obs_warned = False
        self.hand_object_pointcloud_min_dists = torch.zeros(
            self.num_envs,
            self.num_hand_object_distance_bodies,
            device=self.device,
        )
        hand_body_pc_names = self._resolve_hand_body_pc_names()
        self.hand_body_pc_indices = []
        for name in hand_body_pc_names:
            if name in self.robot.body_names:
                self.hand_body_pc_indices.append(self.robot.body_names.index(name))
        self.hand_body_pc_indices.sort()
        self.num_hand_body_pc = len(self.hand_body_pc_indices)
        self.hand_body_pc_batch_dist = torch.zeros(
            self.num_envs,
            self.num_hand_body_pc,
            device=self.device,
        )
        self.hand_object_distance_body_names = list(self.cfg.hand_object_distance_body_names)
        self._palm_pc_indices = [
            i for i, name in enumerate(self.hand_object_distance_body_names) if "palm" in name
        ]
        if not self._palm_pc_indices:
            self._palm_pc_indices = [0]
        self._finger_pc_indices = [
            i for i in range(self.num_hand_object_distance_bodies) if i not in self._palm_pc_indices
        ]
        self.hand_object_hold_value = int(getattr(self.cfg, "hand_object_hold_value", 2))
        self.hand_object_hold_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)

        #=================================================================
        # Palm body index and local axes used to compute live palm direction vectors.
        self.palm_body_idx = self.robot.body_names.index("palm")
        
        def _normalize_vector(vec: torch.Tensor) -> torch.Tensor:
            norm = torch.norm(vec)
            return vec if norm == 0 else vec / norm

        palm_down_axis = _normalize_vector(
            torch.tensor(self.cfg.palm_down_local_axis, device=self.device, dtype=torch.float)
        )
        self._palm_down_local_axis = palm_down_axis.view(1, 3, 1)
        palm_finger_axis = _normalize_vector(
            torch.tensor(self.cfg.palm_finger_local_axis, device=self.device, dtype=torch.float)
        )
        self._palm_finger_local_axis = palm_finger_axis.view(1, 3, 1)

        self.palm_direction_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_finger_direction_vec = torch.zeros(self.num_envs, 3, device=self.device)
        # Target palm direction: world -Z (points down toward the table).
        self._palm_dir_target_world = torch.tensor([0.0, 0.0, -1.0], device=self.device).view(1, 3)
        # Target finger direction: configurable world direction.
        palm_finger_target = _normalize_vector(
            torch.tensor(self.cfg.palm_finger_direction_target, device=self.device, dtype=torch.float)
        )
        self._palm_finger_target_world = palm_finger_target.view(1, 3)
        #====================================================================
        
        # joint limits
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0][:, self.actuated_dof_indices]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1][:, self.actuated_dof_indices]
        # Debug print joint limits for actuated joints
        joint_limits = list(
            zip(
                self.cfg.actuated_joint_names,
                self.robot_dof_lower_limits[0].detach().cpu().tolist(),
                self.robot_dof_upper_limits[0].detach().cpu().tolist(),
            )
        )
        # print("[DEBUG] Actuated joint limits (lower, upper):")
        # for name, lo, hi in joint_limits:
        #     print(f"  {name}: {lo:.4f}, {hi:.4f}")
        # input("debugging joint limits")

        # Setting the target position for the object
        # TODO: need to make these goals dynamic, sampled at the start of the rollout
        self.object_goal =\
            torch.tensor([-0.5, 0., 0.75], device=self.device).repeat((self.num_envs, 1))
        # self.object_goal =\
        #     torch.tensor([-0.5, 0., 0.4], device=self.device).repeat((self.num_envs, 1))
        
        # Nominal reset states for the robot
        self.robot_start_joint_pos =\
            torch.tensor([-1.2, 0.0,  0.7,  1.10, -1.55, 1.5, 0.0, # arm
                          0.25,  # index_joint_0
                          0.25,  # little_joint_0
                          0.25,  # middle_joint_0
                          0.25,  # ring_joint_0
                        #   0.0,  # thumb_joint_0
                          0.78, # thumb joint 0
                          0.386,  # index_joint_1
                          0.386,  # little_joint_1
                          0.386,  # middle_joint_1
                          0.386,  # ring_joint_1
                          0.1,  # thumb_joint_1
                          0.2,  # thumb_joint_2
                          0.4], device=self.device) # thumb_joint_3
        self.robot_start_joint_pos =\
            self.robot_start_joint_pos.repeat(self.num_envs, 1).contiguous()
        # Start with zero initial velocities and accelerations
        self.robot_start_joint_vel =\
            torch.zeros(self.num_envs, self.num_robot_dofs, device=self.device)

        # Nominal finger curled config
        # Only the actuated finger joints – matches robot_dof_pos[:, 7:]
        self.curled_q =\
            torch.tensor([0.25,  # index_joint_0
                          0.25,  # little_joint_0
                          0.25,  # middle_joint_0
                          0.25,  # ring_joint_0
                        #   0.0,  # thumb_joint_0
                          0.78,  # thumb_joint_0
                          0.1], device=self.device)  # thumb_joint_1
        self.curled_q = self.curled_q.repeat(self.num_envs, 1).contiguous()

        # Set up ADR
        self.dextrah_adr =\
            DextrahADR(self.event_manager, self.cfg.adr_cfg_dict, self.cfg.adr_custom_cfg_dict)
        self.step_since_last_dr_change = 0
        if self.cfg.distillation:
            self.cfg.starting_adr_increments = self.cfg.num_adr_increments
        self.dextrah_adr.set_num_increments(self.cfg.starting_adr_increments)
        self.local_adr_increment = torch.tensor(
            self.cfg.starting_adr_increments,
            device=self.device,
            dtype=torch.int64
        )
        # The global minimum adr increment across all GPUs. initialized to the starting adr
        self.global_min_adr_increment = self.local_adr_increment.clone()

        # PD action buffers
        self.joint_position_targets = torch.clone(self.robot_start_joint_pos)
        self.joint_velocity_targets = torch.zeros_like(self.robot_start_joint_vel)

        # Preallocate some reward related signals
        self.hand_to_object_pos_error = torch.ones(self.num_envs, device=self.device) 

        # Track success statistics
        self.in_success_region = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_in_success_region = torch.zeros(self.num_envs, device=self.device)
        
        # Unit tensors - used in creating random object rotations during spawn
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Wrench tensors
        self.object_applied_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object_applied_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Object noise
        self.object_pos_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.object_rot_noise_width = torch.zeros(self.num_envs, 1, device=self.device)

        # Robot noise
        self.robot_joint_pos_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_bias_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_pos_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_bias = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_pos_noise_width = torch.zeros(self.num_envs, 1, device=self.device)
        self.robot_joint_vel_noise_width = torch.zeros(self.num_envs, 1, device=self.device)

        # For querying 3D points on hand
        # this is a fabric based forward kinematics helper
        # the purpose is to use forward kinematics to generate a noisy fingertip and palm position and vel
        # the noisy pos and vel will be compared with pure pos and vel
        self.urdf_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/kuka_inspirehand/urdf/kuka_inspirehand_test.urdf"
        self.hand_points_taskmap = RobotFrameOriginsTaskMap(self.urdf_path, self.cfg.hand_body_names,
                                                            self.num_envs, self.device)

        # markers
        self.pred_pos_markers = VisualizationMarkers(
            self.cfg.pred_pos_marker_cfg
        )
        self.gt_pos_markers = VisualizationMarkers(
            self.cfg.gt_pos_marker_cfg
        )

        # How many steps to print reward breakdowns for (debugging aid).
        self._reward_debug_steps_remaining = getattr(self.cfg, "debug_reward_steps", 0)

        # original camera poses
        self.camera_pos_orig = torch.tensor(
            self.cfg.camera_pos
        ).to(self.device).unsqueeze(0)
        self.camera_rot_orig = np.array(self.cfg.camera_rot)
        self.camera_rot_eul_orig = R.from_quat(
            self.camera_rot_orig[[1, 2, 3, 0]]
        ).as_euler('xyz', degrees=True)[None, :]
        tf = np.array([
            7.416679444534866883e-02,-9.902696855667120213e-01,1.177507386359286923e-01,-7.236400044878017468e-01,
            -1.274026398887237732e-01,1.076995435286611930e-01,9.859864987275952508e-01,-6.886495877727516479e-01,
            -9.890742408692511090e-01,-8.812921292808308105e-02,-1.181752422362273985e-01,6.366771698474239516e-01,
            0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00
        ]).reshape(4,4)
        self.camera_pose = np.tile(
            tf, (self.num_envs, 1, 1)
        )
        self.right_to_left_pose = np.array([
            [-1., 0., 0., 0.065],
            [0., -1., 0., -0.062],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])

        self.camera_right_pos_orig = torch.tensor(
            self.right_to_left_pose[:3, 3]
        ).to(self.device).unsqueeze(0)
        self.camera_right_rot_orig = R.from_matrix(
            self.right_to_left_pose[:3, :3]
        ).as_quat()
        self.camera_right_rot_eul_orig = R.from_quat(
            self.camera_right_rot_orig
        ).as_euler('xyz', degrees=True)[None, :]
        self.camera_right_pose = np.tile(
            self.right_to_left_pose, (self.num_envs, 1, 1)
        )
        self.intrinsic_matrix = torch.tensor(
            self.cfg.intrinsic_matrix,
            device=self.device, dtype=torch.float64
        )
        self.left_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self.left_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self.right_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self.right_rot = torch.zeros(self.num_envs, 4).to(self.device)

        # Set the starting default joint friction coefficients
        friction_coeff = torch.tensor(self.cfg.starting_robot_dof_friction_coefficients,
                                      device=self.device)
        friction_coeff = friction_coeff.repeat((self.num_envs, 1))
        self.robot.data.default_joint_friction_coeff = friction_coeff

        # input("end of init")

    def find_num_unique_objects(self, objects_dir):
        module_path = os.path.dirname(__file__)
        root_path = os.path.dirname(os.path.dirname(module_path))
        scene_objects_usd_path = os.path.join(root_path, "assets/")

        objects_full_path = scene_objects_usd_path + objects_dir + "/USD"

        # List all subdirectories in the target directory
        sub_dirs = sorted(os.listdir(objects_full_path))

        # Filter out all subdirectories deeper than one level
        sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(
            os.path.join(objects_full_path, object_name))]

        num_unique_objects = len(sub_dirs)

        return num_unique_objects

    def _setup_policy_params(self):
        # Determine number of unique objects in target object dir
        if self.cfg.objects_dir not in self.cfg.valid_objects_dir:
            raise ValueError(f"Need to specify valid directory of objects for training: {self.cfg.valid_objects_dir}")

        num_unique_objects = self.find_num_unique_objects(self.cfg.objects_dir)

        # Hardcode observation sizes (base + num_unique_objects), similar to upstream pattern
        # num_actuated = 13, num_hand_bodies = 6 -> student obs = 96
        # encode hand to object pointcloud distances if enabled
        encode_hand_object_dist = getattr(self.cfg, "encode_hand_object_dist", False)
        pc_dist_dim = len(self.cfg.hand_body_pc_names) if (encode_hand_object_dist and self.cfg.hand_body_pc_names) else (
            len(self.cfg.hand_body_names) if encode_hand_object_dist else 0
        )
        # Student: base 91 plus num_unique_objects + pc_dist_dim
        self.cfg.num_student_observations = 78 + pc_dist_dim
        
        # Teacher: base 104 plus num_unique_objects + pc_dist_dim
        self.cfg.num_teacher_observations = 86 + num_unique_objects + pc_dist_dim
        if self.cfg.distillation:
            self.cfg.num_observations = self.cfg.num_student_observations
        else:
            self.cfg.num_observations = self.cfg.num_teacher_observations
        
        # Critic: base 150 plus num_unique_objects + pc_dist_dim
        self.cfg.num_states = 132 + num_unique_objects

        self.cfg.state_space = self.cfg.num_states
        self.cfg.observation_space = self.cfg.num_observations
        self.cfg.action_space = self.cfg.num_actions

    def _set_pos_marker(self, pos):
        pos = pos + self.scene.env_origins
        self.pred_pos_markers.visualize(pos, self.object_rot)
    
    def _set_gt_pos_marker(self, pos):
        pos = pos + self.scene.env_origins
        self.gt_pos_markers.visualize(pos, self.object_rot)

    def _resolve_hand_body_pc_names(self):
        # derive hand body pointcloud names to log to observation
        cfg_names = getattr(self.cfg, "hand_body_pc_names", None)
        if cfg_names:
            names = list(cfg_names)
        self._hand_body_pc_names = names
        return names

    def _setup_scene(self):
        # add robot, objects 
        # TODO: add goal objects?
        self.robot = Articulation(self.cfg.robot_cfg)
        
        self.table = RigidObject(self.cfg.table_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=True)

        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        
        ## add contact sensors
        # object hand contact sensors
        self.object_contact_sensors = []
        self.object_contact_links = []
        object_contact_cfgs = [
            ("palm", self.cfg.palm_object_contact_sensor),
            ("index_link_0", self.cfg.index0_object_contact_sensor),
            ("middle_link_0", self.cfg.middle0_object_contact_sensor),
            ("ring_link_0", self.cfg.ring0_object_contact_sensor),
            ("little_link_0", self.cfg.little0_object_contact_sensor),
            ("index_link_1", self.cfg.index1_object_contact_sensor),
            ("middle_link_1", self.cfg.middle1_object_contact_sensor),
            ("ring_link_1", self.cfg.ring1_object_contact_sensor),
            ("little_link_1", self.cfg.little1_object_contact_sensor),
            ("thumb_link_0", self.cfg.thumb0_object_contact_sensor),
            ("thumb_link_1", self.cfg.thumb1_object_contact_sensor),
            ("thumb_link_2", self.cfg.thumb2_object_contact_sensor),
            ("thumb_link_3", self.cfg.thumb3_object_contact_sensor),
        ]
        for link_name, sensor_cfg in object_contact_cfgs:
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors[f"object_contact_sensor_{link_name}"] = sensor
            self.object_contact_sensors.append(sensor)
            self.object_contact_links.append(link_name)
        
        # table arm contact sensors
        self.table_contact_sensors = []
        self.table_contact_links = []
        table_contact_cfgs = [
            ("iiwa7_link_1", self.cfg.iiwa7_link_1_table_contact_sensor),
            ("iiwa7_link_2", self.cfg.iiwa7_link_2_table_contact_sensor),
            ("iiwa7_link_3", self.cfg.iiwa7_link_3_table_contact_sensor),
            ("iiwa7_link_4", self.cfg.iiwa7_link_4_table_contact_sensor),
            ("iiwa7_link_5", self.cfg.iiwa7_link_5_table_contact_sensor),
            ("iiwa7_link_6", self.cfg.iiwa7_link_6_table_contact_sensor),
            ("iiwa7_link_7", self.cfg.iiwa7_link_7_table_contact_sensor),
        ]
        for link_name, sensor_cfg in table_contact_cfgs:
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors[f"table_contact_sensor_{link_name}"] = sensor
            self.table_contact_sensors.append(sensor)
            self.table_contact_links.append(link_name)
        # print("current contact links = ", self.table_contact_links)
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add cameras
        if self.cfg.distillation:
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

        # Determine obs sizes for policies and VF
        self._setup_policy_params()

        # Create the objects for grasping
        self._setup_objects()
        if self.cfg.distillation:
            import omni.replicator.core as rep
            rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
            table_texture_dir = self.cfg.table_texture_dir
            self.table_texture_files = glob.glob(
                os.path.join(table_texture_dir, "*.png")
            )
            self.stage = omni.usd.get_context().get_stage()

            if not self.cfg.disable_dome_light_randomization:
                dome_light_dir = self.cfg.dome_light_dir
                self.dome_light_files = sorted(glob.glob(
                    os.path.join(dome_light_dir, "*.exr")
                ))
                dome_light_texture = random.choice(self.dome_light_files)
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:texture:file"
                ).Set(dome_light_texture)
            else:
                print("Disabling dome light random initialization")

            UsdGeom.Imageable(
                self.stage.GetPrimAtPath("/World/ground")
            ).MakeInvisible()

            self.object_textures = glob.glob(
                os.path.join(
                    self.cfg.metropolis_asset_dir,
                    "**", "*.png"
                ), recursive=True
            )
            try:
                UsdGeom.Imageable(
                    self.stage.GetPrimAtPath("/Environment/defaultLight")
                ).MakeInvisible()
            except:
                pass

    def _load_object_pointcloud(self, objects_urdf_path, object_name, target_count=None):
        if target_count is None:
            target_count = self.cfg.object_pointcloud_num_points

        if not hasattr(self, "_object_pointcloud_cache"):
            self._object_pointcloud_cache = {}
        cache_key = (object_name, int(target_count))
        if cache_key in self._object_pointcloud_cache:
            return self._object_pointcloud_cache[cache_key]

        cloud_dir = os.path.join(objects_urdf_path, object_name)
        cloud_path = os.path.join(cloud_dir, f"point_cloud_{target_count}_pts.pkl")
        if not os.path.isfile(cloud_path):
            candidates = []
            for path in glob.glob(os.path.join(cloud_dir, "point_cloud_*_pts.pkl")):
                match = re.search(r"point_cloud_(\d+)_pts\.pkl", os.path.basename(path))
                if match:
                    candidates.append((int(match.group(1)), path))
            if not candidates:
                print(f"[WARN] No point cloud files found for {object_name} in {cloud_dir}")
                self._object_pointcloud_cache[cache_key] = None
                return None
            cloud_path = max(candidates, key=lambda item: item[0])[1]

        try:
            with open(cloud_path, "rb") as f:
                points = _TrimeshCompatUnpickler(f).load()
        except Exception as exc:
            print(f"[WARN] Failed to load point cloud for {object_name} ({cloud_path}): {exc}")
            self._object_pointcloud_cache[cache_key] = None
            return None

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"[WARN] Invalid point cloud shape for {object_name}: {points.shape}")
            self._object_pointcloud_cache[cache_key] = None
            return None
        if points.shape[0] > target_count:
            points = points[:target_count]
        elif points.shape[0] < target_count:
            repeat = int(np.ceil(target_count / points.shape[0]))
            points = np.tile(points, (repeat, 1))[:target_count]

        self._object_pointcloud_cache[cache_key] = points
        return points

    def _add_object_pointcloud_overlay(self, stage, object_prim_path, points):
        base_link_path = f"{object_prim_path}/baseLink"
        parent_path = base_link_path if stage.GetPrimAtPath(base_link_path).IsValid() else object_prim_path
        points_path = f"{parent_path}/point_cloud"
        if stage.GetPrimAtPath(points_path).IsValid():
            return

        points_geom = UsdGeom.Points.Define(stage, points_path)
        point_list = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]
        points_geom.CreatePointsAttr().Set(Vt.Vec3fArray(point_list))

        point_size = float(self.cfg.object_pointcloud_marker_size)
        points_geom.CreateWidthsAttr().Set(Vt.FloatArray([point_size] * len(point_list)))

        color = self.cfg.object_pointcloud_color
        color_vec = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        points_geom.CreateDisplayColorAttr().Set(Vt.Vec3fArray([color_vec] * len(point_list)))
        if hasattr(self.cfg, "object_pointcloud_opacity"):
            opacity = float(self.cfg.object_pointcloud_opacity)
            points_geom.CreateDisplayOpacityAttr().Set(Vt.FloatArray([opacity] * len(point_list)))

    def _setup_objects(self):
        module_path = os.path.dirname(__file__)
        root_path = os.path.dirname(os.path.dirname(module_path))
        scene_objects_usd_path = os.path.join(root_path, "assets/")

        objects_full_path = scene_objects_usd_path + self.cfg.objects_dir + "/USD"
        objects_urdf_path = os.path.join(root_path, "assets", self.cfg.objects_dir, "urdf")

        # List all subdirectories in the target directory
        sub_dirs = sorted(os.listdir(objects_full_path))

        # Filter out all subdirectories deeper than one level
        sub_dirs = [object_name for object_name in sub_dirs if os.path.isdir(
            os.path.join(objects_full_path, object_name))]

        self.num_unique_objects = len(sub_dirs)
        self.multi_object_idx =\
            torch.remainder(torch.arange(self.num_envs), self.num_unique_objects).to(self.device)

        # Create one-hot encoding of object ID for usage as feature input
        self.multi_object_idx_onehot = F.one_hot(
            self.multi_object_idx, num_classes=self.num_unique_objects).float()

        stage = omni.usd.get_context().get_stage()
        self.object_mat_prims = list()
        self.arm_mat_prims = list()
        # Tensor of scales applied to each object. Setup to do this deterministically...
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
        state = torch.get_rng_state() # get the hidden rng state of torch
        torch.manual_seed(42) # set the rng seed
        scale_range = self.cfg.object_scale_max - self.cfg.object_scale_min
        self.total_object_scales = scale_range * torch.rand(total_gpus * self.num_envs, 1, device=self.device) +\
            self.cfg.object_scale_min
        torch.set_rng_state(state) # reset the rng state of torch

        self.device_index = self.total_object_scales.device.index
        self.object_scale = self.total_object_scales[self.device_index * self.num_envs :
                                                     (self.device_index + 1) * self.num_envs]

        # If object scaling is deactivated, then just set all the scalings to 1.
        if self.cfg.deactivate_object_scaling:
            self.object_scale = torch.ones_like(self.object_scale)

        self.use_pointcloud_hand_object_dist = getattr(self.cfg, "use_pointcloud_hand_object_dist", False)
        need_pointclouds = self.use_pointcloud_hand_object_dist or self.encode_hand_object_dist
        object_pointclouds_local = []
        pointcloud_dist_enabled = need_pointclouds

        for i in range(self.num_envs):
            # TODO: check to see that the below config settings make sense
            object_name = sub_dirs[self.multi_object_idx[i]]
            object_usd_path = objects_full_path + "/" + object_name + "/" + object_name + ".usd"
            print('Object name', object_name)
            print('object usd path', object_usd_path)

            object_prim_name = "object_" + str(i) + "_" + object_name
            prim_path = "/World/envs/" + "env_" + str(i) + "/object/" + object_prim_name
            print('Object prim name', object_prim_name)
            print('Object prim path', prim_path)

            print('Object Scale', self.object_scale[i])

            object_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_usd_path,
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
                    # scale = (10,10,10),
                    scale=(self.object_scale[i],
                           self.object_scale[i],
                           self.object_scale[i]), #default
                    mass_props=sim_utils.MassPropertiesCfg(density=500.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(-0.5, 0., 0.5),
                    rot=(1.0, 0.0, 0.0, 0.0)),
            )
            # add object to scene
            object_for_grasping = RigidObject(object_cfg)

            # remove baseLink
            set_prim_attribute_value(
                prim_path=prim_path+"/baseLink",
                attribute_name="physxArticulation:articulationEnabled",
                value=False
            )

            if need_pointclouds:
                points = self._load_object_pointcloud(
                    objects_urdf_path,
                    object_name,
                    target_count=self.cfg.hand_object_pointcloud_num_points,
                )
                if points is None:
                    pointcloud_dist_enabled = False
                else:
                    object_pointclouds_local.append(points)

            if self.cfg.enable_object_pointcloud_overlay:
                overlay_env_ids = getattr(self.cfg, "pointcloud_overlay_env_ids", None)
                overlay_enabled = True if overlay_env_ids is None else i in overlay_env_ids
                if overlay_enabled:
                    points = self._load_object_pointcloud(objects_urdf_path, object_name)
                    if points is not None:
                        self._add_object_pointcloud_overlay(stage, prim_path, points)

            # Get shaders
            prim = stage.GetPrimAtPath(prim_path)
            self.object_mat_prims.append(prim.GetChildren()[0].GetChildren()[0].GetChildren()[0])

            arm_shader_prims = list()
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/arm_gray/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/arm_orange/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/allegro_black/Shader"
                )
            )
            arm_shader_prims.append(
                stage.GetPrimAtPath(
                    "/World/envs/" + "env_" + str(i) + "/Robot/Looks/allegro_biotac/Shader"
                )
            )
            self.arm_mat_prims.append(arm_shader_prims)

        if need_pointclouds and pointcloud_dist_enabled:
            if len(object_pointclouds_local) != self.num_envs:
                print("[WARN] Point cloud distance disabled: missing per-env point clouds.")
                self.use_pointcloud_hand_object_dist = False
                self.object_pointclouds_local = None
            else:
                points_local = torch.tensor(
                    np.stack(object_pointclouds_local, axis=0),
                    device=self.device,
                    dtype=torch.float,
                )
                self.object_pointclouds_local = points_local * self.object_scale.unsqueeze(-1)
        else:
            self.use_pointcloud_hand_object_dist = False
            self.object_pointclouds_local = None
        # Now create one more RigidObject with regex on existing object prims
        # so that we can add all the above objects into one RigidObject object
        # for batch querying their states, forces, etc.
        regex = "/World/envs/env_.*/object/.*"
        multi_object_cfg = RigidObjectCfg(
            prim_path=regex,
            spawn=None,
        )

        # Add to scene
        self.object = RigidObject(multi_object_cfg)
        self.scene.rigid_objects["object"] = self.object

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # print_prim_tree_once(self)
        # Find the current global minimum adr increment
        local_adr_increment = self.local_adr_increment.clone()
        # Query for the global minimum adr increment across all GPUs
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.all_reduce(local_adr_increment, op=dist.ReduceOp.MIN)
        self.global_min_adr_increment = local_adr_increment

        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # Map actions to joint position/velocity targets
        self.compute_actions(self.actions)

        # Add F/T wrench to object
        self.apply_object_wrench()

    def _apply_action(self) -> None:
        # Set position target
        self.robot.set_joint_position_target(
            self.dof_pos_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        # Set velocity target
        vel_scale = self.dextrah_adr.get_custom_param_value(
            "pd_targets", "velocity_target_factor"
        )
        self.robot.set_joint_velocity_target(
            vel_scale * self.dof_vel_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        policy_obs = self.compute_policy_observations()
        critic_obs = self.compute_critic_observations()

        if self.use_camera and not self.simulate_stereo:
            depth_map = self._tiled_camera.data.output["depth"].clone()
            mask = depth_map.permute((0, 3, 1, 2)) > self.cfg.d_max
            depth_map[depth_map <= 1e-8] = 10
            depth_map[depth_map > self.cfg.d_max] = 0.
            depth_map[depth_map < self.cfg.d_min] = 0.

            student_policy_obs = self.compute_student_policy_observations()
            teacher_policy_obs = self.compute_policy_observations()
            critic_obs = self.compute_critic_observations()

            aux_info = {
                "object_pos": self.object_pos
            }

            observations = {
                "policy": student_policy_obs,
                # "policy": teacher_policy_obs,
                "img": depth_map.permute((0, 3, 1, 2)),
                "rgb": self._tiled_camera.data.output["rgb"].clone().permute((0, 3, 1, 2)) / 255.,
                "expert_policy": teacher_policy_obs,
                "critic": critic_obs,
                "aux_info": aux_info,
                "mask": mask
            }
        elif self.simulate_stereo:
            left_rgb = self._tiled_camera.data.output["rgb"].clone() / 255.
            left_depth = self._tiled_camera.data.output["depth"].clone()
            left_mask = left_depth > self.cfg.d_max*10
            left_depth[left_depth <= 1e-8] = 10
            left_depth[left_depth > self.cfg.d_max] = 0.
            left_depth[left_depth < self.cfg.d_min] = 0.
            right_to_world = torch.from_numpy(
                np.matmul(self.camera_pose, self.camera_right_pose)
            ).to(self.device)
            right_to_world_rot = torch.tensor(R.from_matrix(
                right_to_world[:, :3, :3].cpu().numpy()
            ).as_quat()[:, [3, 0, 1, 2]]).to(self.device)
            self._tiled_camera.set_world_poses(
                positions=right_to_world[:, :3, 3],
                orientations=right_to_world_rot,
                env_ids=self.robot._ALL_INDICES,
                convention="ros"
            )
            self.sim.render()
            self._tiled_camera.update(0, force_recompute=True)
            object_pos_world = torch.cat(
                [
                    self.object_pos,
                    torch.ones(
                        self.object_pos.shape[0], 1,
                        device=self.device,
                        dtype=right_to_world.dtype
                    )
                ], dim=-1
            )
            # (N, 4, 4)
            T_right_world = torch.eye(4, device=self.device, dtype=right_to_world.dtype).unsqueeze(0).repeat(
                self.num_envs, 1, 1
            )
            T_right_world[:, :3, :3] = right_to_world[:, :3, :3].transpose(1, 2)
            T_right_world[:, :3, 3] = torch.bmm(
                -T_right_world[:, :3, :3],
                right_to_world[:, :3, 3:4] - self.scene.env_origins.unsqueeze(-1)
            ).squeeze(-1)
            obj_pos_right = torch.bmm(
                T_right_world,
                object_pos_world.unsqueeze(-1)
            )[:, :3, :]
            obj_uv_right = torch.matmul(
                self.intrinsic_matrix,
                obj_pos_right
            ).squeeze(-1)
            obj_uv_right[:, :2] /= obj_uv_right[:, 2:3]
            # right image is flipped
            obj_uv_right[:, 0] = self.cfg.img_width - obj_uv_right[:, 0]
            obj_uv_right[:, 1] = self.cfg.img_height - obj_uv_right[:, 1]


            # self.sim.render()
            # self._tiled_camera.update(0, force_recompute=True)
            right_rgb = self._tiled_camera.data.output["rgb"].clone() / 255.
            right_depth = self._tiled_camera.data.output["depth"].clone()
            right_mask = right_depth > self.cfg.d_max*10
            right_depth[right_depth <= 1e-8] = 10
            right_depth[right_depth > self.cfg.d_max] = 0.
            right_depth[right_depth < self.cfg.d_min] = 0.
            self._tiled_camera.set_world_poses(
                positions=self.left_pos,
                orientations=self.left_rot,
                env_ids=self.robot._ALL_INDICES,
                convention="ros"
            )

            object_pos_world = torch.cat(
                [
                    self.object_pos,
                    torch.ones(
                        self.object_pos.shape[0], 1,
                        device=self.device,
                        dtype=right_to_world.dtype
                    )
                ], dim=-1
            )
            T_left_world = torch.eye(4, device=self.device, dtype=right_to_world.dtype).unsqueeze(0).repeat(
                self.num_envs, 1, 1
            )
            T_left_world[:, :3, :3] = torch.from_numpy(self.camera_pose[:, :3, :3]).to(self.device).transpose(1, 2)
            T_left_world[:, :3, 3] = torch.bmm(
                -T_left_world[:, :3, :3],
                torch.from_numpy(self.camera_pose[:, :3, 3:4]).to(self.device) - self.scene.env_origins.unsqueeze(-1)
            ).squeeze(-1)
            obj_pos_left = torch.bmm(
                T_left_world,
                object_pos_world.unsqueeze(-1)
            )[:, :3, :]
            obj_uv_left = torch.matmul(
                self.intrinsic_matrix,
                obj_pos_left
            ).squeeze(-1)
            obj_uv_left[:, :2] /= obj_uv_left[:, 2:3]
            # from PIL import Image
            # img_np_left = left_rgb.cpu().numpy()
            # img_np_right = right_rgb.cpu().numpy()
            # im_left = Image.fromarray(img_np_left[0])
            # im_right = Image.fromarray(img_np_right[0])
            student_policy_obs = self.compute_student_policy_observations()
            teacher_policy_obs = self.compute_policy_observations()
            critic_obs = self.compute_critic_observations()

            # normalize uvs by img dims
            obj_uv_left[:, 0] /= self.cfg.img_width
            obj_uv_left[:, 1] /= self.cfg.img_height
            obj_uv_right[:, 0] /= self.cfg.img_width
            obj_uv_right[:, 1] /= self.cfg.img_height

            aux_info = {
                "object_pos": self.object_pos,
                "left_img_depth": left_depth.permute((0, 3, 1, 2)),
                "obj_uv_left": obj_uv_left[:, :2],
                "obj_uv_right": obj_uv_right[:, :2],
            }

            observations = {
                "policy": student_policy_obs,
                "depth_left": left_depth.permute((0, 3, 1, 2)),
                "depth_right": right_depth.permute((0, 3, 1, 2)),
                "mask_left": left_mask.permute((0, 3, 1, 2)),
                "mask_right": right_mask.permute((0, 3, 1, 2)),
                "img_left": left_rgb.permute((0, 3, 1, 2)),
                "img_right": right_rgb.permute((0, 3, 1, 2)),
                "expert_policy": teacher_policy_obs,
                "critic": critic_obs,
                "aux_info": aux_info,
                "obj_uv_left": obj_uv_left[:, :2],
                "obj_uv_right": obj_uv_right[:, :2],
            }
        else:
            observations = {"policy": policy_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Update signals related to reward
        self.compute_intermediate_reward_values()

        (
            hand_to_object_reward,
            object_to_goal_reward,
            finger_curl_reg_raw,
            lift_reward,
            palm_direction_alignment_reward,
            palm_finger_alignment_reward,
            contact_reward,
            joint_vel_penalty,
            action_rate_penalty,
        ) = compute_rewards(
                self.reset_buf,
                self.in_success_region,
                self.max_episode_length,
                self.hand_to_object_pos_error,
                self.hand_object_palm_dist,
                self.hand_object_finger_dist,
                self.object_to_object_goal_pos_error,
                self.object_vertical_error,
                self.robot_dof_pos[:, 7:], # NOTE: only the finger joints
                self.curled_q,
                self.cfg.hand_to_object_weight,
                self.cfg.hand_to_object_sharpness,
                self.cfg.hand_object_palm_dist_weight,
                self.cfg.hand_object_finger_dist_weight,
                self.cfg.object_to_goal_weight,
                self.dextrah_adr.get_custom_param_value("reward_weights", "object_to_goal_sharpness"),
                self.dextrah_adr.get_custom_param_value("reward_weights", "finger_curl_reg"),
                self.dextrah_adr.get_custom_param_value("reward_weights", "lift_weight"),
                self.cfg.lift_sharpness,
                self.cfg.palm_direction_alignment_weight,  
                self.palm_direction_vec,  
                self._palm_dir_target_world.expand_as(self.palm_direction_vec),  # world -Z target
                self.cfg.palm_finger_alignment_weight,
                self.palm_finger_direction_vec,
                self._palm_finger_target_world.expand_as(self.palm_finger_direction_vec),
                self.object_contact_counts,
                self.cfg.hand_object_contact_weight,
                self.robot_dof_vel,
                self.cfg.joint_velocity_penalty_weight,
                self.action_delta,
                self.cfg.action_rate_penalty_weight,
                self.cfg.hand_joint_velocity_penalty_scale,
                self.cfg.hand_action_rate_penalty_scale,
            )

        # Original curl reward (ADR-weighted).
        # finger_curl_reg = torch.clamp(
        #     finger_curl_reg_raw,
        #     min=self.cfg.finger_curl_reg_min,
        #     max=self.cfg.finger_curl_reg_max,
        # )

        finger_curl_dist = (self.robot_dof_pos[:, 7:] - self.curled_q).norm(p=2, dim=-1)
        curl_weight_far = self.cfg.finger_curl_reg_weight_far
        curl_weight_near = self.cfg.finger_curl_reg_weight_near
        curl_switch_dist = self.cfg.finger_curl_switch_dist
        curl_weight = torch.where(
            self.hand_to_object_pos_error < curl_switch_dist,
            torch.full_like(finger_curl_dist, curl_weight_near),
            torch.full_like(finger_curl_dist, curl_weight_far),
        )
        finger_curl_reg = curl_weight * finger_curl_dist ** 2
        finger_curl_reg = torch.clamp(
            finger_curl_reg,
            min=self.cfg.finger_curl_reg_min,
            max=self.cfg.finger_curl_reg_max,
        )

        min_steps = getattr(self.cfg, "min_num_episode_steps", 0)
        if min_steps > 0:
            lift_reward = torch.where(
                self.episode_length_buf >= min_steps,
                lift_reward,
                torch.zeros_like(lift_reward),
            )

        # give episode length reward to let the robot survice around the grasping position
        episode_length_weight = getattr(self.cfg, "episode_length_reward_weight", 0.0)
        episode_length_reward = episode_length_weight * self.episode_length_buf.to(
            hand_to_object_reward.dtype
        )
        # Gate survival reward until the hand is within 0.3m of the object.
        gate_dist = getattr(self.cfg, "episode_length_gate_dist", 0.3)
        episode_length_reward = torch.where(
            self.hand_to_object_pos_error < gate_dist,
            episode_length_reward,
            torch.zeros_like(episode_length_reward),
        )

        # palm velocity penalty
        palm_lin_vel_weight = getattr(self.cfg, "palm_linear_velocity_penalty_weight", 0.0)
        palm_lin_vel = self.robot.data.body_vel_w[:, self.palm_body_idx, :3] 
        palm_lin_vel_penalty = -5e-4 * (palm_lin_vel ** 2).sum(dim=-1)

        lift_success = (self.object_contact_counts > 0.0) & (
            self.object_pos[:, 2] > self.cfg.object_height_thresh
        )

        # Add reward signals to tensorboard
        self.extras["hand_to_object_reward"] = hand_to_object_reward.mean()
        self.extras["object_to_goal_reward"] = object_to_goal_reward.mean()
        self.extras["finger_curl_reg"] = finger_curl_reg.mean()
        self.extras["lift_reward"] = lift_reward.mean()
        self.extras["lift_success"] = lift_success.float().mean()
        self.extras["hand_object_contact_reward"] = contact_reward.mean()
        self.extras["palm_direction_alignment_reward"] = palm_direction_alignment_reward.mean()
        self.extras["palm_finger_alignment_reward"] = palm_finger_alignment_reward.mean()
        self.extras["joint_velocity_penalty"] = joint_vel_penalty.mean()
        self.extras["action_rate_penalty"] = action_rate_penalty.mean()
        self.extras["episode_length_reward"] = episode_length_reward.mean()
        self.extras["palm_linear_velocity_penalty"] = palm_lin_vel_penalty.mean()

        total_reward = (
            hand_to_object_reward
            # + object_to_goal_reward
            # + finger_curl_reg
            # + lift_reward
            + palm_direction_alignment_reward
            + palm_finger_alignment_reward
            # + contact_reward
            + action_rate_penalty
            # + episode_length_reward
            # + palm_lin_vel_penalty
            # + joint_vel_penalty
        )

        # Optional reward debug printout for the first N steps.
        if self._reward_debug_steps_remaining < 0:
            step_id = int(self.episode_length_buf.max().item())
            print(
                f"[REWARD DEBUG] step={step_id} "
                f"total_mean={total_reward.mean().item():.4f} "
                f"hand_obj={hand_to_object_reward.mean().item():.4f} "
                f"obj_goal={object_to_goal_reward.mean().item():.4f} "
                f"lift={lift_reward.mean().item():.4f} "
                f"palm_align={palm_direction_alignment_reward.mean().item():.4f} "
                f"palm_finger_align={palm_finger_alignment_reward.mean().item():.4f} "
                f"contact={contact_reward.mean().item():.4f} "
                f"finger_curl={finger_curl_reg.mean().item():.4f} "
                f"joint_vel_pen={joint_vel_penalty.mean().item():.4f} "
                f"action_rate_pen={action_rate_penalty.mean().item():.4f} "
                f"ep_len={episode_length_reward.mean().item():.4f} "
                f"palm_lin_vel_pen={palm_lin_vel_penalty.mean().item():.4f} "
                f"vel_max={self.robot_dof_vel.abs().max().item():.3f} "
                f"action_delta_max={self.action_delta.abs().max().item():.3f}"
            )
            self._reward_debug_steps_remaining -= 1

        # Log other information
        self.extras["num_adr_increases"] = self.dextrah_adr.num_increments()
        self.extras["in_success_region"] = self.in_success_region.float().mean()

        # print('reach reward', hand_to_object_reward.mean())
        # print('lift reward', lift_reward.mean())

        return total_reward

    def _get_dones(self) -> torch.Tensor:
        # This should be in start
        self._compute_intermediate_values()

        # Determine if the object is out of reach by checking XYZ position
        # XY should be within certain limits on the table to stay in the workspace

        # If Z is too low, then it has probably fallen off
        object_outside_upper_x = self.object_pos[:,0] > (self.cfg.x_center + self.cfg.x_width / 2.)
        object_outside_lower_x = self.object_pos[:,0] < (self.cfg.x_center - self.cfg.x_width / 2.)

        object_outside_upper_y = self.object_pos[:,1] > (self.cfg.y_center + self.cfg.y_width / 2.)
        object_outside_lower_y = self.object_pos[:,1] < (self.cfg.y_center - self.cfg.y_width / 2.)

        z_height_cutoff = 0.2
        object_too_low = self.object_pos[:,2] < z_height_cutoff

        bbox_margin = getattr(self.cfg, "hand_bbox_margin", 0.0)
        table_half_x = self.cfg.table_size_x * 0.5 + bbox_margin
        table_half_y = self.cfg.table_size_y * 0.5 + bbox_margin
        table_top_z = self.table_pos_z + self.cfg.table_size_z * 0.5

        # Hand termination: keep palm origin inside a box above the table surface
        palm_pos = self.robot.data.body_pos_w[:, self.palm_body_idx] - self.scene.env_origins
        palm_x_out = (palm_pos[:, 0] > (self.table_pos[:, 0] + table_half_x)) | \
                     (palm_pos[:, 0] < (self.table_pos[:, 0] - table_half_x))
        palm_y_out = (palm_pos[:, 1] > (self.table_pos[:, 1] + table_half_y)) | \
                     (palm_pos[:, 1] < (self.table_pos[:, 1] - table_half_y))
        # Constrain palm height to remain between the table surface and a band above it
        palm_z_out = (palm_pos[:, 2] < table_top_z) | (palm_pos[:, 2] > (table_top_z + 1.0 + bbox_margin))
        hand_too_far = palm_x_out | palm_y_out | palm_z_out

        # Hand termination: any palm/fingertip point closer than 2 cm to table surface
        hand_min_z = self.hand_pos[..., 2].min(dim=1).values
        clearance_thresh = table_top_z + 0.01  # 2 cm above table
        hand_too_close = hand_min_z < clearance_thresh

        # Palm flip termination: if palm direction deviates too far from -Z target.
        palm_flip_cos = torch.sum(self.palm_direction_vec * self._palm_dir_target_world, dim=-1)
        palm_flipped = palm_flip_cos < self.cfg.palm_flip_cos_thresh

        out_of_reach = (
            object_outside_upper_x
            | object_outside_lower_x
            | object_outside_upper_y
            | object_outside_lower_y
            | object_too_low
            | hand_too_far
            | hand_too_close
            | self.arm_table_contact_mask
            | palm_flipped
        )
        no_contact_timeout = (
            (self.episode_length_buf >= self.no_contact_max_steps)
            & (~self.had_object_contact)
        )
        # out_of_reach = out_of_reach | no_contact_timeout
#============================================================================================================================
        # if out_of_reach.any():
        #     env_ids = torch.nonzero(out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #     print(f"termination triggered in envs: {env_ids}")

        #     object_fail = (
        #         object_outside_upper_x
        #         | object_outside_lower_x
        #         | object_outside_upper_y
        #         | object_outside_lower_y
        #         | object_too_low
        #     )
        #     hand_fail = hand_too_far | hand_too_close | palm_flipped

        #     if object_fail.any():
        #         envs = torch.nonzero(object_fail & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #         print(f"object out of range termination: envs {envs}")
        #     if hand_fail.any():
        #         envs = torch.nonzero(hand_fail & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #         print(f"hand out of range termination: envs {envs}")
        #         if hand_too_far.any():
        #             envs = torch.nonzero(hand_too_far & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #             print(f"hand too far: envs {envs}")
        #         if hand_too_close.any():
        #             envs = torch.nonzero(hand_too_close & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #             print(f"hand too close: envs {envs}")
        #         # if palm_flipped.any():
        #         #     envs = torch.nonzero(palm_flipped & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #         #     print(f"palm flipped: envs {envs}")
        #     if self.arm_table_contact_mask.any():
        #         envs = torch.nonzero(self.arm_table_contact_mask & out_of_reach, as_tuple=False).squeeze(-1).tolist()
        #         print(f"arm colliding with the table termination: envs {envs}")
        #     # input("debugging termination conditions")
#===================================================================================================================================

        # Terminate rollout if maximum episode length reached
        if self.cfg.distillation:
            time_out = torch.logical_or(
                self.episode_length_buf >= self.max_episode_length - 1,
                self.time_in_success_region >= self.cfg.success_timeout
            )
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1

        #return out_of_reach, time_out
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        if self.cfg.disable_out_of_reach_done:
            if env_ids.shape[0] != self.num_envs:
                return

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        num_ids = env_ids.shape[0]

        # Reset object state
        object_start_state = torch.zeros(self.num_envs, 13, device=self.device)
        # Shift and scale the X-Y spawn locations
        object_xy = torch.rand(num_ids, 2, device=self.device) - 0.5 # [-.5, .5]
        x_width_spawn = self.dextrah_adr.get_custom_param_value("object_spawn", "x_width_spawn")
        y_width_spawn = self.dextrah_adr.get_custom_param_value("object_spawn", "y_width_spawn")
        object_xy[:, 0] *= x_width_spawn
        object_xy[:, 0] += self.cfg.x_center
        object_xy[:, 1] *= y_width_spawn
        object_xy[:, 1] += self.cfg.y_center
        object_start_state[env_ids, :2] = object_xy
        # Keep drop height the same
        # object_start_state[:, 2] = 0.5
        object_start_state[:, 2] = 0.4

        #object_start_state[:, 3] = 1.
        rotation = self.dextrah_adr.get_custom_param_value("object_spawn", "rotation") 
        rot_noise = sample_uniform(-rotation, rotation, (num_ids, 2), device=self.device)  # noise for X and Y rotation
        object_start_state[env_ids, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state = object_start_state[env_ids]

        # Add the env origin translations
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )

        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # Spawning robot
        joint_pos_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_pos_noise")
        joint_vel_noise = self.dextrah_adr.get_custom_param_value("robot_spawn" ,"joint_vel_noise")

        num_actuated = len(self.actuated_dof_indices)
        joint_pos_deltas = 2. * (torch.rand(num_ids, num_actuated, device=self.device) - 0.5)
        joint_vel_deltas = 2. * (torch.rand(num_ids, num_actuated, device=self.device) - 0.5)

        # Calculate joint positions
        dof_pos = self.robot_start_joint_pos[env_ids].clone()
        dof_pos[:, self.actuated_dof_indices] += joint_pos_noise * joint_pos_deltas

        # Now clamp
        dof_pos[:, self.actuated_dof_indices] = torch.clamp(dof_pos[:, self.actuated_dof_indices],
                                                            min=self.robot_dof_lower_limits[0],
                                                            max=self.robot_dof_upper_limits[0])

        dof_vel = self.robot_start_joint_vel[env_ids].clone()
        dof_vel[:, self.actuated_dof_indices] += joint_vel_noise * joint_vel_deltas

        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        
        # Reset position and velocity targets to the actual robot position and velocity
        self.robot.set_joint_position_target(dof_pos[:, self.actuated_dof_indices],
            env_ids=env_ids, joint_ids=self.actuated_dof_indices)
        self.robot.set_joint_velocity_target(dof_vel[:, self.actuated_dof_indices],
            env_ids=env_ids, joint_ids=self.actuated_dof_indices)

        # Poll robot and object data
        self._compute_intermediate_values()

        # Reset success signals
        self.in_success_region[env_ids] = False
        self.time_in_success_region[env_ids] = 0.
        self.had_object_contact[env_ids] = False

        # Get object mass - this is used in F/T disturbance, etc.
        # NOTE: object mass on the CPU, so we only query infrequently
        self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)

        # Get robot properties
        self.robot_dof_stiffness = self.robot.root_physx_view.get_dof_stiffnesses().to(device=self.device)
        self.robot_dof_damping = self.robot.root_physx_view.get_dof_dampings().to(device=self.device)
        self.robot_material_props =\
            self.robot.root_physx_view.get_material_properties().to(device=self.device).view(self.num_envs, -1)

        # OBJECT NOISE---------------------------------------------------------------------------
        # Sample widths of uniform distribtion controlling pose bias
        self.object_pos_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_bias") *\
            torch.rand(num_ids, device=self.device)
        self.object_rot_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_bias") *\
            torch.rand(num_ids, device=self.device)

        # Now sample the uniform distributions for bias
        self.object_pos_bias[env_ids, 0] = self.object_pos_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)
        self.object_rot_bias[env_ids, 0] = self.object_rot_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)

        # Sample width of per-step noise
        self.object_pos_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_pos_noise") *\
            torch.rand(num_ids, device=self.device)
        self.object_rot_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("object_state_noise", "object_rot_noise") *\
            torch.rand(num_ids, device=self.device)

        # ROBOT NOISE---------------------------------------------------------------------------
        # Sample widths of uniform distribution controlling robot state bias
        self.robot_joint_pos_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_pos_bias") *\
            torch.rand(num_ids, device=self.device)
        self.robot_joint_vel_bias_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_vel_bias") *\
            torch.rand(num_ids, device=self.device)

        # Now sample the uniform distributions for bias
        self.robot_joint_pos_bias[env_ids, 0] = self.robot_joint_pos_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)
        self.robot_joint_vel_bias[env_ids, 0] = self.robot_joint_vel_bias_width[env_ids, 0] *\
            (torch.rand(num_ids, device=self.device) - 0.5)

        # Sample width of per-step noise
        self.robot_joint_pos_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_pos_noise") *\
            torch.rand(num_ids, device=self.device)
        self.robot_joint_vel_noise_width[env_ids, 0] =\
            self.dextrah_adr.get_custom_param_value("robot_state_noise", "robot_joint_vel_noise") *\
            torch.rand(num_ids, device=self.device)

        # Update DR ranges
        if self.cfg.enable_adr:
            if self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change and \
                (self.in_success_region.float().mean() > self.cfg.success_for_adr) and\
                (self.local_adr_increment == self.global_min_adr_increment):
                self.step_since_last_dr_change = 0
                self.dextrah_adr.increase_ranges(increase_counter=True)
                self.event_manager.reset(env_ids=self.robot._ALL_INDICES)
                self.event_manager.apply(env_ids=self.robot._ALL_INDICES, mode="reset", global_env_step_count=0)
                self.local_adr_increment = torch.tensor(self.dextrah_adr.num_increments(), device=self.device, dtype=torch.int64)
            else:
                #print('not increasing DR ranges')
                self.step_since_last_dr_change += 1

        # randomize camera position
        if self.use_camera:
            rand_rots = np.random.uniform(
                -self.cfg.camera_rand_rot_range,
                self.cfg.camera_rand_rot_range,
                size=(num_ids, 3)
            )
            new_rots = rand_rots + self.camera_rot_eul_orig
            new_rots_quat = R.from_euler('xyz', new_rots, degrees=True).as_quat()
            new_rots_quat = new_rots_quat[:, [3, 0, 1, 2]]
            new_rots_quat = torch.tensor(new_rots_quat).to(self.device).float()
            new_pos = self.camera_pos_orig + torch.empty(
                num_ids, 3, device=self.device
            ).uniform_(
                -self.cfg.camera_rand_pos_range,
                self.cfg.camera_rand_pos_range
            )
            np_env_ids = env_ids.cpu().numpy()
            self.camera_pose[np_env_ids, :3, :3] = R.from_euler(
                'xyz', new_rots, degrees=True
            ).as_matrix()
            self.camera_pose[np_env_ids, :3, 3] = (
                new_pos + self.scene.env_origins[env_ids]
            ).cpu().numpy()
            self.left_pos[env_ids] = new_pos + self.scene.env_origins[env_ids]
            self.left_rot[env_ids] = new_rots_quat
            self._tiled_camera.set_world_poses(
                positions=new_pos + self.scene.env_origins[env_ids],
                orientations=new_rots_quat,
                env_ids=env_ids,
                convention="ros"
            )

            rand_rots = np.random.uniform(
                -2, 2, size=(num_ids, 3)
            )
            new_rots = rand_rots + self.camera_right_rot_eul_orig
            new_rots_quat = R.from_euler('xyz', new_rots, degrees=True).as_quat()
            new_rots_quat = new_rots_quat[:, [3, 0, 1, 2]]
            new_rots_quat = torch.tensor(new_rots_quat).to(self.device).float() 
            new_pos = self.camera_right_pos_orig + torch.empty(
                num_ids, 3, device=self.device
            ).uniform_(
                -3e-3, 3e-3
            )
            self.camera_right_pose[np_env_ids, :3, :3] = R.from_euler(
                'xyz', new_rots, degrees=True
            ).as_matrix()
            self.camera_right_pose[np_env_ids, :3, 3] = new_pos.cpu().numpy()


            if self.cfg.disable_dome_light_randomization:
                dome_light_rand_ratio = 0.0
            else:
                dome_light_rand_ratio = 0.3
            if random.random() < dome_light_rand_ratio:
                dome_light_texture = random.choice(self.dome_light_files)
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:texture:file"
                ).Set(dome_light_texture)
                x, y, z, w = R.random().as_quat()
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "xformOp:orient"
                ).Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
                self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                    "inputs:intensity"
                ).Set(np.random.uniform(1000., 4000.))
                # # Define hue range for cooler colors (e.g., 180° to 300° in HSV)
                # # Hue in colorsys is between 0 and 1, corresponding to 0° to 360°
                # cool_hue_min = 0.5  # 180°
                # cool_hue_max = 0.833  # 300°

                # # Generate random hue within the cooler range
                # hue = np.random.uniform(cool_hue_min, cool_hue_max)

                # # Generate random saturation and value within desired ranges
                # saturation = np.random.uniform(0.5, 1.0)  # Moderate to high saturation
                # value = np.random.uniform(0.5, 1.0)       # Moderate to high brightness

                # # Convert HSV to RGB
                # r, g, b = hsv_to_rgb(hue, saturation, value)

                # self.stage.GetPrimAtPath("/World/Light").GetAttribute(
                #     "inputs:color"
                # ).Set(
                #     Gf.Vec3f(r, g, b)
                # )

            rand_attributes = [
                "diffuse_texture",
                "project_uvw",
                "texture_scale",
                "diffuse_tint",
                "reflection_roughness_constant",
                "metallic_constant",
                "specular_level",
            ]
            attribute_types = [
                Sdf.ValueTypeNames.Asset,
                Sdf.ValueTypeNames.Bool,
                Sdf.ValueTypeNames.Float2,
                Sdf.ValueTypeNames.Color3f,
                Sdf.ValueTypeNames.Float,
                Sdf.ValueTypeNames.Float,
                Sdf.ValueTypeNames.Float,
            ]
            for env_id in np_env_ids:
                mat_prim = self.object_mat_prims[env_id]
                property_names = mat_prim.GetPropertyNames()
                rand_attribute_vals = [
                    random.choice(self.object_textures),
                    True,
                    tuple(np.random.uniform(0.7, 5, size=(2))),
                    tuple(np.random.rand(3)),
                    np.random.uniform(0., 1.),
                    np.random.uniform(0., 1.),
                    np.random.uniform(0., 1.),
                ]
                for attribute_name, attribute_type, value in zip(
                    rand_attributes,
                    attribute_types,
                    rand_attribute_vals,
                ):
                    disp_name = "inputs:" + attribute_name
                    if disp_name not in property_names:
                        shader = UsdShade.Shader(
                            omni.usd.get_shader_from_material(
                                mat_prim.GetParent(),
                                True
                            )
                        )
                        shader.CreateInput(
                            attribute_name, attribute_type
                        )
                    mat_prim.GetAttribute(
                        disp_name
                    ).Set(value)

            if not self.cfg.disable_arm_randomization:
                with Sdf.ChangeBlock():
                    for idx, arm_shader_prim in enumerate(self.arm_mat_prims):
                        if idx not in env_ids:
                            continue
                        for arm_shader in arm_shader_prim:
                            arm_shader.GetAttribute("inputs:reflection_roughness_constant").Set(
                                np.random.uniform(0.2, 1.)
                            )
                            arm_shader.GetAttribute("inputs:metallic_constant").Set(
                                np.random.uniform(0, 0.8)
                            )
                            arm_shader.GetAttribute("inputs:specular_level").Set(
                                np.random.uniform(0., 1.)
                            )
                    for i in np_env_ids:
                        shader_path = f"/World/envs/env_{i}/table/Looks/OmniPBR/Shader"
                        shader_prim = self.stage.GetPrimAtPath(shader_path)
                        shader_prim.GetAttribute("inputs:diffuse_texture").Set(
                            random.choice(self.table_texture_files)
                        )
                        shader_prim.GetAttribute("inputs:diffuse_tint").Set(
                            Gf.Vec3d(
                                np.random.uniform(0.3, 0.6),
                                np.random.uniform(0.2, 0.4),
                                np.random.uniform(0.1, 0.2)
                            )
                        )
                        shader_prim.GetAttribute("inputs:specular_level").Set(
                            np.random.uniform(0., 1.)
                        )
                        shader_prim.GetAttribute("inputs:reflection_roughness_constant").Set(
                            np.random.uniform(0.3, 0.9)
                        )
                        shader_prim.GetAttribute("inputs:texture_rotate").Set(
                            np.random.uniform(0., 2*np.pi)
                        )

    def _collect_table_contacts(self):
        """Collect table contacts per env, update mask, and return [[env_idx, contacts], ...].

        Contacts are reported as (link_name, filter_name) tuples.
        """
        contacts_per_env = [[] for _ in range(self.num_envs)]
        self.arm_table_contact_mask.zero_()

        for link_name, sensor in zip(self.table_contact_links, self.table_contact_sensors):
            data = sensor.data
            if data is None or data.force_matrix_w is None:
                continue

            fm = data.force_matrix_w
            contact_mask = (fm.abs().sum(-1) > 1e-4)

            per_env_contact = contact_mask
            if per_env_contact.dim() > 2:
                per_env_contact = per_env_contact.any(dim=-1)
            if per_env_contact.dim() > 1:
                per_env_contact = per_env_contact.any(dim=-1)
            self.arm_table_contact_mask |= per_env_contact.to(self.arm_table_contact_mask.device)

            nz = contact_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue

            filters = getattr(sensor.cfg, "filter_prim_paths_expr", [])
            for env_idx, body_idx, filter_idx in nz.tolist():
                filter_name = filters[filter_idx] if filter_idx < len(filters) else f"filter_{filter_idx}"
                contacts_per_env[env_idx].append((link_name, filter_name))

        return [[env_idx, contacts] for env_idx, contacts in enumerate(contacts_per_env) if contacts]

    def _collect_object_contacts(self):
        """Collect object contacts per env. Returns [[env_idx, [num_contacts, link_names]], ...]."""
        contact_counts = [0 for _ in range(self.num_envs)]
        contact_links = [[] for _ in range(self.num_envs)]

        for link_name, sensor in zip(self.object_contact_links, self.object_contact_sensors):
            data = sensor.data
            if data is None or data.force_matrix_w is None:
                continue

            contact_mask = (data.force_matrix_w.abs().sum(-1) > 1e-4)
            nz = contact_mask.nonzero(as_tuple=False)
            if len(nz) == 0:
                continue

            for env_idx, _, _ in nz.tolist():
                contact_counts[env_idx] += 1
                if link_name not in contact_links[env_idx]:
                    contact_links[env_idx].append(link_name)

        self.object_contact_counts = torch.tensor(
            contact_counts, device=self.device, dtype=torch.float
        )
        return [
            [env_idx, [contact_counts[env_idx], contact_links[env_idx]]]
            for env_idx in range(self.num_envs)
            if contact_counts[env_idx] > 0
        ]

    def _compute_intermediate_values(self):
        # Data from robot--------------------------
        # Robot measured joint position and velocity
        self.robot_dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        # print(self.robot_dof_pos)
        self.robot_dof_pos_noisy = self.robot_dof_pos +\
            self.robot_joint_pos_noise_width *\
            2. * (torch.rand_like(self.robot_dof_pos) - 0.5) +\
            self.robot_joint_pos_bias

        self.robot_dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        self.robot_dof_vel_noisy = self.robot_dof_vel +\
            self.robot_joint_vel_noise_width *\
            2. * (torch.rand_like(self.robot_dof_vel) - 0.5) +\
            self.robot_joint_vel_bias
        self.robot_dof_vel_noisy *= self.dextrah_adr.get_custom_param_value(
            "observation_annealing"
            ,"coefficient"
        )

        # Full-DOF tensors for taskmap kinematics
        robot_dof_pos_noisy_full = self.robot.data.joint_pos.clone()
        robot_dof_vel_noisy_full = self.robot.data.joint_vel.clone()
        robot_dof_pos_noisy_full[:, self.actuated_dof_indices] = self.robot_dof_pos_noisy
        robot_dof_vel_noisy_full[:, self.actuated_dof_indices] = self.robot_dof_vel_noisy

        # Robot fingertip and palm position. NOTE: currently not adding orientation
        self.hand_pos = self.robot.data.body_pos_w[:, self.hand_bodies]
        self.hand_pos -= self.scene.env_origins.repeat((1, self.num_hand_bodies
            )).reshape(self.num_envs, self.num_hand_bodies, 3)

        # Hand points used for distance-to-object calculation.
        self.hand_object_distance_pos = self.robot.data.body_pos_w[:, self.hand_object_distance_bodies]
        self.hand_object_distance_pos -= self.scene.env_origins.repeat(
            (1, self.num_hand_object_distance_bodies)
        ).reshape(self.num_envs, self.num_hand_object_distance_bodies, 3)

        # Robot fingertip and palm velocity. 6D
        self.hand_vel = self.robot.data.body_vel_w[:, self.hand_bodies]

        # Noisy hand point position and velocity from hand taskmap
        self.hand_pos_noisy, hand_points_jac = self.hand_points_taskmap(robot_dof_pos_noisy_full, None)
        self.hand_vel_noisy = torch.bmm(hand_points_jac, robot_dof_vel_noisy_full.unsqueeze(2)).squeeze(2)
        self.hand_vel_noisy *= self.dextrah_adr.get_custom_param_value(
            "observation_annealing"
            ,"coefficient"
        )

        # Compute table data (per-step)
        self.table_pos = self.table.data.root_pos_w - self.scene.env_origins
        self.table_pos_z = self.table_pos[:, 2]

        # Contact sensors: populate per-env arm-table mask and optionally report pairs.
        self.table_contact_report = self._collect_table_contacts()
        # if self.table_contact_report:
        #     for env_idx, contacts in self.table_contact_report:
        #         print(f"[env {env_idx}] table contact pairs: {contacts}")      
        #     print("#################")

        # Query the finger forces
        self.hand_forces =\
            self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.hand_bodies]
        self.hand_forces =\
            self.hand_forces.view(self.num_envs, self.num_hand_bodies * 6)
        
        # Query the measured torque on the joints
        self.measured_joint_torque =\
            self.robot.root_physx_view.get_dof_projected_joint_forces()
        
        # Debugging
        # Contact forces/mask for all robot bodies (commented out)
        # self.contact_forces.zero_()
        # self.contact_mask.zero_()
        # self.arm_in_contact_with_table.zero_()
        # if not self._contact_debug_printed:
        #     print("[DEBUG] contact reporting disabled; forces/mask zeroed.")
        #     self._contact_debug_printed = True

        # Data from objects------------------------
        # Object translational position, 3D
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # NOTE: noise on object pos and rot is per-step sampled uniform noise and sustained
        # bias noise sampled only at start of rollout
        self.object_pos_noisy = self.object_pos +\
            self.object_pos_noise_width *\
            2. * (torch.rand_like(self.object_pos) - 0.5) +\
            self.object_pos_bias

        # Object rotational position, 4D
        self.object_rot = self.object.data.root_quat_w
        self.object_rot_noisy = self.object_rot +\
            self.object_rot_noise_width *\
            2. * (torch.rand_like(self.object_rot) - 0.5) +\
            self.object_rot_bias

        # Object full velocity, 6D
        self.object_vel = self.object.data.root_vel_w

        # Update palm directions (world-frame) from current palm orientation
        palm_rot = quat_to_rotmat(self.robot.data.body_quat_w[:, self.palm_body_idx])
        self.palm_direction_vec = torch.bmm(
            palm_rot, self._palm_down_local_axis.expand(self.num_envs, 3, 1)
        ).squeeze(-1)
        self.palm_finger_direction_vec = torch.bmm(
            palm_rot, self._palm_finger_local_axis.expand(self.num_envs, 3, 1)
        ).squeeze(-1)
        # # Debug: print palm direction in world frame per env
        # for env_idx, vec in enumerate(self.palm_direction_vec.detach().cpu().numpy()):
        #     print(f"[PalmDir] env {env_idx}: {vec}")
        # input()

    def compute_intermediate_reward_values(self):
        # Calculate distance between object and its goal position: object to goal error
        self.object_to_object_goal_pos_error =\
            torch.norm(self.object_pos - self.object_goal, dim=-1)

        # Calculate vertical error: lift error
        self.object_vertical_error = torch.abs(self.object_goal[:, 2] - self.object_pos[:, 2])

        # Calculate whether object is within success region: in_success_region
        self.in_success_region = self.object_to_object_goal_pos_error < self.cfg.object_goal_tol
        # if not in success region, reset time in success region, else increment
        self.time_in_success_region = torch.where(
            self.in_success_region,
            self.time_in_success_region + self.cfg.sim.dt*self.cfg.decimation,
            0.
        )

        # Palm to object center distance (world frame).
        palm_pos = self.robot.data.body_pos_w[:, self.palm_body_idx] - self.scene.env_origins
        self.hand_object_palm_center_dist = torch.norm(palm_pos - self.object_pos, dim=-1)

        # Object to hand points distance (average over selected bodies)
        min_dists = None
        object_points_world = None
        if self.object_pointclouds_local is not None and (
            self.use_pointcloud_hand_object_dist or self.encode_hand_object_dist
        ):
            # query object rotation
            object_rot_mat = quat_to_rotmat(self.object_rot)
            
            # transform object point clouds to world frame
            object_points_world = torch.bmm(
                self.object_pointclouds_local,
                object_rot_mat.transpose(1, 2),
            ) + self.object_pos[:, None, :]

            # compute hand-object point cloud min distances
            min_dists = torch.cdist(self.hand_object_distance_pos, object_points_world).min(dim=-1).values
            dist_clip = getattr(self.cfg, "hand_object_pointcloud_dist_clip", None)
            if dist_clip is not None:
                min_dists = torch.clamp(min_dists, max=dist_clip)
            
            # query palm and finger to object pc distances
            self.hand_object_pointcloud_min_dists = min_dists
            
            # query palm pc distance
            palm_dists = min_dists[:, self._palm_pc_indices]
            self.hand_object_palm_dist = palm_dists.mean(dim=-1)
            
            # query fingertip pc distance: sum of min distances across fingertips
            if self._finger_pc_indices:
                self.hand_object_finger_dist = min_dists[:, self._finger_pc_indices].sum(dim=-1)
            else:
                self.hand_object_finger_dist = torch.zeros_like(self.hand_object_palm_dist)
        
        elif self.encode_hand_object_dist: # failsafe
            if not self._pc_obs_warned:
                print("[WARN] encode_hand_object_dist enabled without point clouds; filling zeros.")
                self._pc_obs_warned = True
            self.hand_object_pointcloud_min_dists = torch.zeros(
                self.num_envs,
                self.num_hand_object_distance_bodies,
                device=self.device,
            )
            self.hand_object_palm_dist = torch.zeros(self.num_envs, device=self.device)
            self.hand_object_finger_dist = torch.zeros(self.num_envs, device=self.device)

        if object_points_world is not None and self.num_hand_body_pc > 0:
            hand_body_pos = (
                self.robot.data.body_pos_w[:, self.hand_body_pc_indices]
                - self.scene.env_origins[:, None, :]
            )
            self.hand_body_pc_batch_dist = torch.cdist(hand_body_pos, object_points_world).min(dim=-1).values
        elif self.num_hand_body_pc > 0:
            self.hand_body_pc_batch_dist = torch.zeros(
                self.num_envs,
                self.num_hand_body_pc,
                device=self.device,
            )
        
        # output final hand to object position error
        # calculate hold flag
        max_finger_dist = getattr(self.cfg, "hand_object_hold_max_finger_dist", 0.3)
        max_palm_dist = getattr(self.cfg, "hand_object_hold_max_palm_dist", 0.06)
        self.hand_object_hold_flag = (
            (self.hand_object_finger_dist <= max_finger_dist).to(torch.int64)
            + (self.hand_object_palm_dist <= max_palm_dist).to(torch.int64)
        )

        # set final hand to object position error
        if self.use_pointcloud_hand_object_dist and min_dists is not None:
            self.hand_to_object_pos_error = self.hand_object_palm_center_dist ** 2
        else:
            self.hand_to_object_pos_error = (
                torch.norm(self.hand_object_distance_pos - self.object_pos[:, None, :], dim=-1).max(dim=-1).values
            )
            self.hand_object_palm_dist = self.hand_to_object_pos_error
            self.hand_object_finger_dist = torch.zeros_like(self.hand_object_palm_dist)

        

        # contact counts
        self.object_contact_report = self._collect_object_contacts()
        self.had_object_contact |= self.object_contact_counts > 0.0
        # if self.object_contact_report:
        #     print(self.object_contact_report)
        #     print("#==========================")
        # action delta
        self.action_delta = self.actions - self.prev_actions

    def compute_actions(self, actions: torch.Tensor) -> None: #torch.Tensor:
        assert_equals(actions.shape, (self.num_envs, self.cfg.num_actions))

        # Scale actions to joint limits and set PD targets
        joint_targets = compute_absolute_action(
            raw_actions=actions,
            lower_limits=self.robot_dof_lower_limits[0],
            upper_limits=self.robot_dof_upper_limits[0],
        )

        self.joint_position_targets[:, self.actuated_dof_indices] = joint_targets
        self.dof_pos_targets[:, self.actuated_dof_indices] = joint_targets
        self.dof_vel_targets[:, self.actuated_dof_indices] =\
            self.joint_velocity_targets[:, self.actuated_dof_indices]

    def compute_student_policy_observations(self):
        pc_dists = (
            self.hand_body_pc_batch_dist if self.encode_hand_object_dist else None
        )
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos_noisy, # 0:23
                self.robot_dof_vel_noisy, # 23:46
                self.hand_pos_noisy, # 46:61
                self.hand_vel_noisy, # 61:76
                # object goal
                self.object_goal, # 76:79
                # hand-object point cloud distances
                pc_dists if pc_dists is not None else self.object_goal[:, :0],
                # last action
                self.actions, # 79:90
            ),
            dim=-1,
        )

        return obs

    def compute_policy_observations(self):
        pc_dists = (
            self.hand_body_pc_batch_dist if self.encode_hand_object_dist else None
        )
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos_noisy,
                self.robot_dof_vel_noisy,
                self.hand_pos_noisy,
                self.hand_vel_noisy,

                # noisy object position, orientation
                self.object_pos_noisy,
                self.object_rot_noisy,
                #self.object_vel, # NOTE: took this out because it's fairly privileged
                
                # object goal
                self.object_goal,

                # hand-object point cloud distances
                pc_dists if pc_dists is not None else self.object_goal[:, :0],
                
                # one-hot encoding of object ID
                self.multi_object_idx_onehot,
                
                # object scales
                self.object_scale,
                
                # last action
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_critic_observations(self):
        obs = torch.cat(
            (
                # robot
                self.robot_dof_pos,
                self.robot_dof_vel,
                self.hand_pos.view(self.num_envs, self.num_hand_bodies * 3),
                self.hand_vel.view(self.num_envs, self.num_hand_bodies * 6),
                self.hand_forces[:, :3], # 3D forces on fingertips and palm.
                self.measured_joint_torque, # meeasured joint torque
                # object
                self.object_pos,
                self.object_rot,
                self.object_vel,
                # object goal
                self.object_goal,
                # one-hot encoding of object ID
                self.multi_object_idx_onehot,
                # object scale
                self.object_scale,
                # last action
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def apply_object_wrench(self):
        # Update whether to apply wrench based on whether object is at goal
        self.apply_wrench = torch.where(
            self.hand_to_object_pos_error <= self.cfg.hand_to_object_dist_threshold,
            True,
            False
        )

        body_ids = None # targets all bodies
        env_ids = None # targets all envs

        num_bodies = self.object.num_bodies

        # Generates the random wrench
        max_linear_accel = self.dextrah_adr.get_custom_param_value("object_wrench", "max_linear_accel")
        linear_accel = max_linear_accel * torch.rand(self.num_envs, 1, device=self.device)
        max_force = (linear_accel * self.object_mass).unsqueeze(2)
        max_torque = (self.object_mass * linear_accel * self.cfg.torsional_radius).unsqueeze(2)
        forces =\
            max_force * torch.nn.functional.normalize(
                torch.randn(self.num_envs, num_bodies, 3, device=self.device),
                dim=-1
            )
        torques =\
            max_torque * torch.nn.functional.normalize(
                torch.randn(self.num_envs, num_bodies, 3, device=self.device),
                dim=-1
            )
        
        self.object_applied_force = torch.where(
            (self.episode_length_buf.view(-1, 1, 1) % self.cfg.wrench_trigger_every) == 0,
            forces,
            self.object_applied_force
        )

        self.object_applied_force = torch.where(
            self.apply_wrench[:, None, None],
            self.object_applied_force,
            torch.zeros_like(self.object_applied_force)
        )

        self.object_applied_torque = torch.where(
            (self.episode_length_buf.view(-1, 1, 1) % self.cfg.wrench_trigger_every) == 0,
            torques,
            self.object_applied_torque
        )

        self.object_applied_torque = torch.where(
            self.apply_wrench[:, None, None],
            self.object_applied_torque,
            torch.zeros_like(self.object_applied_torque)
        )

        # Set the wrench to the buffers
        self.object.set_external_force_and_torque(
            forces=self.object_applied_force,
            torques=self.object_applied_torque,
            body_ids = body_ids,
            env_ids = env_ids
        )

        # Write wrench data to sim
        self.object.write_data_to_sim()

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    in_success_region: torch.Tensor,
    max_episode_length: float,
    hand_to_object_pos_error: torch.Tensor,
    hand_object_palm_dist: torch.Tensor,
    hand_object_finger_dist: torch.Tensor,
    object_to_object_goal_pos_error: torch.Tensor,
    object_vertical_error: torch.Tensor,
    robot_dof_pos: torch.Tensor,
    curled_q: torch.Tensor,
    hand_to_object_weight: float,
    hand_to_object_sharpness: float,
    hand_object_palm_dist_weight: float,
    hand_object_finger_dist_weight: float,
    object_to_goal_weight: float,
    object_to_goal_sharpness: float,
    finger_curl_reg_weight: float,
    lift_weight: float,
    lift_sharpness: float,
    palm_alignment_weight: float,
    palm_dir: torch.Tensor,
    palm_dir_target: torch.Tensor,
    palm_finger_alignment_weight: float,
    palm_finger_dir: torch.Tensor,
    palm_finger_target: torch.Tensor,
    contact_count: torch.Tensor,
    contact_count_weight: float,
    joint_vel: torch.Tensor,
    joint_vel_penalty_weight: float,
    action_delta: torch.Tensor,
    action_rate_penalty_weight: float,
    hand_vel_scale: float,
    hand_action_scale: float,
):

    # Binary mask: only award lift/goal progress after hand-object contact.
    contact_mask = (contact_count > 0.0).to(contact_count.dtype)

    # Reward for moving fingertip and palm points closer to object centroid point
    hand_to_object_reward = (
        hand_object_palm_dist_weight * hand_object_palm_dist
        + hand_object_finger_dist_weight * hand_object_finger_dist
    )

    # Reward for moving the object to the goal translational position
    object_to_goal_reward =\
        object_to_goal_weight * torch.exp(object_to_goal_sharpness * object_to_object_goal_pos_error) * contact_mask

    # Regularizer on hand joints towards a nominally curled config
    # I brought this in because the fingers seem to curl in a lot to play with the object
    # A good strategy is to approach the object with wider set fingers and then encase the object
    # flexing inwards
    finger_curl_dist = (robot_dof_pos - curled_q).norm(p=2, dim=-1)
    finger_curl_reg =\
        finger_curl_reg_weight * finger_curl_dist ** 2

    # Reward for lifting object off table and towards object goal
    lift_reward = lift_weight * torch.exp(-lift_sharpness * object_vertical_error) * contact_mask

    # Palm alignment penalty: 0 when perfectly aligned, negative when deviating from target.
    cos_sim = torch.sum(palm_dir * palm_dir_target, dim=-1).clamp(-1.0, 1.0)
    palm_dir_align_reward = -palm_alignment_weight * (1.0 - cos_sim)

    cos_sim_finger = torch.sum(palm_finger_dir * palm_finger_target, dim=-1).clamp(-1.0, 1.0)
    palm_finger_align_reward = -palm_finger_alignment_weight * (1.0 - cos_sim_finger)

    # Reward for making contact with the object (more contacts -> higher reward).
    contact_reward = contact_count_weight * contact_count
    
    # penalize on joint velocity
    arm_vel = joint_vel[:, :7]
    hand_vel = joint_vel[:, 7:]
    joint_vel_penalty = -joint_vel_penalty_weight * (
        (arm_vel ** 2).sum(dim=-1) + hand_vel_scale * (hand_vel ** 2).sum(dim=-1)
    )

    # penalize on action rate
    arm_delta = action_delta[:, :7]
    hand_delta = action_delta[:, 7:]
    action_rate_penalty = -action_rate_penalty_weight * (
        (arm_delta ** 2).sum(dim=-1) + hand_action_scale * (hand_delta ** 2).sum(dim=-1)
    )

    return (
        hand_to_object_reward,
        object_to_goal_reward,
        finger_curl_reg,
        lift_reward,
        palm_dir_align_reward,
        palm_finger_align_reward,
        contact_reward,
        joint_vel_penalty,
        action_rate_penalty,
    )
    
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
