"""Isaac Lab DirectRLEnv port of the reference SimToolReal task for TG2-InspireHand."""

from __future__ import annotations

import math
from collections.abc import Sequence
import re

import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .simtoolreal_tg2_env_cfg import SimToolRealTg2EnvCfg
from .simtoolreal_tg2_utils import compute_joint_pos_targets, unscale


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., 0:1]), dim=-1)


class SimToolRealTg2Env(DirectRLEnv):
    """SimToolReal-style manipulation task retargeted to TG2-InspireHand.

    This keeps the reference reward, action, delay, and observation structure
    while using TG2's 13 actuated joints and five fingertip bodies.
    """

    cfg: SimToolRealTg2EnvCfg

    def __init__(self, cfg: SimToolRealTg2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._apply_object_mass()

        self.num_robot_dofs = self.robot.num_joints
        self.policy_joint_names = list(getattr(cfg, "policy_joint_names", cfg.actuated_joint_names))
        self.actuated_joint_names = list(cfg.actuated_joint_names)
        self.actuated_dof_indices = [self.robot.joint_names.index(name) for name in self.policy_joint_names]
        self.target_dof_indices = [self.robot.joint_names.index(name) for name in self.actuated_joint_names]
        self.num_actions = len(self.actuated_dof_indices)
        self.cfg.num_actions = self.num_actions

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_action_targets = torch.zeros_like(self.actions)
        self.dof_pos_targets = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device)
        self._policy_joint_name_to_action_idx = {
            name: idx for idx, name in enumerate(self.policy_joint_names)
        }
        self._target_joint_name_to_target_idx = {
            name: idx for idx, name in enumerate(self.actuated_joint_names)
        }

        self.palm_body_idx = self._resolve_body_index(self.cfg.palm_body_name)
        self.fingertip_body_indices = [self._resolve_body_index(name) for name in self.cfg.fingertip_body_names]
        self.palm_offset = torch.tensor(self.cfg.palm_offset, device=self.device)
        self.fingertip_offsets = torch.tensor(self.cfg.fingertip_offsets, device=self.device)

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0][:, self.actuated_dof_indices]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1][:, self.actuated_dof_indices]
        self.target_dof_lower_limits = joint_pos_limits[..., 0][:, self.target_dof_indices]
        self.target_dof_upper_limits = joint_pos_limits[..., 1][:, self.target_dof_indices]

        self.robot_start_joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot_start_joint_vel = torch.zeros_like(self.robot_start_joint_pos)
        self.dof_pos_targets[:] = self.robot_start_joint_pos
        self.prev_action_targets[:] = self.robot_start_joint_pos[:, self.actuated_dof_indices]

        self.object_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_goal_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_goal_rot[:, 0] = 1.0
        self.object_base_scales = torch.tensor(self.cfg.object_scales, device=self.device).repeat(self.num_envs, 1)
        self.object_scale_noise_multiplier = torch.ones((self.num_envs, 3), device=self.device)
        self.object_scales = self.object_base_scales * self.object_scale_noise_multiplier
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_drop_height = torch.zeros(self.num_envs, device=self.device)
        self.object_last_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.successes = torch.zeros(self.num_envs, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.initial_success_tolerance = float(self.cfg.success_tolerance)
        self.success_tolerance = float(self.cfg.success_tolerance)
        self.target_success_tolerance = float(getattr(self.cfg, "target_success_tolerance", 0.01))
        self.last_curriculum_update = 0
        self.frame_since_restart = 0
        num_fingertips = len(self.fingertip_body_indices)
        self.finger_rew_coeffs = torch.ones((self.num_envs, num_fingertips), device=self.device)
        self.finger_contact_bonus_awarded = torch.zeros(
            (self.num_envs, num_fingertips), dtype=torch.bool, device=self.device
        )
        self.closest_fingertip_dist = -torch.ones((self.num_envs, num_fingertips), device=self.device)
        self.furthest_hand_dist = -torch.ones(self.num_envs, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, device=self.device)
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(self.num_envs, device=self.device)

        self.keypoint_offsets = self._make_keypoint_offsets()
        self.grasp_bounding_box_offsets = self._make_grasp_bounding_box_offsets()
        self.fixed_size_keypoint_offsets = self._make_keypoint_offsets(self.cfg.fixed_size)
        debug_draw_enabled = self.cfg.debug_keypoints or self.cfg.debug_grasp_bounding_box or self.cfg.debug_fingertips
        self.keypoint_debug_draw = self._make_keypoint_debug_draw() if debug_draw_enabled else None
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._init_delay_noise_queues()
        self.table_contact_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.table_contact_force_smoothed = torch.zeros_like(self.table_contact_force)
        self.max_table_contact_force_norm_smoothed = torch.zeros(self.num_envs, device=self.device)
        self._setup_object_disturbances()
        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal_object = RigidObject(self.cfg.goal_object_cfg)
        self.table_contact_sensor = None
        if self.cfg.with_table_force_sensor:
            self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor)
        self.hand_object_contact_sensors = {}
        if self.cfg.require_lift_contact or (
            self.cfg.finger_contact_bonus > 0.0 and bool(self.cfg.enable_finger_contact_bonus_reward)
        ):
            object_contact_filter_paths = [
                f"{self.cfg.object_cfg.prim_path}/{self.cfg.object_name}",
            ]
            for body_name in self.cfg.lift_contact_body_names:
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                        update_period=0.0,
                        filter_prim_paths_expr=object_contact_filter_paths,
                        debug_vis=False,
                    )
                )
                self.hand_object_contact_sensors[body_name] = sensor
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=not self.scene.cfg.replicate_physics)
        if not self.scene.cfg.replicate_physics or self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self._disable_goal_object_collisions()
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_object"] = self.goal_object
        if self.table_contact_sensor is not None:
            self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor
        for body_name, sensor in self.hand_object_contact_sensors.items():
            self.scene.sensors[f"{body_name}_object_contact_sensor"] = sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_goal_object_collisions(self) -> None:
        stage = sim_utils.get_current_stage()
        goal_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetName() == "goal_object"]
        for goal_path in goal_paths:
            sim_utils.make_uninstanceable(goal_path, stage)

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not any(prim_path == goal_path or prim_path.startswith(f"{goal_path}/") for goal_path in goal_paths):
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            if collision_api:
                collision_api.CreateCollisionEnabledAttr(False)

    def _apply_object_mass(self) -> None:
        self._apply_mass_to_rigid_object(self.object)
        self._apply_mass_to_rigid_object(self.goal_object)

    def _apply_mass_to_rigid_object(self, rigid_object: RigidObject) -> None:
        masses = rigid_object.root_physx_view.get_masses().clone()
        inertias = rigid_object.root_physx_view.get_inertias().clone()
        old_masses = masses.clone()
        per_body_mass = self.cfg.object_mass / rigid_object.num_bodies
        masses[:] = per_body_mass

        ratios = torch.where(old_masses > 0.0, masses / old_masses, torch.ones_like(masses))
        inertia_ratios = ratios
        while inertia_ratios.dim() < inertias.dim():
            inertia_ratios = inertia_ratios.unsqueeze(-1)
        inertias[:] = inertias * inertia_ratios

        env_ids = torch.arange(rigid_object.num_instances, device="cpu", dtype=torch.long)
        rigid_object.root_physx_view.set_masses(masses, env_ids)
        rigid_object.root_physx_view.set_inertias(inertias, env_ids)
        rigid_object.data.default_mass = masses.clone()
        rigid_object.data.default_inertia = inertias.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        actions = actions.clone().clamp(-1.0, 1.0)
        self.action_queue = self._update_queue(self.action_queue, actions)
        if getattr(self.cfg, "use_action_delay", True):
            actions = self._sample_from_queue(self.action_queue)
        self.actions = actions.clone()
        self._compute_action_targets()

    def _apply_action(self) -> None:
        self._update_object_disturbances()
        self.robot.set_joint_position_target(
            self.dof_pos_targets[:, self.target_dof_indices], joint_ids=self.target_dof_indices
        )

    def _resolve_body_index(self, body_name: str) -> int:
        matches, matched_names = self.robot.find_bodies(body_name)
        if matches:
            return matches[0]

        matches, matched_names = self.robot.find_bodies(f".*{re.escape(body_name)}")
        if len(matches) == 1:
            return matches[0]

        available_names = ", ".join(self.robot.body_names)
        if len(matches) > 1:
            raise ValueError(
                f"Body pattern '{body_name}' matched multiple bodies {matched_names}. "
                f"Available bodies: {available_names}"
            )
        raise ValueError(f"Body '{body_name}' not found. Available bodies: {available_names}")

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = self._compute_reference_observations()
        obs = torch.clamp(obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations["critic"] = self._compute_reference_states()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self.frame_since_restart += 1
        self._update_success_tolerance_curriculum()

        lifting_reward, lift_bonus_reward, lifted_object = self._lifting_reward()
        fingertip_delta_reward, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        finger_contact_bonus_reward = self._finger_contact_bonus_reward()
        keypoint_reward = self._keypoint_reward(lifted_object)

        keypoints_max_dist = self._reward_keypoints_max_dist()
        keypoint_success_tolerance = self.success_tolerance * self.cfg.keypoint_scale
        near_goal = keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.long()) * near_goal.long()
        else:
            self.near_goal_steps += near_goal.long()

        reached_goal = self.near_goal_steps >= self.cfg.success_steps
        self.successes += reached_goal.float()
        self.consecutive_successes.copy_(self.successes)

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_vel[:, 0:3]), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_vel[:, 3:6]), dim=-1)
        object_lin_vel_penalty *= self.cfg.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.cfg.object_ang_vel_penalty_scale

        arm_action_penalty, hand_action_penalty = self._action_penalties()

        reach_bonus = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)
        if self.cfg.force_consecutive_near_goal_steps:
            reach_bonus = reached_goal.float() * self.cfg.reach_goal_bonus

        reward_term_values = {
            "fingertip_delta_reward": fingertip_delta_reward,
            "hand_delta_penalty": hand_delta_penalty,
            "lifting_reward": lifting_reward,
            "lift_bonus_reward": lift_bonus_reward,
            "finger_contact_bonus_reward": finger_contact_bonus_reward,
            "keypoint_reward": keypoint_reward,
            "reach_bonus": reach_bonus,
            "arm_action_penalty": arm_action_penalty,
            "hand_action_penalty": hand_action_penalty,
            "object_lin_vel_penalty": object_lin_vel_penalty,
            "object_ang_vel_penalty": object_ang_vel_penalty,
        }
        for term_name, term_value in reward_term_values.items():
            if not bool(getattr(self.cfg, f"enable_{term_name}")):
                reward_term_values[term_name] = torch.zeros_like(term_value)
        reward = torch.zeros_like(lifting_reward)
        for term_value in reward_term_values.values():
            reward += term_value
        self.object_last_pos.copy_(self.object_pos)

        success_env_ids = reached_goal.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() > 0:
            self._reset_goals(success_env_ids, is_first_goal=False)
            self.near_goal_steps[success_env_ids] = 0
            self.closest_keypoint_max_dist[success_env_ids] = -1.0
            self.closest_keypoint_max_dist_fixed_size[success_env_ids] = -1.0
            if self.cfg.max_consecutive_successes > 0:
                self.episode_length_buf[success_env_ids] = 0

        reward_terms = {term_name: term_value.mean() for term_name, term_value in reward_term_values.items()}
        reward_terms["total_reward"] = reward.mean()
        self.extras["log"] = {
            "success_rate": reached_goal.float().mean(),
            "success_tolerance": self.success_tolerance,
            "keypoints_max_dist": self._reward_keypoints_max_dist().mean(),
            "object_lift": (0.05 + self.object_pos[:, 2] - self.object_init_pos[:, 2]).mean(),
            "hand_object_contact_rate": self.hand_object_contact.float().mean(),
            "hand_object_contact_force": self.hand_object_contact_forces.sum(dim=-1).mean(),
            "hand_raw_contact_rate": (
                self.hand_raw_contact_forces > float(self.cfg.lift_contact_force_threshold)
            ).any(dim=-1).float().mean(),
            "hand_raw_contact_force": self.hand_raw_contact_forces.sum(dim=-1).mean(),
            "contact_gated_lift_rate": self.just_lifted.float().mean(),
            "new_finger_contact_bonus_count": self.new_finger_contact_bonus.sum(dim=-1).float().mean(),
            **reward_terms,
        }
        # Keep a separate copy for the compact terminal report. The algorithm
        # observer excludes this namespace from TensorBoard because the same
        # values are already emitted above as canonical Episode/* scalars.
        self.extras["reward_terms"] = reward_terms
        return reward

    def _update_success_tolerance_curriculum(self) -> None:
        eval_success_tolerance = getattr(self.cfg, "eval_success_tolerance", None)
        if eval_success_tolerance is not None:
            self.success_tolerance = float(eval_success_tolerance)
            return

        frames_since_restart = self.frame_since_restart
        curriculum_interval = int(getattr(self.cfg, "tolerance_curriculum_interval", 3000))
        if frames_since_restart - self.last_curriculum_update < curriculum_interval:
            return

        mean_successes_per_episode = self.prev_episode_successes.mean()
        if mean_successes_per_episode < 3.0:
            return

        self.success_tolerance *= float(getattr(self.cfg, "tolerance_curriculum_increment", 0.9))
        self.success_tolerance = min(self.success_tolerance, self.initial_success_tolerance)
        self.success_tolerance = max(self.success_tolerance, self.target_success_tolerance)
        self.last_curriculum_update = frames_since_restart
        print(
            f"Prev episode successes: {mean_successes_per_episode.item()}, "
            f"success tolerance: {self.success_tolerance}",
            flush=True,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_fell = self.object_pos[:, 2] < self.cfg.object_z_low_reset_threshold
        hand_far_from_object = self.curr_fingertip_distances.max(dim=-1).values > self.cfg.hand_far_from_object_threshold
        if self.cfg.reset_when_dropped:
            dropped = (self.object_pos[:, 2] < self.object_drop_height) & self.lifted_object
        else:
            dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.table_contact_sensor is not None:
            table_force_too_high = self.max_table_contact_force_norm_smoothed > self.cfg.table_force_threshold
        else:
            table_force_too_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_successes > 0:
            too_many_successes = self.consecutive_successes >= self.cfg.max_consecutive_successes
        else:
            too_many_successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = object_fell | too_many_successes | hand_far_from_object | dropped | table_force_too_high
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._all_env_ids
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        num_ids = env_ids.shape[0]
        table_state = self.table.data.default_root_state[env_ids].clone()
        table_reset_z = (
            self.cfg.table_cfg.init_state.pos[2]
            + sample_uniform(-self.cfg.table_reset_z_range, self.cfg.table_reset_z_range, (num_ids,), self.device)
        )
        table_init_pos = torch.tensor(self.cfg.table_cfg.init_state.pos, device=self.device, dtype=table_state.dtype)
        table_state[:, 0:3] = table_init_pos + self.scene.env_origins[env_ids]
        table_state[:, 2] = table_reset_z + self.scene.env_origins[env_ids, 2]
        table_state[:, 7:13] = 0.0
        self.table.write_root_state_to_sim(table_state, env_ids)

        object_state = self.object.data.default_root_state[env_ids].clone()
        object_spawn_center = torch.tensor(
            self.cfg.object_spawn_center,
            device=self.device,
            dtype=object_state.dtype,
        )
        object_xy_range = float(self.cfg.object_spawn_xy_range)
        object_z_range = float(self.cfg.object_spawn_z_range)
        xy_noise = torch.stack(
            (
                sample_uniform(-object_xy_range, object_xy_range, (num_ids,), self.device),
                sample_uniform(-object_xy_range, object_xy_range, (num_ids,), self.device),
            ),
            dim=-1,
        )
        z_noise = sample_uniform(-object_z_range, object_z_range, (num_ids,), self.device)
        object_state[:, 0] = object_spawn_center[0] + xy_noise[:, 0]
        object_state[:, 1] = object_spawn_center[1] + xy_noise[:, 1]
        object_state[:, 2] = object_spawn_center[2] + z_noise
        object_drop_height = object_state[:, 2].clone()
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_state[:, 3:7] = self._sample_object_quat(num_ids)
        if self.cfg.object_start_pose is not None:
            object_start_pose = torch.tensor(self.cfg.object_start_pose, device=self.device, dtype=object_state.dtype)
            object_state[:, 0:3] = object_start_pose[0:3] + self.scene.env_origins[env_ids]
            object_state[:, 3:7] = object_start_pose[[6, 3, 4, 5]]
            object_drop_height = object_start_pose[2].expand(num_ids)
        object_state[:, 7:13] = 0.0
        self.object.write_root_state_to_sim(object_state, env_ids)

        dof_pos = self.robot_start_joint_pos[env_ids].clone()
        dof_vel = self.robot_start_joint_vel[env_ids].clone()
        default_targets = dof_pos[:, self.actuated_dof_indices]
        delta_max = self.robot_dof_upper_limits[env_ids] - default_targets
        delta_min = self.robot_dof_lower_limits[env_ids] - default_targets
        rand_dof = sample_uniform(0.0, 1.0, (num_ids, self.num_actions), self.device)
        rand_delta = delta_min + (delta_max - delta_min) * rand_dof
        noise_coeff = torch.empty_like(default_targets)
        noise_coeff[:, :7] = self.cfg.reset_dof_pos_noise_arm
        noise_coeff[:, 7:] = self.cfg.reset_dof_pos_noise_fingers
        dof_pos[:, self.actuated_dof_indices] = default_targets + noise_coeff * rand_delta
        dof_pos[:, self.actuated_dof_indices] = saturate(
            dof_pos[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[env_ids],
            self.robot_dof_upper_limits[env_ids],
        )
        dof_vel[:, self.actuated_dof_indices] = sample_uniform(
            -self.cfg.reset_dof_vel_noise,
            self.cfg.reset_dof_vel_noise,
            (num_ids, self.num_actions),
            self.device,
        )
        self._apply_mimic_wrapper_to_full_targets(dof_pos[:, self.actuated_dof_indices], dof_pos, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(
            dof_pos[:, self.target_dof_indices],
            env_ids=env_ids,
            joint_ids=self.target_dof_indices,
        )

        self.dof_pos_targets[env_ids] = dof_pos
        self.prev_action_targets[env_ids] = dof_pos[:, self.actuated_dof_indices]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        if hasattr(self, "finger_contact_bonus_awarded"):
            self.finger_contact_bonus_awarded[env_ids] = False
        self._reset_object_scale_noise(env_ids)
        self._reset_object_disturbances(env_ids)
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0
        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0.0
        self.consecutive_successes[env_ids] = 0.0
        self.table_contact_force[env_ids] = 0.0
        self.table_contact_force_smoothed[env_ids] = 0.0
        self.max_table_contact_force_norm_smoothed[env_ids] = 0.0
        self.obs_queue[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self.object_state_queue[env_ids] = 0.0

        self.object_init_pos[env_ids] = object_state[:, 0:3] - self.scene.env_origins[env_ids]
        self.object_drop_height[env_ids] = object_drop_height
        self._reset_goals(env_ids, is_first_goal=True)
        self._compute_intermediate_values()
        self.object_init_pos[env_ids] = self.object_pos[env_ids]
        self.object_last_pos[env_ids] = self.object_pos[env_ids]

    def set_train_info(self, env_frames: int, *args, **kwargs) -> None:
        self.total_train_env_frames = int(env_frames)

    def get_env_state(self) -> dict:
        return {
            "success_tolerance": self.success_tolerance,
            "last_curriculum_update": self.last_curriculum_update,
            "frame_since_restart": self.frame_since_restart,
        }

    def set_env_state(self, env_state: dict | None) -> None:
        if not env_state:
            return
        success_tolerance = env_state.get("success_tolerance")
        if success_tolerance is not None:
            self.success_tolerance = float(success_tolerance)
        last_curriculum_update = env_state.get("last_curriculum_update")
        if last_curriculum_update is not None:
            self.last_curriculum_update = int(last_curriculum_update)
        frame_since_restart = env_state.get("frame_since_restart")
        if frame_since_restart is not None:
            self.frame_since_restart = int(frame_since_restart)

    def _init_delay_noise_queues(self) -> None:
        obs_queue_length = max(1, int(getattr(self.cfg, "obs_delay_max", 3)))
        action_queue_length = max(1, int(getattr(self.cfg, "action_delay_max", 3)))
        object_state_queue_length = max(1, int(getattr(self.cfg, "object_state_delay_max", 10)))
        self.obs_queue = torch.zeros(
            (self.num_envs, obs_queue_length, self.cfg.num_observations),
            device=self.device,
        )
        self.action_queue = torch.zeros(
            (self.num_envs, action_queue_length, self.num_actions),
            device=self.device,
        )
        self.object_state_queue = torch.zeros(
            (self.num_envs, object_state_queue_length, 13),
            device=self.device,
        )

    def _update_queue(self, queue: torch.Tensor, current_values: torch.Tensor) -> torch.Tensor:
        queue_length = queue.shape[1]
        episode_start = self.episode_length_buf <= 1
        queue[:] = torch.where(
            episode_start[:, None, None],
            current_values[:, None, :].expand(-1, queue_length, -1),
            queue,
        )
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values.clone()
        return queue

    def _sample_from_queue(self, queue: torch.Tensor) -> torch.Tensor:
        delay_indices = torch.randint(0, queue.shape[1], (self.num_envs,), device=self.device)
        return queue[self._all_env_ids, delay_indices].clone()

    def _compute_action_targets(self) -> None:
        if self.cfg.use_relative_control:
            speed = self.cfg.dof_speed_scale * self.step_dt
            targets = self.prev_action_targets + speed * self.actions
            targets = saturate(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        else:
            targets = compute_joint_pos_targets(
                actions=self.actions,
                prev_targets=self.prev_action_targets,
                lower_limits=self.robot_dof_lower_limits,
                upper_limits=self.robot_dof_upper_limits,
                hand_moving_average=self.cfg.hand_moving_average,
                arm_moving_average=self.cfg.arm_moving_average,
                arm_dof_speed_scale=self.cfg.dof_speed_scale,
                dt=self.step_dt,
            )
        self.prev_action_targets.copy_(targets)
        self._apply_mimic_wrapper_to_full_targets(targets, self.dof_pos_targets)

    def _apply_mimic_wrapper_to_full_targets(
        self,
        policy_targets: torch.Tensor,
        full_targets: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        full_targets[:, self.actuated_dof_indices] = policy_targets
        target_values = full_targets[:, self.target_dof_indices].clone()
        target_lower_limits = self.target_dof_lower_limits
        target_upper_limits = self.target_dof_upper_limits
        if env_ids is not None:
            target_lower_limits = target_lower_limits[env_ids]
            target_upper_limits = target_upper_limits[env_ids]
        mimic_joint_map = getattr(self.cfg, "mimic_joint_map", {})
        resolved_target_values: dict[str, torch.Tensor] = {}

        for joint_name, action_idx in self._policy_joint_name_to_action_idx.items():
            if joint_name in self._target_joint_name_to_target_idx:
                resolved_target_values[joint_name] = policy_targets[:, action_idx]

        for joint_name, (source_name, multiplier, offset) in mimic_joint_map.items():
            source_value = resolved_target_values.get(source_name)
            if source_value is None and source_name in self._target_joint_name_to_target_idx:
                source_value = target_values[:, self._target_joint_name_to_target_idx[source_name]]
            if source_value is None:
                raise RuntimeError(f"Cannot resolve mimic source joint '{source_name}' for '{joint_name}'.")
            target_value = source_value * float(multiplier) + float(offset)
            resolved_target_values[joint_name] = target_value
            if joint_name in self._target_joint_name_to_target_idx:
                target_values[:, self._target_joint_name_to_target_idx[joint_name]] = target_value

        target_values = saturate(target_values, target_lower_limits, target_upper_limits)
        full_targets[:, self.target_dof_indices] = target_values

    def _sample_log_uniform(self, min_value: float, max_value: float, shape: tuple[int, ...]) -> torch.Tensor:
        if min_value <= 0.0 or max_value <= 0.0:
            return torch.zeros(shape, device=self.device)
        log_min = math.log(min_value)
        log_max = math.log(max_value)
        return torch.exp(sample_uniform(log_min, log_max, shape, self.device))

    def _setup_object_disturbances(self) -> None:
        masses = self.object.root_physx_view.get_masses().to(self.device)
        self.object_body_masses = masses.unsqueeze(-1)
        self.object_forces = torch.zeros((self.num_envs, self.object.num_bodies, 3), device=self.device)
        self.object_torques = torch.zeros_like(self.object_forces)
        self.random_force_prob = self._sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1], (self.num_envs,)
        )
        self.random_torque_prob = self._sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1], (self.num_envs,)
        )

    def _reset_object_scale_noise(self, env_ids: torch.Tensor) -> None:
        noise_min, noise_max = self.cfg.object_scale_noise_multiplier_range
        self.object_scale_noise_multiplier[env_ids] = sample_uniform(
            noise_min,
            noise_max,
            (env_ids.shape[0], 3),
            self.device,
        )
        self.object_scales[env_ids] = self.object_base_scales[env_ids] * self.object_scale_noise_multiplier[env_ids]

    def _reset_object_disturbances(self, env_ids: torch.Tensor) -> None:
        self.object_forces[env_ids] = 0.0
        self.object_torques[env_ids] = 0.0
        self.random_force_prob[env_ids] = self._sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1], (env_ids.shape[0],)
        )
        self.random_torque_prob[env_ids] = self._sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1], (env_ids.shape[0],)
        )

    def _update_object_disturbances(self) -> None:
        if self.cfg.force_scale <= 0.0 and self.cfg.torque_scale <= 0.0:
            self.object.set_external_force_and_torque(
                torch.zeros(0, 3, device=self.device),
                torch.zeros(0, 3, device=self.device),
            )
            return

        if self.cfg.force_scale > 0.0:
            decay = float(self.cfg.force_decay) ** (self.step_dt / max(float(self.cfg.force_decay_interval), 1.0e-6))
            self.object_forces *= decay
            force_env_ids = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero(as_tuple=False).squeeze(-1)
            if force_env_ids.numel() > 0:
                self.object_forces[force_env_ids] = (
                    torch.randn((force_env_ids.numel(), self.object.num_bodies, 3), device=self.device)
                    * self.object_body_masses[force_env_ids]
                    * self.cfg.force_scale
                )
            if self.cfg.force_only_when_lifted:
                self.object_forces *= self.lifted_object[:, None, None]
        else:
            self.object_forces.zero_()

        if self.cfg.torque_scale > 0.0:
            decay = float(self.cfg.torque_decay) ** (self.step_dt / max(float(self.cfg.torque_decay_interval), 1.0e-6))
            self.object_torques *= decay
            torque_env_ids = (torch.rand(self.num_envs, device=self.device) < self.random_torque_prob).nonzero(as_tuple=False).squeeze(-1)
            if torque_env_ids.numel() > 0:
                self.object_torques[torque_env_ids] = (
                    torch.randn((torque_env_ids.numel(), self.object.num_bodies, 3), device=self.device)
                    * self.object_body_masses[torque_env_ids]
                    * self.cfg.torque_scale
                )
            if self.cfg.torque_only_when_lifted:
                self.object_torques *= self.lifted_object[:, None, None]
        else:
            self.object_torques.zero_()

        self.object.set_external_force_and_torque(self.object_forces, self.object_torques, is_global=False)

    def _compute_intermediate_values(self) -> None:
        # Observe all 19 arm/hand joints after mimic expansion, while the
        # policy still emits only the 13 independent command dimensions.
        self.robot_dof_pos = self.robot.data.joint_pos[:, self.target_dof_indices]
        self.robot_dof_vel = self.robot.data.joint_vel[:, self.target_dof_indices]
        self.policy_dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        self._update_hand_object_contact_buffers()
        self.palm_rot = self.robot.data.body_quat_w[:, self.palm_body_idx]
        palm_offset_w = quat_apply(self.palm_rot, self.palm_offset.unsqueeze(0).expand(self.num_envs, -1))
        self.palm_pos = self.robot.data.body_pos_w[:, self.palm_body_idx] + palm_offset_w - self.scene.env_origins
        fingertip_rot = self.robot.data.body_quat_w[:, self.fingertip_body_indices]
        fingertip_offset_w = quat_apply(
            fingertip_rot.reshape(-1, 4),
            self.fingertip_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, -1, 3)
        self.fingertip_pos = (
            self.robot.data.body_pos_w[:, self.fingertip_body_indices]
            + fingertip_offset_w
            - self.scene.env_origins[:, None, :]
        )
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_vel = torch.cat((self.object.data.root_lin_vel_w, self.object.data.root_ang_vel_w), dim=-1)
        self.object_keypoints = self._compute_keypoints(self.object_pos, self.object_rot, self.object_scales)
        self.object_goal_pos = self.goal_object.data.root_pos_w - self.scene.env_origins
        self.object_goal_rot = self.goal_object.data.root_quat_w
        self.goal_keypoints = self._compute_keypoints(self.object_goal_pos, self.object_goal_rot, self.object_scales)
        self.object_keypoints_fixed_size = self._compute_keypoints(
            self.object_pos, self.object_rot, torch.ones_like(self.object_scales), self.fixed_size_keypoint_offsets
        )
        self.goal_keypoints_fixed_size = self._compute_keypoints(
            self.object_goal_pos,
            self.object_goal_rot,
            torch.ones_like(self.object_scales),
            self.fixed_size_keypoint_offsets,
        )
        self.fingertip_pos_rel_object = self.fingertip_pos - self.object_pos[:, None, :]
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )
        self.keypoint_distances = torch.norm(self.object_keypoints - self.goal_keypoints, dim=-1)
        self.keypoints_max_dist = self.keypoint_distances.max(dim=-1).values
        self.keypoint_distances_fixed_size = torch.norm(self.object_keypoints_fixed_size - self.goal_keypoints_fixed_size, dim=-1)
        self.keypoints_max_dist_fixed_size = self.keypoint_distances_fixed_size.max(dim=-1).values
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )
        self._visualize_debug_shapes()
        if self.table_contact_sensor is not None:
            self._update_table_contact_force_buffers(self.table_contact_sensor.data.net_forces_w)
        else:
            self.table_contact_force = torch.zeros((self.num_envs, 3), device=self.device)
            self.table_contact_force_smoothed = torch.zeros_like(self.table_contact_force)
            self.max_table_contact_force_norm_smoothed = torch.zeros(self.num_envs, device=self.device)

    def _update_hand_object_contact_buffers(self) -> None:
        if not self.hand_object_contact_sensors:
            self.hand_object_contact_forces = torch.zeros((self.num_envs, 0), device=self.device)
            self.hand_raw_contact_forces = torch.zeros((self.num_envs, 0), device=self.device)
            self.hand_object_contact_force_by_body = {}
            self.hand_object_contact_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.hand_object_contact = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return

        forces = []
        raw_forces = []
        for sensor in self.hand_object_contact_sensors.values():
            net_forces = sensor.data.net_forces_w
            if net_forces is None:
                raw_force = torch.zeros(self.num_envs, device=self.device)
            else:
                raw_force = net_forces.norm(dim=-1).reshape(self.num_envs, -1).sum(dim=-1)
            raw_forces.append(raw_force)

            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                force = torch.zeros(self.num_envs, device=self.device)
            else:
                force = force_matrix.norm(dim=-1).reshape(self.num_envs, -1).sum(dim=-1)
            forces.append(force)

        self.hand_object_contact_forces = torch.stack(forces, dim=-1)
        self.hand_raw_contact_forces = torch.stack(raw_forces, dim=-1)
        self.hand_object_contact_force_by_body = dict(
            zip(self.hand_object_contact_sensors, self.hand_object_contact_forces.unbind(dim=-1), strict=True)
        )
        contacts = self.hand_object_contact_forces > float(self.cfg.lift_contact_force_threshold)
        self.hand_object_contact_count = contacts.sum(dim=-1)
        required_count = max(1, min(int(self.cfg.lift_contact_required_body_count), contacts.shape[-1]))
        self.hand_object_contact = self.hand_object_contact_count >= required_count

    def _finger_contact_bonus_reward(self) -> torch.Tensor:
        num_fingertips = len(self.fingertip_body_indices)
        if (
            self.cfg.finger_contact_bonus <= 0.0
            or not bool(self.cfg.enable_finger_contact_bonus_reward)
            or not self.hand_object_contact_force_by_body
        ):
            self.new_finger_contact_bonus = torch.zeros(
                (self.num_envs, num_fingertips), dtype=torch.bool, device=self.device
            )
            return torch.zeros(self.num_envs, device=self.device)

        # The one-time per-finger contact bonus is now gated only by filtered
        # distal-link object contact force.  We intentionally do not request
        # ContactSensor contact-point tracking here because PhysX/Fabric can
        # device-assert on that path.
        distal_body_names = self.cfg.lift_contact_body_names[1:]
        if len(distal_body_names) != num_fingertips:
            raise RuntimeError(
                f"Expected one distal contact body per fingertip, got {len(distal_body_names)} "
                f"contact bodies and {num_fingertips} fingertips."
            )
        contact_forces = torch.stack(
            [self.hand_object_contact_force_by_body[name] for name in distal_body_names], dim=-1
        )
        distal_contacts = contact_forces > float(self.cfg.lift_contact_force_threshold)

        self.new_finger_contact_bonus = distal_contacts & (~self.finger_contact_bonus_awarded)
        self.finger_contact_bonus_awarded |= distal_contacts
        return self.new_finger_contact_bonus.sum(dim=-1).float() * float(self.cfg.finger_contact_bonus)

    def _update_table_contact_force_buffers(self, net_forces_w: torch.Tensor) -> None:
        self.table_contact_force = net_forces_w.sum(dim=1)
        smoothing_alpha = 0.1
        self.table_contact_force_smoothed += smoothing_alpha * (
            self.table_contact_force - self.table_contact_force_smoothed
        )
        table_contact_force_norm_smoothed = self.table_contact_force_smoothed.norm(dim=-1)
        self.max_table_contact_force_norm_smoothed = torch.maximum(
            self.max_table_contact_force_norm_smoothed, table_contact_force_norm_smoothed
        )

    def _compute_reference_observations(self) -> torch.Tensor:
        fingertip_pos_rel_palm = (self.fingertip_pos - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        observed_object_pos, observed_object_rot, _observed_object_vel = self._observed_object_state_for_policy()
        observed_object_keypoints = self._compute_keypoints(observed_object_pos, observed_object_rot, self.object_scales)
        keypoints_rel_palm = (observed_object_keypoints - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        keypoints_rel_goal = (observed_object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1)
        joint_vel = self.robot_dof_vel
        joint_velocity_obs_noise_std = float(getattr(self.cfg, "joint_velocity_obs_noise_std", 0.1))
        if joint_velocity_obs_noise_std > 0.0:
            joint_vel = joint_vel + torch.randn_like(joint_vel) * joint_velocity_obs_noise_std
        obs = torch.cat(
            (
                unscale(self.robot_dof_pos, self.target_dof_lower_limits, self.target_dof_upper_limits),
                joint_vel,
                self.dof_pos_targets[:, self.target_dof_indices],
                self.palm_pos,
                quat_wxyz_to_xyzw(self.palm_rot),
                quat_wxyz_to_xyzw(observed_object_rot),
                fingertip_pos_rel_palm,
                keypoints_rel_palm,
                keypoints_rel_goal,
                self.object_scales,
            ),
            dim=-1,
        )
        if obs.shape[-1] != self.cfg.num_observations:
            raise RuntimeError(f"Expected {self.cfg.num_observations} observations, got {obs.shape[-1]}.")
        self.obs_queue = self._update_queue(self.obs_queue, obs)
        if getattr(self.cfg, "use_obs_delay", True):
            obs = self._sample_from_queue(self.obs_queue)
        return obs

    def _observed_object_state_for_policy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_state = torch.cat((self.object_pos, self.object_rot, self.object_vel), dim=-1)
        self.object_state_queue = self._update_queue(self.object_state_queue, object_state)

        observed_object_state = object_state.clone()
        if getattr(self.cfg, "use_object_state_delay_noise", True):
            observed_object_state = self._sample_from_queue(self.object_state_queue)
            xyz_noise_std = float(getattr(self.cfg, "object_state_xyz_noise_std", 0.01))
            if xyz_noise_std > 0.0:
                observed_object_state[:, 0:3] += torch.randn_like(observed_object_state[:, 0:3]) * xyz_noise_std
            rotation_noise_degrees = float(getattr(self.cfg, "object_state_rotation_noise_degrees", 5.0))
            if rotation_noise_degrees > 0.0:
                observed_object_state[:, 3:7] = self._sample_delta_quat(
                    observed_object_state[:, 3:7], rotation_noise_degrees
                )

        return (
            observed_object_state[:, 0:3],
            observed_object_state[:, 3:7],
            observed_object_state[:, 7:13],
        )

    def _compute_reference_states(self) -> torch.Tensor:
        fingertip_pos_rel_palm = (self.fingertip_pos - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        keypoints_rel_palm = (self.object_keypoints - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        keypoints_rel_goal = (self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1)
        palm_vel = torch.cat(
            (
                self.robot.data.body_lin_vel_w[:, self.palm_body_idx],
                self.robot.data.body_ang_vel_w[:, self.palm_body_idx],
            ),
            dim=-1,
        )
        reward = getattr(self, "reward_buf", torch.zeros(self.num_envs, device=self.device))
        state = torch.cat(
            (
                unscale(self.robot_dof_pos, self.target_dof_lower_limits, self.target_dof_upper_limits),
                self.robot_dof_vel,
                self.dof_pos_targets[:, self.target_dof_indices],
                self.palm_pos,
                quat_wxyz_to_xyzw(self.palm_rot),
                palm_vel,
                quat_wxyz_to_xyzw(self.object_rot),
                self.object_vel,
                fingertip_pos_rel_palm,
                keypoints_rel_palm,
                keypoints_rel_goal,
                self.object_scales,
                self._reward_closest_keypoint_max_dist().unsqueeze(-1),
                self.closest_fingertip_dist,
                self.lifted_object.float().unsqueeze(-1),
                torch.log(self.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
                torch.log(self.successes + 1.0).unsqueeze(-1),
                (0.01 * reward).unsqueeze(-1),
            ),
            dim=-1,
        )
        if state.shape[-1] != self.cfg.num_states:
            raise RuntimeError(f"Expected {self.cfg.num_states} critic states, got {state.shape[-1]}.")
        return state

    def _reset_goals(self, env_ids: torch.Tensor, is_first_goal: bool) -> None:
        num_ids = env_ids.shape[0]
        goal_state = self.goal_object.data.default_root_state[env_ids].clone()

        if self.cfg.goal_object_pose is not None:
            goal_object_pose = torch.tensor(self.cfg.goal_object_pose, device=self.device, dtype=goal_state.dtype)
            goal_pos = goal_object_pose[0:3].expand(num_ids, 3)
            goal_rot = goal_object_pose[[6, 3, 4, 5]].expand(num_ids, 4)
        else:
            lift_min, lift_max = sorted(float(value) for value in self.cfg.goal_height_above_object_range)
            goal_pos = self.object_init_pos[env_ids].clone()
            goal_pos[:, 2] += sample_uniform(lift_min, lift_max, (num_ids,), self.device)
            goal_rot = self.object.data.root_quat_w[env_ids].clone()

        self.object_goal_pos[env_ids] = goal_pos
        self.object_goal_rot[env_ids] = goal_rot
        goal_state[:, 0:3] = goal_pos + self.scene.env_origins[env_ids]
        goal_state[:, 3:7] = goal_rot
        goal_state[:, 7:13] = 0.0
        self.goal_object.write_root_state_to_sim(goal_state, env_ids)

    def _target_volume_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
        maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)
        center = (mins + maxs) * 0.5
        half_range = (maxs - mins) * 0.5 * self.cfg.target_volume_region_scale
        return center - half_range, center + half_range

    def _sample_object_quat(self, num_ids: int) -> torch.Tensor:
        quat = torch.zeros(num_ids, 4, device=self.device)
        quat[:, 0] = 1.0
        if self.cfg.randomize_object_rotation:
            quat = self._sample_random_quat(num_ids)
        return quat

    def _sample_random_quat(self, num_ids: int) -> torch.Tensor:
        uvw = sample_uniform(0.0, 1.0, (num_ids, 3), self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 1])
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 1])
        q_y = torch.sqrt(uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 2])
        q_z = torch.sqrt(uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 2])
        return torch.stack((q_w, q_x, q_y, q_z), dim=-1)

    def _sample_delta_quat(self, quat_wxyz: torch.Tensor, delta_rotation_degrees: float) -> torch.Tensor:
        random_direction = sample_uniform(0.0, 1.0, (quat_wxyz.shape[0], 3), self.device)
        random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True).clamp_min(1e-6)
        delta_rotation_radians = math.radians(delta_rotation_degrees)
        angle = sample_uniform(-delta_rotation_radians, delta_rotation_radians, (quat_wxyz.shape[0],), self.device)
        delta_quat = quat_from_angle_axis(angle, random_direction)
        return quat_mul(quat_wxyz, delta_quat)

    def _make_keypoint_offsets(self, object_size: tuple[float, float, float] | None = None) -> torch.Tensor:
        if object_size is None:
            object_size = (
                self.cfg.object_base_size,
                self.cfg.object_base_size,
                self.cfg.object_base_size,
            )
        offsets = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
            device=self.device,
        )
        size = torch.tensor(object_size, device=self.device, dtype=offsets.dtype)
        return offsets * (0.5 * self.cfg.keypoint_scale * size)

    def _make_grasp_bounding_box_offsets(self) -> torch.Tensor:
        offsets = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        return offsets * (0.5 * self.cfg.keypoint_scale * self.cfg.object_base_size)

    def _compute_keypoints(
        self,
        pos: torch.Tensor,
        quat_wxyz: torch.Tensor,
        object_scales: torch.Tensor,
        keypoint_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos.dim() == 3 and pos.shape[1] == 1:
            pos = pos[:, 0, :]
        if quat_wxyz.dim() == 3 and quat_wxyz.shape[1] == 1:
            quat_wxyz = quat_wxyz[:, 0, :]
        if object_scales.dim() == 3 and object_scales.shape[1] == 1:
            object_scales = object_scales[:, 0, :]

        if quat_wxyz.shape[0] == 1 and pos.shape[0] > 1:
            quat_wxyz = quat_wxyz.expand(pos.shape[0], -1)
        elif quat_wxyz.shape[0] != pos.shape[0]:
            raise RuntimeError(
                f"Keypoint batch mismatch: pos shape={tuple(pos.shape)}, quat shape={tuple(quat_wxyz.shape)}"
            )
        if object_scales.shape[0] == 1 and pos.shape[0] > 1:
            object_scales = object_scales.expand(pos.shape[0], -1)
        elif object_scales.shape[0] != pos.shape[0]:
            raise RuntimeError(
                f"Keypoint scale batch mismatch: pos shape={tuple(pos.shape)}, "
                f"scale shape={tuple(object_scales.shape)}"
            )

        if keypoint_offsets is None:
            keypoint_offsets = self.keypoint_offsets
        offsets = keypoint_offsets.unsqueeze(0).expand(pos.shape[0], -1, -1) * object_scales[:, None, :]
        quat = quat_wxyz.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        rotated_offsets = quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape[0], -1, 3)
        return pos.unsqueeze(1) + rotated_offsets

    def _make_keypoint_debug_draw(self):
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            from isaacsim.util.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception:
            from omni.debugdraw import acquire_debug_draw_interface

            return acquire_debug_draw_interface()

    def _visualize_debug_shapes(self) -> None:
        if getattr(self, "keypoint_debug_draw", None) is None:
            return

        self.keypoint_debug_draw.clear_points()
        if hasattr(self.keypoint_debug_draw, "clear_lines"):
            self.keypoint_debug_draw.clear_lines()

        if self.cfg.debug_keypoints:
            object_keypoints_w = self.object_keypoints + self.scene.env_origins[:, None, :]
            goal_keypoints_w = self.goal_keypoints + self.scene.env_origins[:, None, :]
            object_points = [tuple(point) for point in object_keypoints_w.reshape(-1, 3).detach().cpu().tolist()]
            goal_points = [tuple(point) for point in goal_keypoints_w.reshape(-1, 3).detach().cpu().tolist()]
            point_size = max(1.0, self.cfg.debug_keypoint_radius * 1000.0)

            self.keypoint_debug_draw.draw_points(
                object_points + goal_points,
                [(0.1, 0.45, 1.0, 1.0)] * len(object_points) + [(0.1, 1.0, 0.25, 1.0)] * len(goal_points),
                [point_size] * (len(object_points) + len(goal_points)),
            )

        if self.cfg.debug_fingertips:
            fingertip_pos_w = self.fingertip_pos + self.scene.env_origins[:, None, :]
            fingertip_points = [tuple(point) for point in fingertip_pos_w.reshape(-1, 3).detach().cpu().tolist()]
            point_size = max(1.0, float(self.cfg.debug_fingertip_radius) * 1000.0)
            self.keypoint_debug_draw.draw_points(
                fingertip_points,
                [(1.0, 0.85, 0.1, 1.0)] * len(fingertip_points),
                [point_size] * len(fingertip_points),
            )

        if self.cfg.debug_grasp_bounding_box:
            self._visualize_grasp_bounding_box()

    def _visualize_grasp_bounding_box(self) -> None:
        object_corners = self._compute_keypoints(
            self.object_pos, self.object_rot, self.object_scales, self.grasp_bounding_box_offsets
        )
        goal_corners = self._compute_keypoints(
            self.object_goal_pos, self.object_goal_rot, self.object_scales, self.grasp_bounding_box_offsets
        )
        object_corners_w = object_corners + self.scene.env_origins[:, None, :]
        goal_corners_w = goal_corners + self.scene.env_origins[:, None, :]
        edge_indices = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        point_size = max(1.0, self.cfg.debug_keypoint_radius * 700.0)
        object_points = [tuple(point) for point in object_corners_w.reshape(-1, 3).detach().cpu().tolist()]
        goal_points = [tuple(point) for point in goal_corners_w.reshape(-1, 3).detach().cpu().tolist()]
        self.keypoint_debug_draw.draw_points(
            object_points + goal_points,
            [(0.1, 0.45, 1.0, 1.0)] * len(object_points) + [(0.1, 1.0, 0.25, 1.0)] * len(goal_points),
            [point_size] * (len(object_points) + len(goal_points)),
        )

        if not hasattr(self.keypoint_debug_draw, "draw_lines"):
            return
        line_starts = []
        line_ends = []
        line_colors = []
        for corners, color in (
            (object_corners_w, (0.1, 0.45, 1.0, 1.0)),
            (goal_corners_w, (0.1, 1.0, 0.25, 1.0)),
        ):
            corners_cpu = corners.detach().cpu()
            for env_idx in range(corners_cpu.shape[0]):
                for start_idx, end_idx in edge_indices:
                    line_starts.append(tuple(corners_cpu[env_idx, start_idx].tolist()))
                    line_ends.append(tuple(corners_cpu[env_idx, end_idx].tolist()))
                    line_colors.append(color)
        line_widths = [float(self.cfg.debug_grasp_bounding_box_line_width)] * len(line_starts)
        self.keypoint_debug_draw.draw_lines(line_starts, line_ends, line_colors, line_widths)

    def _distance_delta_rewards(self, lifted_object: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])

        fingertip_deltas = torch.clamp(fingertip_deltas_closest, 0.0, 10.0) * self.finger_rew_coeffs
        fingertip_delta_reward = torch.sum(fingertip_deltas, dim=-1) * (~lifted_object)

        hand_delta_penalty = torch.clamp(hand_deltas_furthest, -10.0, 0.0) * (~lifted_object)
        hand_delta_penalty = hand_delta_penalty * self.fingertip_offsets.shape[0]

        fingertip_delta_reward = fingertip_delta_reward * self.cfg.distance_delta_rew_scale
        hand_delta_penalty = hand_delta_penalty * self.cfg.hand_delta_penalty_scale
        return fingertip_delta_reward, hand_delta_penalty

    def _lifting_reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_pos[:, 2]
        lifting_reward = torch.clamp(z_lift, 0.0, 0.5)

        crossed_lift_height = z_lift > self.cfg.lifting_bonus_threshold

        # Contact-gated latch: the dense lift reward runs until the contact-gated
        # "lifted" label fires, so there is no gap between the dense reward and
        # the one-shot bonus.  The reward *value* is a pure function of height —
        # contact is only required for the shutoff + bonus.
        contact_gate = self.hand_object_contact if self.cfg.require_lift_contact else torch.ones_like(crossed_lift_height)
        self.contact_gated_lift = crossed_lift_height & contact_gate
        lifted_object = self.contact_gated_lift | self.lifted_object
        just_lifted = lifted_object & (~self.lifted_object)
        self.just_lifted = just_lifted
        lift_bonus_reward = self.cfg.lifting_bonus * just_lifted.float()

        # Dense reward shuts off when lifted_object fires (same moment as bonus).
        lifting_reward = lifting_reward * (~lifted_object)

        self.lifted_object = lifted_object
        lifting_reward = lifting_reward * self.cfg.lifting_rew_scale
        return lifting_reward, lift_bonus_reward, lifted_object

    def _keypoint_reward(self, lifted_object: torch.Tensor) -> torch.Tensor:
        keypoint_deltas = self._reward_closest_keypoint_max_dist() - self._reward_keypoints_max_dist()
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        self.closest_keypoint_max_dist_fixed_size = torch.minimum(
            self.closest_keypoint_max_dist_fixed_size,
            self.keypoints_max_dist_fixed_size,
        )
        keypoint_deltas = torch.clamp(keypoint_deltas, 0.0, 100.0)
        return keypoint_deltas * lifted_object * self.cfg.keypoint_rew_scale

    def _reward_keypoints_max_dist(self) -> torch.Tensor:
        if self.cfg.fixed_size_keypoint_reward:
            return self.keypoints_max_dist_fixed_size
        return self.keypoints_max_dist

    def _reward_closest_keypoint_max_dist(self) -> torch.Tensor:
        if self.cfg.fixed_size_keypoint_reward:
            return self.closest_keypoint_max_dist_fixed_size
        return self.closest_keypoint_max_dist

    def _action_penalties(self) -> tuple[torch.Tensor, torch.Tensor]:
        arm_actions_penalty = (
            torch.sum(torch.abs(self.policy_dof_vel[:, :7]), dim=-1) * self.cfg.arm_actions_penalty_scale
        )
        hand_actions_penalty = (
            torch.sum(torch.abs(self.policy_dof_vel[:, 7:]), dim=-1) * self.cfg.hand_actions_penalty_scale
        )
        return -arm_actions_penalty, -hand_actions_penalty
