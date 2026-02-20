"""Shared evaluation helpers and metric definitions.

Evaluation metrics used by teacher eval and distillation eval:

1) `lift_success_rate`
   - Episode-level success: object must be above lift height threshold,
     pass contact gating, and satisfy hold-time gating.
   - Reported as the mean over evaluated episodes.

2) `unsafe_episode_rate`
   - Episode is unsafe if any raw `out_of_reach` step occurs.
   - Formula: `unsafe_episode_rate = (# unsafe episodes) / (# evaluated episodes)`.

3) `unsafe_reason_pct/<category>`
   - Category breakdown over unsafe episodes only.
   - For each category:
     `100 * (# unsafe episodes first-labeled as that category) / (# unsafe episodes)`.
   - Categories:
     `object_out_of_bound`: object_x_out/object_y_out/object_too_low grouped.
     `hand_too_far`: standalone category.
     `harmful_collision`: hand_too_close/arm_table_contact grouped.
     `palm_flipped`: palm orientation flip termination.
"""

from __future__ import annotations

import torch


UNSAFE_REASON_NAMES: tuple[str, ...] = (
    "object_out_of_bound",
    "hand_too_far",
    "harmful_collision",
    "palm_flipped",
)


def as_bool_mask(values, num_envs: int, device: torch.device) -> torch.Tensor:
    try:
        mask = torch.as_tensor(values, device=device, dtype=torch.bool).flatten()
    except Exception:
        return torch.zeros((num_envs,), dtype=torch.bool, device=device)
    if mask.numel() == 1:
        mask = mask.repeat(num_envs)
    if mask.numel() < num_envs:
        padded = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        padded[: mask.numel()] = mask
        return padded
    return mask[:num_envs]


def compute_out_of_reach_reason_masks(ov_env, num_envs: int, device: torch.device) -> dict[str, torch.Tensor]:
    zeros = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    masks = {name: zeros.clone() for name in UNSAFE_REASON_NAMES}
    if not hasattr(ov_env, "object_pos"):
        return masks

    object_out_of_bound = zeros.clone()
    hand_too_far = zeros.clone()
    hand_too_close = zeros.clone()
    arm_table_contact = zeros.clone()
    palm_flipped = zeros.clone()

    try:
        object_outside_upper_x = ov_env.object_pos[:, 0] > (ov_env.cfg.x_center + ov_env.cfg.x_width / 2.0)
        object_outside_lower_x = ov_env.object_pos[:, 0] < (ov_env.cfg.x_center - ov_env.cfg.x_width / 2.0)
        object_outside_upper_y = ov_env.object_pos[:, 1] > (ov_env.cfg.y_center + ov_env.cfg.y_width / 2.0)
        object_outside_lower_y = ov_env.object_pos[:, 1] < (ov_env.cfg.y_center - ov_env.cfg.y_width / 2.0)
        object_too_low = ov_env.object_pos[:, 2] < 0.2
        object_out_of_bound = (
            object_outside_upper_x
            | object_outside_lower_x
            | object_outside_upper_y
            | object_outside_lower_y
            | object_too_low
        )
    except Exception:
        pass

    try:
        bbox_margin = getattr(ov_env.cfg, "hand_bbox_margin", 0.0)
        table_half_x = ov_env.cfg.table_size_x * 0.5 + bbox_margin
        table_half_y = ov_env.cfg.table_size_y * 0.5 + bbox_margin

        if hasattr(ov_env, "table_pos"):
            table_pos = ov_env.table_pos
            table_pos_z = ov_env.table_pos[:, 2]
        else:
            table_pos = ov_env.table.data.root_pos_w - ov_env.scene.env_origins
            table_pos_z = table_pos[:, 2]
        table_top_z = table_pos_z + ov_env.cfg.table_size_z * 0.5

        middle_pos = ov_env.robot.data.body_pos_w[:, ov_env.middle_link_0_body_idx] - ov_env.scene.env_origins
        middle_x_out = (middle_pos[:, 0] > (table_pos[:, 0] + table_half_x)) | (
            middle_pos[:, 0] < (table_pos[:, 0] - table_half_x)
        )
        middle_y_out = (middle_pos[:, 1] > (table_pos[:, 1] + table_half_y)) | (
            middle_pos[:, 1] < (table_pos[:, 1] - table_half_y)
        )
        middle_z_out = (middle_pos[:, 2] < table_top_z) | (middle_pos[:, 2] > (table_top_z + 1.0 + bbox_margin))
        hand_too_far = middle_x_out | middle_y_out | middle_z_out

        hand_min_z = ov_env.hand_pos[..., 2].min(dim=1).values
        clearance_thresh = table_top_z + 0.01
        hand_too_close = hand_min_z < clearance_thresh
    except Exception:
        pass

    try:
        arm_table_contact = as_bool_mask(ov_env.arm_table_contact_mask, num_envs, device)
    except Exception:
        pass

    try:
        palm_flip_cos = torch.sum(ov_env.palm_direction_vec * ov_env._palm_dir_target_world, dim=-1)
        palm_flipped = palm_flip_cos < ov_env.cfg.palm_flip_cos_thresh
    except Exception:
        pass

    masks["object_out_of_bound"] = object_out_of_bound
    masks["hand_too_far"] = hand_too_far
    masks["harmful_collision"] = hand_too_close | arm_table_contact
    masks["palm_flipped"] = palm_flipped
    return masks


def classify_out_of_reach_reasons(
    ov_env,
    out_of_reach: torch.Tensor,
    reason_names: tuple[str, ...],
    reason_to_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    num_envs = ov_env.num_envs
    reason_masks = compute_out_of_reach_reason_masks(ov_env, num_envs, device)
    reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    unassigned = out_of_reach.clone()
    for name in reason_names:
        idx = reason_to_idx[name]
        mask = reason_masks.get(name, torch.zeros_like(unassigned)) & unassigned
        reason_idx[mask] = idx
        unassigned = unassigned & (~mask)
    return reason_idx


def unsafe_reason_percentages_from_counts(
    counts: dict[str, int], total_unsafe: int, reason_names: tuple[str, ...]
) -> dict[str, float]:
    if total_unsafe <= 0:
        return {name: 0.0 for name in reason_names}
    return {
        name: 100.0 * float(counts.get(name, 0)) / float(total_unsafe)
        for name in reason_names
    }
