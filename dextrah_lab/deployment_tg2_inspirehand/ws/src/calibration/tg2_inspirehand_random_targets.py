"""
Pose target generation utilities for TG2 Inspirehand calibration.

Defines a simple interpolated trajectory between two 6D poses:
    [x, y, z, yaw, pitch, roll] in ZYX order (radians).
"""

from __future__ import annotations

from typing import List

import torch


def interpolate_poses(
    start_pose: torch.Tensor, end_pose: torch.Tensor, num_steps: int, include_endpoints: bool = True
) -> List[torch.Tensor]:
    """
    Linearly interpolate between two poses.

    Args:
        start_pose: shape (1, 6) tensor [x, y, z, yaw, pitch, roll].
        end_pose: shape (1, 6) tensor [x, y, z, yaw, pitch, roll].
        num_steps: number of interpolation steps.
        include_endpoints: include start/end in the returned list.

    Returns:
        List of pose tensors, each shape (1, 6).
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")

    device = start_pose.device
    dtype = start_pose.dtype

    if include_endpoints:
        alphas = torch.linspace(0.0, 1.0, steps=num_steps, device=device, dtype=dtype)
    else:
        if num_steps <= 2:
            return []
        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 2, device=device, dtype=dtype)[1:-1]

    poses: List[torch.Tensor] = []
    for alpha in alphas:
        pose = (1.0 - alpha) * start_pose + alpha * end_pose
        poses.append(pose.clone())
    return poses


def build_calibration_trajectory(
    home_pose: torch.Tensor, target_pose: torch.Tensor, num_steps: int
) -> List[torch.Tensor]:
    """Convenience wrapper for calibration trajectories."""
    return interpolate_poses(home_pose, target_pose, num_steps, include_endpoints=True)
