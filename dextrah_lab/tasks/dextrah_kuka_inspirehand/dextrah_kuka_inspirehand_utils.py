# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Imports
from typing import Optional

import omni.usd
import torch

def assert_equals(a, b) -> None:
    # Saves space typing out the full assert and text
    assert a == b, f"{a} != {b}"

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def compute_absolute_action(
    raw_actions: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    N, D = raw_actions.shape
    assert_equals(lower_limits.shape, (D,))
    assert_equals(upper_limits.shape, (D,))

    # Apply actions to hand
    absolute_action = scale(
        x=raw_actions,
        lower=lower_limits,
        upper=upper_limits,
    )
    absolute_action = tensor_clamp(
        t=absolute_action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return absolute_action

def compute_delta_action(
    raw_actions: torch.Tensor,
    base_action: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    delta_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Apply delta actions around a base target and clamp to joint limits."""
    N, D = raw_actions.shape
    assert_equals(base_action.shape, (N, D))
    assert_equals(lower_limits.shape, (D,))
    assert_equals(upper_limits.shape, (D,))

    if not torch.is_tensor(delta_scale):
        delta_scale = torch.tensor(
            delta_scale,
            device=raw_actions.device,
            dtype=raw_actions.dtype,
        )
    delta = raw_actions * delta_scale
    action = base_action + delta
    action = tensor_clamp(
        t=action,
        min_t=lower_limits,
        max_t=upper_limits,
    )

    return action

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    return torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )


def print_contact_debug(sensor, tag: str) -> None:
    """Helper to print env/contact pairs (link vs filter target)."""
    data = sensor.data
    if data is None or data.force_matrix_w is None:
        return
    body_names = getattr(sensor, "body_names", [])
    filters = getattr(sensor.cfg, "filter_prim_paths_expr", [])
    fm = data.force_matrix_w
    nz = (fm.abs().sum(-1) > 1e-4).nonzero(as_tuple=False)
    if len(nz) == 0:
        return
    pairs = []
    for env_idx, body_idx, filter_idx in nz.tolist():
        body_name = body_names[body_idx] if body_idx < len(body_names) else f"body_{body_idx}"
        filter_name = filters[filter_idx] if filter_idx < len(filters) else f"filter_{filter_idx}"
        pairs.append((env_idx, body_name, filter_name))
    print(f"contact pairs: {pairs}")


def print_prim_tree_once(env, root_path: str = "/World/envs/env_0", max_depth: Optional[int] = None) -> None:
    if getattr(env, "_prim_tree_printed", False):
        return
    env._prim_tree_printed = True
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[PRIM TREE] stage unavailable")
        return
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        print(f"[PRIM TREE] invalid root prim: {root_path}")
        return
    print(f"[PRIM TREE] root: {root_path}")

    def _recurse(prim, depth: int) -> None:
        indent = "  " * depth
        print(f"{indent}{prim.GetPath()} ({prim.GetTypeName()})")
        if max_depth is not None and depth >= max_depth:
            return
        for child in prim.GetChildren():
            _recurse(child, depth + 1)

    _recurse(root_prim, 0)
