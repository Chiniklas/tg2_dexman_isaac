from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) in wxyz format to rotation matrix.

    Args:
        quaternions_wxyz: tensor of shape (..., 4), ordered as [w, x, y, z].

    Returns:
        Rotation matrices with shape (..., 3, 3).
    """
    if quaternions_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_to_matrix expects [..., 4] input in wxyz order.")

    q = quaternions_wxyz
    # Normalize to handle slightly non-unit quaternions from sensors.
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)

    w, x, y, z = q.unbind(dim=-1)

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
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    mat = torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )
    return mat


def euler_to_matrix(euler_zyx: torch.Tensor) -> torch.Tensor:
    """Convert Euler ZYX angles to rotation matrix.

    Input order is [yaw(z), pitch(y), roll(x)].
    """
    if euler_zyx.shape[-1] != 3:
        raise ValueError("euler_to_matrix expects [..., 3] input in [yaw, pitch, roll] order.")

    yaw, pitch, roll = euler_zyx.unbind(dim=-1)

    cz = torch.cos(yaw)
    sz = torch.sin(yaw)
    cy = torch.cos(pitch)
    sy = torch.sin(pitch)
    cx = torch.cos(roll)
    sx = torch.sin(roll)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    m00 = cz * cy
    m01 = cz * sy * sx - sz * cx
    m02 = cz * sy * cx + sz * sx

    m10 = sz * cy
    m11 = sz * sy * sx + cz * cx
    m12 = sz * sy * cx - cz * sx

    m20 = -sy
    m21 = cy * sx
    m22 = cy * cx

    mat = torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )
    return mat
