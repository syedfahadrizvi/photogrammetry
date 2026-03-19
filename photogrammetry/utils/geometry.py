"""Camera models, SE3 transforms, and quaternion utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quaternion_to_rotation_matrix(q: NDArray) -> NDArray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_matrix_to_quaternion(R: NDArray) -> NDArray:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def compose_se3(R: NDArray, t: NDArray) -> NDArray:
    """Build a 4x4 SE3 transformation matrix from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_se3(T: NDArray) -> NDArray:
    """Invert a 4x4 SE3 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def project_points(
    points_3d: NDArray,
    K: NDArray,
    R: NDArray,
    t: NDArray,
) -> NDArray:
    """Project 3D world points to 2D image coordinates.

    Args:
        points_3d: (N, 3) world coordinates
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation (world-to-camera)
        t: (3,) translation (world-to-camera)

    Returns:
        (N, 2) pixel coordinates
    """
    cam_pts = (R @ points_3d.T).T + t
    proj = (K @ cam_pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def compute_reprojection_error(
    points_3d: NDArray,
    points_2d: NDArray,
    K: NDArray,
    R: NDArray,
    t: NDArray,
) -> float:
    """Mean reprojection error in pixels."""
    projected = project_points(points_3d, K, R, t)
    return float(np.mean(np.linalg.norm(projected - points_2d, axis=1)))
