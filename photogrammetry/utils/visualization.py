"""Point cloud and mesh visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def visualize_point_cloud(
    points: NDArray,
    colors: NDArray | None = None,
    window_name: str = "Point Cloud",
) -> None:
    """Display a point cloud interactively using Open3D."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_mesh(mesh_path: str | Path, window_name: str = "Mesh") -> None:
    """Display a mesh file interactively using Open3D."""
    import open3d as o3d

    mesh_path = Path(mesh_path)
    logger.info("Loading mesh from {}", mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name=window_name)


def save_point_cloud_ply(
    path: str | Path,
    points: NDArray,
    colors: NDArray | None = None,
) -> None:
    """Save a point cloud to PLY format."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(path), pcd)
    logger.info("Saved point cloud ({} points) to {}", len(points), path)
