"""Export meshes and point clouds to PLY, OBJ, and GLB formats."""

from __future__ import annotations

from pathlib import Path

import trimesh
from loguru import logger


def export_mesh(
    mesh_path: Path,
    output_dir: Path,
    formats: list[str] | None = None,
    include_textures: bool = True,
) -> dict[str, Path]:
    """Convert a mesh to one or more output formats.

    Args:
        mesh_path: Path to the source mesh (PLY, OBJ, etc.)
        output_dir: Directory for exported files.
        formats: List of target formats (ply, obj, glb). Defaults to [ply, obj].
        include_textures: Whether to include textures in formats that support them.

    Returns:
        Dict mapping format name to output path.
    """
    if formats is None:
        formats = ["ply", "obj"]

    mesh_path = Path(mesh_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(mesh_path), process=False)
    stem = mesh_path.stem
    exported: dict[str, Path] = {}

    for fmt in formats:
        fmt = fmt.lower().strip(".")
        out_path = output_dir / f"{stem}.{fmt}"

        if fmt == "ply":
            mesh.export(str(out_path), file_type="ply")
        elif fmt == "obj":
            mesh.export(str(out_path), file_type="obj", include_texture=include_textures)
        elif fmt == "glb":
            mesh.export(str(out_path), file_type="glb")
        elif fmt == "stl":
            mesh.export(str(out_path), file_type="stl")
        else:
            logger.warning("Unsupported export format: {}", fmt)
            continue

        exported[fmt] = out_path
        logger.info("Exported {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)

    return exported


def export_point_cloud(
    points_path: Path,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Export a point cloud to various formats.

    Args:
        points_path: Path to the source point cloud (PLY, etc.)
        output_dir: Directory for exported files.
        formats: Target formats (ply, xyz). Defaults to [ply].

    Returns:
        Dict mapping format name to output path.
    """
    if formats is None:
        formats = ["ply"]

    import open3d as o3d

    points_path = Path(points_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(points_path))
    stem = points_path.stem
    exported: dict[str, Path] = {}

    for fmt in formats:
        fmt = fmt.lower().strip(".")
        out_path = output_dir / f"{stem}.{fmt}"

        if fmt == "ply":
            o3d.io.write_point_cloud(str(out_path), pcd)
        elif fmt == "xyz":
            o3d.io.write_point_cloud(str(out_path), pcd)
        else:
            logger.warning("Unsupported point cloud format: {}", fmt)
            continue

        exported[fmt] = out_path
        logger.info("Exported point cloud {}", out_path)

    return exported
