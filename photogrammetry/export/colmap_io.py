"""COLMAP binary format read/write utilities.

Writes cameras.bin, images.bin, and points3D.bin in the COLMAP binary
format so that downstream tools (MILo, 3DGS, etc.) can consume the data.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from photogrammetry.utils.geometry import rotation_matrix_to_quaternion


def write_colmap_binary(
    cameras: dict[str, dict[str, Any]],
    points_3d: np.ndarray,
    image_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Write a COLMAP-format sparse reconstruction.

    Creates the directory structure:
        output_dir/
            sparse/0/
                cameras.bin
                images.bin
                points3D.bin
            images/  (symlinks or copies)

    Args:
        cameras: Per-image camera parameters with 'extrinsic' and 'intrinsic'.
        points_3d: (N, 3) world points.
        image_paths: Paths to original images.
        output_dir: Root output directory.

    Returns:
        Path to the sparse/0/ directory.
    """
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        dst = images_dir / p.name
        if not dst.exists():
            try:
                dst.symlink_to(p.resolve())
            except OSError:
                import shutil
                shutil.copy2(p, dst)

    _write_cameras_bin(cameras, sparse_dir / "cameras.bin")
    _write_images_bin(cameras, sparse_dir / "images.bin")
    _write_points3d_bin(points_3d, sparse_dir / "points3D.bin")

    logger.info(
        "Wrote COLMAP binary: {} cameras, {} points -> {}",
        len(cameras), len(points_3d), sparse_dir,
    )
    return sparse_dir


def _write_cameras_bin(
    cameras: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Write cameras.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))

        for idx, (name, cam) in enumerate(cameras.items()):
            intrinsic = np.array(cam["intrinsic"])
            fx = float(intrinsic[0, 0]) if intrinsic.ndim == 2 else 1000.0
            fy = float(intrinsic[1, 1]) if intrinsic.ndim == 2 else fx
            cx = float(intrinsic[0, 2]) if intrinsic.ndim == 2 else 0.0
            cy = float(intrinsic[1, 2]) if intrinsic.ndim == 2 else 0.0

            width = int(cx * 2) if cx > 0 else 1600
            height = int(cy * 2) if cy > 0 else 1200

            camera_id = idx + 1
            model_id = 1  # PINHOLE
            num_params = 4

            f.write(struct.pack("<I", camera_id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", width))
            f.write(struct.pack("<Q", height))
            for p in [fx, fy, cx, cy]:
                f.write(struct.pack("<d", p))


def _write_images_bin(
    cameras: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Write images.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))

        for idx, (name, cam) in enumerate(cameras.items()):
            image_id = idx + 1
            camera_id = idx + 1

            extrinsic = np.array(cam["extrinsic"])
            R = extrinsic[:3, :3] if extrinsic.shape == (4, 4) else extrinsic[:3, :3]
            t = extrinsic[:3, 3] if extrinsic.shape == (4, 4) else np.zeros(3)

            q = rotation_matrix_to_quaternion(R)

            f.write(struct.pack("<I", image_id))
            for qi in q:
                f.write(struct.pack("<d", float(qi)))
            for ti in t:
                f.write(struct.pack("<d", float(ti)))
            f.write(struct.pack("<I", camera_id))

            name_bytes = name.encode("utf-8") + b"\x00"
            f.write(name_bytes)

            # No 2D points for now
            f.write(struct.pack("<Q", 0))


def _write_points3d_bin(
    points_3d: np.ndarray,
    path: Path,
) -> None:
    """Write points3D.bin in COLMAP binary format."""
    with open(path, "wb") as f:
        n_points = min(len(points_3d), 100000)
        f.write(struct.pack("<Q", n_points))

        for pid in range(n_points):
            xyz = points_3d[pid]
            point_id = pid + 1

            f.write(struct.pack("<Q", point_id))
            for c in xyz:
                f.write(struct.pack("<d", float(c)))

            # RGB color
            f.write(struct.pack("<BBB", 128, 128, 128))

            # Reprojection error
            f.write(struct.pack("<d", 0.0))

            # Empty track (no 2D observations)
            f.write(struct.pack("<Q", 0))


def read_colmap_cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
    """Read cameras.bin and return a dict of camera_id -> params."""
    cameras: dict[int, dict[str, Any]] = {}
    with open(path, "rb") as f:
        (n_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_cameras):
            (camera_id,) = struct.unpack("<I", f.read(4))
            (model_id,) = struct.unpack("<i", f.read(4))
            (width,) = struct.unpack("<Q", f.read(8))
            (height,) = struct.unpack("<Q", f.read(8))

            num_params = {1: 4, 0: 3, 2: 4, 3: 5}.get(model_id, 4)
            params = [struct.unpack("<d", f.read(8))[0] for _ in range(num_params)]

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras
