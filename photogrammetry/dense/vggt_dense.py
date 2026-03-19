"""VGGT dense depth maps and 3D point cloud reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from photogrammetry.dense.base import DenseBackend


class VGGTDense(DenseBackend):
    """Extract dense depth maps and point clouds from VGGT predictions.

    VGGT produces per-pixel depth and 3D point maps as part of its
    feed-forward pass. This module extracts those predictions and converts
    them to COLMAP format for downstream surface reconstruction.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def reconstruct(
        self,
        image_dir: Path,
        sparse_result: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = sparse_result.get("predictions")
        cameras = sparse_result["cameras"]
        image_paths = sparse_result.get("image_paths", [])

        depth_maps = {}
        if self.cfg.get("use_depth_maps", True) and predictions is not None:
            depth_maps = self._extract_depth_maps(predictions, image_paths, output_dir)

        point_cloud = self._build_dense_point_cloud(
            predictions, sparse_result, output_dir
        )

        colmap_dir = self._export_colmap_format(
            cameras, point_cloud, image_paths, output_dir
        )

        point_cloud_path = output_dir / "dense_points.ply"
        self._save_ply(point_cloud, point_cloud_path)

        logger.info(
            "Dense reconstruction: {} depth maps, {} points, COLMAP dir at {}",
            len(depth_maps), len(point_cloud), colmap_dir,
        )

        return {
            "point_cloud_path": point_cloud_path,
            "colmap_dir": colmap_dir,
            "depth_maps": depth_maps,
            "output_dir": output_dir,
        }

    def _extract_depth_maps(
        self,
        predictions: dict,
        image_paths: list[Path],
        output_dir: Path,
    ) -> dict[str, np.ndarray]:
        """Extract per-image depth maps from VGGT predictions."""
        depth_dir = output_dir / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)

        depth_key = "depth" if "depth" in predictions else "depth_out"
        if depth_key not in predictions:
            logger.warning("No depth maps in VGGT predictions")
            return {}

        depths_tensor = predictions[depth_key][0].cpu().numpy()
        depth_maps = {}

        for i, path in enumerate(image_paths):
            if i < len(depths_tensor):
                depth = depths_tensor[i]
                depth_path = depth_dir / f"{path.stem}_depth.npy"
                np.save(depth_path, depth)
                depth_maps[path.name] = depth

        logger.info("Extracted {} depth maps", len(depth_maps))
        return depth_maps

    def _build_dense_point_cloud(
        self,
        predictions: dict | None,
        sparse_result: dict[str, Any],
        output_dir: Path,
    ) -> np.ndarray:
        """Build a dense point cloud from VGGT world points or sparse points."""
        max_points = self.cfg.get("max_points", 500000)
        conf_threshold = self.cfg.get("point_confidence_threshold", 0.5)

        if (
            predictions is not None
            and self.cfg.get("use_point_maps", True)
            and "world_points_out" in predictions
        ):
            pts = predictions["world_points_out"][0].cpu().numpy().reshape(-1, 3)

            if "world_points_conf" in predictions:
                conf = predictions["world_points_conf"][0].cpu().numpy().reshape(-1)
                mask = conf > conf_threshold
                pts = pts[mask]

            if len(pts) > max_points:
                indices = np.random.choice(len(pts), max_points, replace=False)
                pts = pts[indices]

            return pts

        return sparse_result.get("points_3d", np.empty((0, 3)))

    def _export_colmap_format(
        self,
        cameras: dict[str, dict[str, Any]],
        points_3d: np.ndarray,
        image_paths: list[Path],
        output_dir: Path,
    ) -> Path:
        """Export scene to COLMAP binary format for MILo/3DGS consumption."""
        from photogrammetry.export.colmap_io import write_colmap_binary

        colmap_dir = output_dir / "colmap"
        write_colmap_binary(cameras, points_3d, image_paths, colmap_dir)
        return colmap_dir

    def _save_ply(self, points: np.ndarray, path: Path) -> None:
        """Save a point cloud to PLY."""
        from photogrammetry.utils.visualization import save_point_cloud_ply

        if len(points) > 0:
            save_point_cloud_ply(path, points)
