"""Abstract interface for dense reconstruction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class DenseBackend(ABC):
    """Base class for dense reconstruction methods."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def reconstruct(
        self,
        image_dir: Path,
        sparse_result: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Produce dense depth maps and/or point clouds.

        Returns a dict containing at least:
            - ``point_cloud_path``: path to a dense .ply point cloud
            - ``colmap_dir``: path to a COLMAP-format scene directory
            - ``depth_maps``: dict mapping filenames to depth arrays (optional)
        """
        ...
