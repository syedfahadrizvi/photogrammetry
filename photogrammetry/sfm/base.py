"""Abstract interface for Structure from Motion backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class SfMBackend(ABC):
    """Base class for SfM methods."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def estimate(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Estimate camera poses and sparse 3D structure.

        Returns a dict containing at least:
            - ``cameras``: dict mapping image names to camera parameters
            - ``points_3d``: (N, 3) sparse point cloud
            - ``output_dir``: path where results are cached
        """
        ...
