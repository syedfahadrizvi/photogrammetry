"""Abstract interface for feature extraction and matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class FeatureBackend(ABC):
    """Base class for feature extraction and matching."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def extract_and_match(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Extract features from all images and compute pairwise matches.

        Returns a dict containing at least:
            - ``keypoints``: dict[str, NDArray] mapping filenames to (N, 2) arrays
            - ``matches``: list of (img_i, img_j, match_array) tuples
            - ``output_dir``: path where results are cached
        """
        ...
