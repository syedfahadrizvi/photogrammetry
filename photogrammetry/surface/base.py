"""Abstract interface for surface reconstruction backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class SurfaceBackend(ABC):
    """Base class for surface reconstruction methods."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def reconstruct(self, *args: Any, **kwargs: Any) -> Path:
        """Run surface reconstruction and return path to the output mesh."""
        ...
