"""MILo: Mesh-in-the-Loop Gaussian Splatting surface reconstruction.

Wraps the MILo training and mesh extraction pipeline. Expects COLMAP-format
input (images/ + sparse/0/ with cameras.bin, images.bin, points3D.bin).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig

from photogrammetry.surface.base import SurfaceBackend

_MILO_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "milo"


class MILoBackend(SurfaceBackend):
    """Train 3D Gaussian Splatting with MILo and extract mesh."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._milo_dir = _MILO_DIR
        if not self._milo_dir.exists():
            logger.warning(
                "MILo not found at {}. Clone it with: "
                "git clone https://github.com/Anttwo/MILo.git {}",
                self._milo_dir, self._milo_dir,
            )

    def reconstruct(
        self,
        colmap_dir: Path,
        output_dir: Path,
    ) -> Path:
        """Run MILo training and mesh extraction.

        Args:
            colmap_dir: COLMAP-format scene directory containing
                        images/ and sparse/0/{cameras,images,points3D}.bin
            output_dir: Where to write the Gaussian model and mesh.

        Returns:
            Path to the extracted mesh file.
        """
        colmap_dir = Path(colmap_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = output_dir / "gs_model"
        mesh_path = output_dir / "mesh.ply"

        self._train(colmap_dir, model_dir)
        self._extract_mesh(colmap_dir, model_dir, mesh_path)

        return mesh_path

    def _train(self, colmap_dir: Path, model_dir: Path) -> None:
        """Run MILo Gaussian Splatting training."""
        iterations = self.cfg.get("iterations", 15000)
        imp_metric = self.cfg.get("imp_metric", "indoor")
        rasterizer = self.cfg.get("rasterizer", "radegs")

        train_script = self._milo_dir / "milo" / "train.py"
        if not train_script.exists():
            train_script = self._milo_dir / "train.py"

        cmd = [
            sys.executable, str(train_script),
            "-s", str(colmap_dir),
            "-m", str(model_dir),
            "--imp_metric", imp_metric,
            "--rasterizer", rasterizer,
            "--iterations", str(iterations),
        ]

        logger.info("MILo training: {} iterations (metric={})", iterations, imp_metric)
        logger.debug("Command: {}", " ".join(cmd))

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(self._milo_dir),
        )
        logger.debug("MILo train output (last 1000 chars): {}", result.stdout[-1000:])
        logger.info("MILo training complete")

    def _extract_mesh(
        self,
        colmap_dir: Path,
        model_dir: Path,
        mesh_path: Path,
    ) -> None:
        """Extract mesh from trained Gaussian model using MILo SDF extraction."""
        rasterizer = self.cfg.get("rasterizer", "radegs")

        extract_script = self._milo_dir / "milo" / "mesh_extract_sdf.py"
        if not extract_script.exists():
            extract_script = self._milo_dir / "mesh_extract_sdf.py"

        cmd = [
            sys.executable, str(extract_script),
            "-s", str(colmap_dir),
            "-m", str(model_dir),
            "--rasterizer", rasterizer,
            "--output_mesh", str(mesh_path),
        ]

        logger.info("MILo mesh extraction")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(self._milo_dir),
        )
        logger.debug("MILo extract output: {}", result.stdout[-1000:])

        if not mesh_path.exists():
            candidates = sorted(model_dir.glob("**/*.ply"))
            if candidates:
                import shutil
                shutil.copy2(candidates[-1], mesh_path)

        logger.info("MILo mesh extracted to {}", mesh_path)
