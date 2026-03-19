"""AliceVision/Meshroom backend for classical photogrammetry.

Wraps `meshroom_batch` CLI to run the full AliceVision pipeline:
feature extraction → SfM → depth maps → meshing → texturing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig

from photogrammetry.surface.base import SurfaceBackend


class AliceVisionBackend(SurfaceBackend):
    """Run AliceVision/Meshroom as a complete end-to-end pipeline."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._meshroom_bin = self._find_meshroom_bin()

    def _find_meshroom_bin(self) -> str:
        """Locate the meshroom_batch executable."""
        explicit = self.cfg.get("meshroom_bin")
        if explicit:
            return str(explicit)

        env_bin = os.environ.get("MESHROOM_BIN")
        if env_bin:
            bin_dir = Path(env_bin)
            for name in ("meshroom_batch", "meshroom_batch.py"):
                candidate = bin_dir / name
                if candidate.exists():
                    return str(candidate)
            return str(bin_dir / "meshroom_batch")

        found = shutil.which("meshroom_batch")
        if found:
            return found

        logger.warning(
            "meshroom_batch not found on PATH. "
            "Set MESHROOM_BIN env var or surface.alicevision.meshroom_bin in config."
        )
        return "meshroom_batch"

    def reconstruct(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> Path:
        """Run the full AliceVision photogrammetry pipeline.

        Args:
            input_dir: Directory containing input images.
            output_dir: Where to write all pipeline outputs.

        Returns:
            Path to the final textured mesh (OBJ).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self.cfg.get("pipeline", "photogrammetry")
        cache_dir = output_dir / "meshroom_cache"

        cmd = [
            self._meshroom_bin,
            "--input", str(input_dir),
            "--inputRecursive",
            "--pipeline", pipeline,
            "--output", str(output_dir / "meshroom_output"),
            "--cache", str(cache_dir),
        ]

        param_overrides = self.cfg.get("param_overrides", {})
        if param_overrides:
            overrides_parts = []
            for node_param, value in param_overrides.items():
                overrides_parts.append(f"{node_param}:{value}")
            if overrides_parts:
                cmd.extend(["--paramOverrides", ";".join(overrides_parts)])

        logger.info("Running AliceVision: {}", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            logger.debug("AliceVision stdout: {}", result.stdout[-2000:] if result.stdout else "")
        except FileNotFoundError:
            raise RuntimeError(
                f"meshroom_batch not found at '{self._meshroom_bin}'. "
                "Install AliceVision/Meshroom or set MESHROOM_BIN."
            )
        except subprocess.CalledProcessError as e:
            logger.error("AliceVision failed:\n{}", e.stderr[-3000:] if e.stderr else "")
            raise RuntimeError(f"AliceVision pipeline failed with exit code {e.returncode}")

        mesh_path = self._find_output_mesh(output_dir / "meshroom_output")
        logger.info("AliceVision complete. Mesh at: {}", mesh_path)
        return mesh_path

    def _find_output_mesh(self, output_dir: Path) -> Path:
        """Find the final textured mesh in AliceVision output."""
        for pattern in ("**/*.obj", "**/*.ply"):
            meshes = sorted(output_dir.glob(pattern))
            if meshes:
                textured = [m for m in meshes if "texture" in m.stem.lower()]
                return textured[0] if textured else meshes[0]

        raise FileNotFoundError(
            f"No mesh found in AliceVision output directory: {output_dir}"
        )
