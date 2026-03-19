"""NeuRodin: Two-stage neural implicit surface reconstruction.

Uses the vendored and patched SDFStudio in third_party/sdfstudio/.
NeuRodin runs a two-stage training process (density → SDF) and extracts
a high-fidelity mesh via Marching Cubes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from photogrammetry.surface.base import SurfaceBackend

_SDFSTUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "sdfstudio"


class NeuRodinBackend(SurfaceBackend):
    """Run NeuRodin two-stage training and mesh extraction."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._sdfstudio_dir = _SDFSTUDIO_DIR
        if not self._sdfstudio_dir.exists():
            logger.warning(
                "SDFStudio not found at {}. See README.md for setup instructions.",
                self._sdfstudio_dir,
            )

    def reconstruct(
        self,
        image_dir: Path,
        sparse_result: dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Run NeuRodin two-stage training and extract mesh.

        Args:
            image_dir: Directory with input images.
            sparse_result: SfM result with cameras and points.
            output_dir: Where to write training outputs and mesh.

        Returns:
            Path to the extracted mesh.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = output_dir / "neurodin_data"
        self._prepare_data(image_dir, sparse_result, data_dir)

        stage1_dir = self._train_stage(
            data_dir, output_dir, stage=1,
            config_name=self.cfg.get("stage1_config", "indoor-small"),
        )

        stage2_dir = self._train_stage(
            data_dir, output_dir, stage=2,
            config_name=self.cfg.get("stage2_config", "indoor-small"),
            load_dir=stage1_dir,
        )

        mesh_path = self._extract_mesh(stage2_dir, output_dir)
        return mesh_path

    def _prepare_data(
        self,
        image_dir: Path,
        sparse_result: dict[str, Any],
        data_dir: Path,
    ) -> None:
        """Convert SfM outputs to NeuRodin/SDFStudio data format.

        Creates a transforms.json + images/ directory structure.
        """
        import shutil

        data_dir.mkdir(parents=True, exist_ok=True)
        images_out = data_dir / "images"
        images_out.mkdir(exist_ok=True)

        image_dir = Path(image_dir)
        cameras = sparse_result["cameras"]
        image_paths = sparse_result.get("image_paths", [])

        frames = []
        for path in image_paths:
            name = path.name
            if name not in cameras:
                continue

            shutil.copy2(path, images_out / name)

            cam = cameras[name]
            extrinsic = np.array(cam["extrinsic"])

            c2w = np.linalg.inv(extrinsic) if extrinsic.shape == (4, 4) else np.eye(4)

            intrinsic = np.array(cam["intrinsic"])
            fx = float(intrinsic[0, 0]) if intrinsic.ndim == 2 else 1000.0
            fy = float(intrinsic[1, 1]) if intrinsic.ndim == 2 else fx

            frames.append({
                "file_path": f"images/{name}",
                "transform_matrix": c2w.tolist(),
                "fl_x": fx,
                "fl_y": fy,
            })

        camera_positions = []
        for frame in frames:
            T = np.array(frame["transform_matrix"])
            camera_positions.append(T[:3, 3])
        camera_positions = np.array(camera_positions)
        center = camera_positions.mean(axis=0).tolist()
        radius = float(np.linalg.norm(camera_positions - center, axis=1).max() * 1.2)

        transforms_data = {
            "frames": frames,
            "sphere_center": center,
            "sphere_radius": radius,
        }

        with open(data_dir / "transforms.json", "w") as f:
            json.dump(transforms_data, f, indent=2)

        logger.info(
            "Prepared NeuRodin data: {} frames, center={}, radius={:.2f}",
            len(frames), [f"{c:.2f}" for c in center], radius,
        )

    def _train_stage(
        self,
        data_dir: Path,
        output_dir: Path,
        stage: int,
        config_name: str,
        load_dir: Path | None = None,
    ) -> Path:
        """Run one stage of NeuRodin training via ns-train."""
        experiment_name = f"neurodin-stage{stage}"
        scale_factor = self.cfg.get("scale_factor", 0.8)

        method_name = f"neurodin-stage{stage}-{config_name}"

        cmd = [
            "ns-train", method_name,
            "--experiment_name", experiment_name,
            "--output-dir", str(output_dir / "neurodin_output"),
        ]

        if load_dir is not None:
            checkpoints_dir = self._find_checkpoints(load_dir)
            if checkpoints_dir:
                cmd.extend(["--trainer.load_dir", str(checkpoints_dir)])

        cmd.extend([
            "sdfstudio-data",
            "--data", str(data_dir),
            "--scale_factor", str(scale_factor),
        ])

        logger.info("NeuRodin stage {} training (config={})", stage, config_name)
        logger.debug("Command: {}", " ".join(cmd))

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("ns-train output (last 1000 chars): {}", result.stdout[-1000:])

        stage_output = output_dir / "neurodin_output" / experiment_name
        logger.info("NeuRodin stage {} complete: {}", stage, stage_output)
        return stage_output

    def _extract_mesh(self, stage2_dir: Path, output_dir: Path) -> Path:
        """Extract mesh from trained NeuRodin model."""
        resolution = self.cfg.get("resolution", 2048)
        mesh_path = output_dir / "neurodin_mesh.ply"

        extract_script = _SDFSTUDIO_DIR / "zoo" / "extract_surface.py"
        if not extract_script.exists():
            extract_script = _SDFSTUDIO_DIR / "scripts" / "extract_mesh.py"

        config_path = self._find_config(stage2_dir)
        if config_path is None:
            logger.error("Could not find NeuRodin config in {}", stage2_dir)
            raise FileNotFoundError(f"No config found in {stage2_dir}")

        cmd = [
            sys.executable, str(extract_script),
            "--conf", str(config_path),
            "--resolution", str(resolution),
            "--output", str(mesh_path),
        ]

        logger.info("Extracting mesh at resolution {}", resolution)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("Mesh extraction output: {}", result.stdout[-1000:])
        logger.info("NeuRodin mesh extracted to {}", mesh_path)
        return mesh_path

    def _find_checkpoints(self, stage_dir: Path) -> Path | None:
        """Find the checkpoints directory from a training stage."""
        for candidate in stage_dir.rglob("nerfstudio_models"):
            return candidate
        for candidate in stage_dir.rglob("*.ckpt"):
            return candidate.parent
        return None

    def _find_config(self, stage_dir: Path) -> Path | None:
        """Find the training config YAML from a stage output."""
        for candidate in stage_dir.rglob("config.yml"):
            return candidate
        for candidate in stage_dir.rglob("*.yml"):
            return candidate
        return None
