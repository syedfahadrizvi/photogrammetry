"""Pipeline orchestrator that dispatches to the correct backend chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig


@dataclass
class PipelineResult:
    """Holds output paths produced by a pipeline run."""

    mesh_path: Path | None = None
    point_cloud_path: Path | None = None
    texture_path: Path | None = None
    colmap_dir: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PipelineRunner:
    """Runs the photogrammetry pipeline according to the loaded config."""

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        self.input_dir = Path(config.get("input_dir", "."))
        self.output_dir = Path(config.get("output_dir", "./output"))

    def run(self) -> PipelineResult:
        preset = self.cfg.pipeline.preset
        logger.info("Starting pipeline with preset '{}'", preset)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if preset == "classical":
            return self._run_classical()
        elif preset in ("neural", "hybrid"):
            return self._run_modular()
        elif preset == "quality":
            return self._run_quality()
        else:
            raise ValueError(f"Unknown preset: {preset}")

    # ------------------------------------------------------------------
    # Classical: AliceVision end-to-end
    # ------------------------------------------------------------------

    def _run_classical(self) -> PipelineResult:
        from photogrammetry.surface.alicevision import AliceVisionBackend

        logger.info("Classical mode: delegating to AliceVision/Meshroom")
        backend = AliceVisionBackend(self.cfg.surface.alicevision)
        mesh_path = backend.reconstruct(self.input_dir, self.output_dir)

        return PipelineResult(mesh_path=mesh_path)

    # ------------------------------------------------------------------
    # Modular: Features → SfM → Dense → MILo
    # ------------------------------------------------------------------

    def _run_modular(self) -> PipelineResult:
        images = self._preprocess()
        sparse = self._run_sfm(images)
        dense_result = self._run_dense(images, sparse)
        mesh_path = self._run_surface_milo(dense_result)

        return PipelineResult(
            mesh_path=mesh_path,
            point_cloud_path=dense_result.get("point_cloud_path"),
            colmap_dir=dense_result.get("colmap_dir"),
        )

    # ------------------------------------------------------------------
    # Quality: Features → SfM → NeuRodin
    # ------------------------------------------------------------------

    def _run_quality(self) -> PipelineResult:
        images = self._preprocess()
        sparse = self._run_sfm(images)
        mesh_path = self._run_surface_neurodin(images, sparse)

        return PipelineResult(mesh_path=mesh_path)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _preprocess(self) -> dict[str, Any]:
        from photogrammetry.preprocessing.images import preprocess_images

        logger.info("Stage 1: Preprocessing images")
        return preprocess_images(
            self.input_dir,
            self.output_dir / "preprocessed",
            self.cfg.preprocessing,
        )

    def _run_sfm(self, images: dict[str, Any]) -> dict[str, Any]:
        preset = self.cfg.pipeline.preset
        result: dict[str, Any] = {}

        if self.cfg.sfm.backend == "vggt":
            from photogrammetry.sfm.vggt import VGGTSfM

            logger.info("Stage 3: VGGT feed-forward SfM")
            sfm = VGGTSfM(self.cfg.sfm.vggt)
            result = sfm.estimate(
                images["image_dir"], self.output_dir / "sfm"
            )

        if preset in ("hybrid", "quality") and self.cfg.sfm.colmap.bundle_adjustment:
            from photogrammetry.sfm.colmap import COLMAPBundleAdjustment

            logger.info("Stage 3b: COLMAP bundle adjustment refinement")
            ba = COLMAPBundleAdjustment(self.cfg.sfm.colmap)
            result = ba.refine(result, self.output_dir / "sfm_refined")

        if preset in ("hybrid", "quality"):
            from photogrammetry.features.superpoint_lightglue import (
                SuperPointLightGlue,
            )

            logger.info("Stage 2: SuperPoint + LightGlue feature matching")
            matcher = SuperPointLightGlue(self.cfg.features)
            match_result = matcher.extract_and_match(
                images["image_dir"], self.output_dir / "features"
            )
            result["matches"] = match_result

        return result

    def _run_dense(
        self, images: dict[str, Any], sparse: dict[str, Any]
    ) -> dict[str, Any]:
        from photogrammetry.dense.vggt_dense import VGGTDense

        logger.info("Stage 4: VGGT dense reconstruction")
        dense = VGGTDense(self.cfg.dense.vggt)
        return dense.reconstruct(
            images["image_dir"],
            sparse,
            self.output_dir / "dense",
        )

    def _run_surface_milo(self, dense_result: dict[str, Any]) -> Path:
        from photogrammetry.surface.milo import MILoBackend

        logger.info("Stage 5: MILo surface reconstruction")
        backend = MILoBackend(self.cfg.surface.milo)
        return backend.reconstruct(
            dense_result["colmap_dir"],
            self.output_dir / "surface",
        )

    def _run_surface_neurodin(
        self, images: dict[str, Any], sparse: dict[str, Any]
    ) -> Path:
        from photogrammetry.surface.neurodin import NeuRodinBackend

        logger.info("Stage 5: NeuRodin surface reconstruction")
        backend = NeuRodinBackend(self.cfg.surface.neurodin)
        return backend.reconstruct(
            images["image_dir"],
            sparse,
            self.output_dir / "surface",
        )
