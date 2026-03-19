"""SuperPoint + LightGlue feature extraction and matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from photogrammetry.features.base import FeatureBackend


class SuperPointLightGlue(FeatureBackend):
    """Extract SuperPoint keypoints and match with LightGlue."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._extractor = None
        self._matcher = None

    def _lazy_init(self) -> None:
        if self._extractor is not None:
            return

        from lightglue import LightGlue, SuperPoint

        sp_cfg = self.cfg.get("superpoint", {})
        lg_cfg = self.cfg.get("lightglue", {})

        self._extractor = SuperPoint(
            max_num_keypoints=sp_cfg.get("max_num_keypoints", 4096),
            detection_threshold=sp_cfg.get("detection_threshold", 0.005),
        ).eval().to(self._device)

        self._matcher = LightGlue(
            features="superpoint",
            depth_confidence=lg_cfg.get("depth_confidence", 0.95),
            width_confidence=lg_cfg.get("width_confidence", 0.99),
            flash=lg_cfg.get("flash", True),
        ).eval().to(self._device)

        logger.info("SuperPoint + LightGlue initialized on {}", self._device)

    def extract_and_match(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        from lightglue.utils import load_image

        self._lazy_init()

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
        )
        logger.info("Extracting features from {} images", len(image_paths))

        features: dict[str, dict[str, Any]] = {}
        for path in image_paths:
            img = load_image(path).to(self._device)
            feats = self._extractor.extract(img)
            features[path.name] = {
                "keypoints": feats["keypoints"][0].cpu().numpy(),
                "descriptors": feats["descriptors"][0].cpu().numpy(),
                "scores": feats["scores"][0].cpu().numpy(),
                "image_size": feats["image_size"][0].cpu().numpy(),
                "raw": feats,
            }

        logger.info("Matching image pairs exhaustively")
        matches = []
        names = list(features.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                feats0 = features[names[i]]["raw"]
                feats1 = features[names[j]]["raw"]
                match_result = self._matcher({"image0": feats0, "image1": feats1})

                m = match_result["matches"][0].cpu().numpy()
                valid = m[:, 0] >= 0
                matches.append((names[i], names[j], m[valid]))

        keypoints = {
            name: data["keypoints"] for name, data in features.items()
        }

        logger.info(
            "Extracted features and matched {} pairs ({} total matches)",
            len(matches),
            sum(len(m[2]) for m in matches),
        )

        return {
            "keypoints": keypoints,
            "matches": matches,
            "output_dir": output_dir,
        }
