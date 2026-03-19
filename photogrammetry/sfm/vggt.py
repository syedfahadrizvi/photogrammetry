"""VGGT feed-forward Structure from Motion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from PIL import Image

from photogrammetry.sfm.base import SfMBackend


class VGGTSfM(SfMBackend):
    """Feed-forward camera pose and 3D point estimation via VGGT.

    Uses the facebook/VGGT-1B model from Hugging Face to predict camera
    extrinsics, intrinsics, depth maps, and 3D point maps in a single
    forward pass.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download

        model_name = self.cfg.get("model", "facebook/VGGT-1B")
        dtype_str = self.cfg.get("dtype", "float16")
        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        logger.info("Loading VGGT model: {}", model_name)

        try:
            from vggt.models.vggt import VGGT

            self._model = VGGT.from_pretrained(model_name)
        except ImportError:
            logger.warning(
                "VGGT package not found. Install from: "
                "https://github.com/facebookresearch/vggt"
            )
            raise

        self._model = self._model.to(self._device).eval()
        if dtype == torch.float16:
            self._model = self._model.half()

        logger.info("VGGT loaded on {} (dtype={})", self._device, dtype_str)

    def estimate(
        self,
        image_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        self._lazy_init()

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        logger.info("Running VGGT on {} images", len(image_paths))

        images_tensor = self._load_images(image_paths)

        with torch.no_grad():
            predictions = self._model(images_tensor)

        cameras = self._extract_cameras(predictions, image_paths)
        points_3d = self._extract_points(predictions)

        cameras_path = output_dir / "cameras.json"
        with open(cameras_path, "w") as f:
            json.dump(
                {k: {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                      for kk, vv in v.items()}
                 for k, v in cameras.items()},
                f, indent=2,
            )

        points_path = output_dir / "points3d.npy"
        np.save(points_path, points_3d)

        logger.info(
            "VGGT estimated {} cameras, {} 3D points",
            len(cameras), len(points_3d),
        )

        return {
            "cameras": cameras,
            "points_3d": points_3d,
            "predictions": predictions,
            "image_paths": image_paths,
            "output_dir": output_dir,
        }

    def _load_images(self, image_paths: list[Path]) -> torch.Tensor:
        """Load and preprocess images into a batched tensor."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(transform(img))

        return torch.stack(tensors).unsqueeze(0).to(self._device)

    def _extract_cameras(
        self, predictions: dict, image_paths: list[Path]
    ) -> dict[str, dict[str, Any]]:
        """Extract per-image camera parameters from VGGT predictions."""
        cameras: dict[str, dict[str, Any]] = {}

        extrinsics = predictions.get("extrinsic", None)
        intrinsics = predictions.get("intrinsic", None)

        if extrinsics is None or intrinsics is None:
            logger.warning("VGGT predictions missing camera fields")
            return cameras

        extrinsics = extrinsics[0].cpu().numpy()
        intrinsics = intrinsics[0].cpu().numpy()

        conf_threshold = self.cfg.get("confidence_threshold", 0.5)

        for i, path in enumerate(image_paths):
            cameras[path.name] = {
                "extrinsic": extrinsics[i],
                "intrinsic": intrinsics[i],
            }

        return cameras

    def _extract_points(self, predictions: dict) -> np.ndarray:
        """Extract 3D point cloud from VGGT world point predictions."""
        world_points = predictions.get("world_points_out", None)
        if world_points is None:
            logger.warning("VGGT predictions missing world_points_out")
            return np.empty((0, 3))

        pts = world_points[0].cpu().numpy()
        pts = pts.reshape(-1, 3)

        conf = predictions.get("world_points_conf", None)
        if conf is not None:
            conf = conf[0].cpu().numpy().reshape(-1)
            threshold = self.cfg.get("confidence_threshold", 0.5)
            mask = conf > threshold
            pts = pts[mask]
            logger.debug(
                "Filtered points: {} -> {} (threshold={:.2f})",
                len(mask), mask.sum(), threshold,
            )

        return pts
