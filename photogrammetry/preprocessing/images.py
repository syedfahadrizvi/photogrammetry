"""Image loading, EXIF extraction, resizing, and optional masking."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from PIL.ExifTags import Base as ExifBase


def preprocess_images(
    input_dir: Path,
    output_dir: Path,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Load, resize, and optionally mask images.

    Returns a dict with ``image_dir`` and ``image_metadata`` keys.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported = set(cfg.get("supported_formats", ["jpg", "jpeg", "png", "tiff", "bmp"]))
    max_size = cfg.get("max_image_size", 0)
    mask_dir = cfg.get("mask_dir")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lstrip(".").lower() in supported
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    logger.info("Found {} images in {}", len(image_paths), input_dir)
    metadata: list[dict[str, Any]] = []

    for img_path in image_paths:
        exif = _extract_exif(img_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read {}, skipping", img_path.name)
            continue

        h, w = img.shape[:2]
        scale = 1.0
        if max_size > 0 and max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
            logger.debug("Resized {} to {}x{} (scale={:.2f})", img_path.name, w, h, scale)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)

        if mask_dir is not None:
            _apply_mask(img_path, mask_dir, output_dir, scale)

        metadata.append({
            "filename": img_path.name,
            "width": w,
            "height": h,
            "scale": scale,
            "exif": exif,
        })

    logger.info("Preprocessed {} images to {}", len(metadata), output_dir)
    return {
        "image_dir": output_dir,
        "image_metadata": metadata,
    }


def _extract_exif(image_path: Path) -> dict[str, Any]:
    """Extract useful EXIF data from an image."""
    try:
        with Image.open(image_path) as pil_img:
            raw = pil_img.getexif()
            if not raw:
                return {}
            exif: dict[str, Any] = {}
            tag_map = {
                ExifBase.FocalLength: "focal_length",
                ExifBase.Make: "camera_make",
                ExifBase.Model: "camera_model",
                ExifBase.ImageWidth: "image_width",
                ExifBase.ImageLength: "image_height",
            }
            for tag_id, key in tag_map.items():
                val = raw.get(tag_id)
                if val is not None:
                    exif[key] = float(val) if isinstance(val, (int, float)) else str(val)
            return exif
    except Exception:
        return {}


def _apply_mask(
    img_path: Path,
    mask_dir: str | Path,
    output_dir: Path,
    scale: float,
) -> None:
    """Copy and optionally resize mask images alongside the main images."""
    mask_dir = Path(mask_dir)
    stem = img_path.stem
    for ext in (".png", ".mask.png", ".jpg"):
        mask_path = mask_dir / f"{stem}{ext}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and scale != 1.0:
                h, w = mask.shape[:2]
                mask = cv2.resize(
                    mask,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_out = output_dir / f"{stem}.mask.png"
            cv2.imwrite(str(mask_out), mask)
            return
