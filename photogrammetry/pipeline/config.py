"""Configuration loading and validation via OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

_PRESET_FILES = {
    "classical": "classical.yaml",
    "neural": "neural.yaml",
    "hybrid": "hybrid.yaml",
    "quality": "quality.yaml",
}

_VALID_SURFACE_BACKENDS = {"alicevision", "milo", "neurodin"}
_VALID_SFM_BACKENDS = {"vggt", "colmap"}
_VALID_DENSE_BACKENDS = {"vggt"}
_VALID_FEATURE_BACKENDS = {"superpoint_lightglue"}


def load_config(
    config_path: str | Path | None = None,
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load pipeline configuration.

    Resolution order (later overrides earlier):
        1. configs/default.yaml (always loaded)
        2. Preset file (if ``preset`` is given)
        3. Custom config file (if ``config_path`` is given)
        4. Programmatic ``overrides`` dict
    """
    base = OmegaConf.load(_CONFIGS_DIR / "default.yaml")

    if preset is not None:
        if preset not in _PRESET_FILES:
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: {list(_PRESET_FILES)}"
            )
        preset_cfg = OmegaConf.load(_CONFIGS_DIR / _PRESET_FILES[preset])
        base = OmegaConf.merge(base, preset_cfg)

    if config_path is not None:
        user_cfg = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, user_cfg)

    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))

    _validate(base)
    return base


def _validate(cfg: DictConfig) -> None:
    """Raise on invalid configuration values."""
    surface = cfg.surface.backend
    if surface not in _VALID_SURFACE_BACKENDS:
        raise ValueError(
            f"Invalid surface backend '{surface}'. Choose from: {_VALID_SURFACE_BACKENDS}"
        )

    preset = cfg.pipeline.preset
    if preset == "classical" and surface != "alicevision":
        raise ValueError("Classical preset requires surface.backend = 'alicevision'")

    sfm = cfg.sfm.backend
    if sfm not in _VALID_SFM_BACKENDS:
        raise ValueError(
            f"Invalid sfm backend '{sfm}'. Choose from: {_VALID_SFM_BACKENDS}"
        )

    dense = cfg.dense.backend
    if dense not in _VALID_DENSE_BACKENDS:
        raise ValueError(
            f"Invalid dense backend '{dense}'. Choose from: {_VALID_DENSE_BACKENDS}"
        )

    features = cfg.features.backend
    if features not in _VALID_FEATURE_BACKENDS:
        raise ValueError(
            f"Invalid features backend '{features}'. Choose from: {_VALID_FEATURE_BACKENDS}"
        )
