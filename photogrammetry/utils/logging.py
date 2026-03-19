"""Logging configuration with loguru and rich."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure loguru with rich-formatted console output and optional file sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if log_file is not None:
        logger.add(
            str(log_file),
            level=level,
            rotation="50 MB",
            retention="7 days",
        )
    logger.info("Logging configured (level={})", level)
