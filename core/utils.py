"""Core utility helpers (seeding, logging factory) for the MARL microgrid."""
from __future__ import annotations
import os
import random
import logging
from datetime import datetime
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (optionally) other RNG sources.

    Args:
        seed (int): Deterministic seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch optional
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch not installed
        pass


def build_logger(name: str = "marl", level: int = logging.INFO, log_dir: str = "results/logs") -> logging.Logger:
    """Create a configured logger writing to stdout only.

    The previous behaviour wrote timestamped files like ``run_YYYYMMDD_HHMMSS.log``
    under ``results/logs``. Those per-run files are no longer needed and have
    been removed to avoid cluttering the repository. Aggregated analytics
    (``episode_metadata.xlsx``, ``episode_rewards.csv``, etc.) are still
    handled elsewhere and remain unchanged.

    Args:
        name (str): Logger name.
        level (int): Logging level.
        log_dir (str): Kept for backward compatibility, currently unused.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Reuse existing
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Stream handler only (stdout/stderr via logging)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)
    logger.addHandler(sh)

    logger.propagate = False
    return logger
