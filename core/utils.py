"""Core utility helpers (seeding, logging factory) for the MARL microgrid."""
from __future__ import annotations
import os
import random
import logging
from typing import Optional, Dict, Any, Tuple

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


def get_offline_run_id(config: Dict[str, Any]) -> Optional[str]:
    """Return the offline_run identifier from config if present.

    This helper centralizes access to the run identifier used to
    tag offline evaluation artefacts (e.g. evolution CSVs, plots,
    and checkpoints directories).
    """

    value = config.get("offline_run") if isinstance(config, dict) else None
    if value is None:
        return None
    return str(value)


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""

    os.makedirs(path, exist_ok=True)


def get_checkpoints_dir(base_results_dir: str, run_id: str) -> str:
    """Build checkpoints directory path for a given run_id.

    Args:
        base_results_dir: Base results directory (e.g. 'results').
        run_id: Identifier for the training/offline run.

    Returns:
        str: Full path to checkpoints directory for this run.
    """

    return os.path.join(base_results_dir, "checkpoints", run_id)


def save_q_tables(agents: Dict[str, Any], base_results_dir: str, run_id: str) -> str:
    """Persist all agents' Q-tables under a checkpoints directory.

    Each agent's Q-table is stored as a NumPy .npz file using a
    dictionary-of-dicts representation converted to flat arrays.
    This is intentionally simple and specific to this project.
    """

    ckpt_dir = get_checkpoints_dir(base_results_dir, run_id)
    ensure_dir(ckpt_dir)

    for name, agent in agents.items():
        q_table = getattr(agent, "q_table", None)
        if not q_table:
            continue

        states = []
        actions = []
        values = []
        for state, a_dict in q_table.items():
            for action, val in a_dict.items():
                states.append(state)
                actions.append(action)
                values.append(val)

        if not states:
            continue

        # Encode states as tuples of ints/floats; saved as object array
        np_states = np.array(states, dtype=object)
        np_actions = np.array(actions, dtype=int)
        np_values = np.array(values, dtype=float)

        out_path = os.path.join(ckpt_dir, f"qtable_{name}.npz")
        np.savez_compressed(out_path, states=np_states, actions=np_actions, values=np_values)

    return ckpt_dir


def load_q_tables(agents: Dict[str, Any], base_results_dir: str, run_id: str) -> Tuple[bool, str]:
    """Load Q-tables for all agents from a checkpoints directory if present.

    Returns a tuple (loaded_any, ckpt_dir).
    """

    ckpt_dir = get_checkpoints_dir(base_results_dir, run_id)
    if not os.path.isdir(ckpt_dir):
        return False, ckpt_dir

    loaded_any = False
    for name, agent in agents.items():
        path = os.path.join(ckpt_dir, f"qtable_{name}.npz")
        if not os.path.isfile(path):
            continue

        data = np.load(path, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]
        values = data["values"]

        q_table = {}
        for s, a, v in zip(states, actions, values):
            state = tuple(s) if not isinstance(s, tuple) else s
            if state not in q_table:
                # Pre-fill all actions to 0.0 to keep shape consistent
                q_table[state] = {act: 0.0 for act in getattr(agent, "actions", [])}
            q_table[state][int(a)] = float(v)

        agent.q_table = q_table
        loaded_any = True

    return loaded_any, ckpt_dir
