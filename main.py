"""Entry point for MARL emergent microgrid experiments."""

import agents  # noqa: F401  (populate registry via side effects)
import core.policies  # noqa: F401
import core.rewards  # noqa: F401

from analysis.common.utils import clear_directories
from core.simulation import run_training
from configs.loader import load_config


def main(config_path: str = "configs/default.yaml") -> None:
    """Orchestrate a run based on configuration file.

    Args:
        config_path (str): Path to YAML config.
    """
    config = load_config(config_path)

    mode = config.get("mode", "train")
    if mode == "train":
        # Training mode: full tabular Q-learning with exploration and updates
        clear_directories()
        run_training(config)
    elif mode == "offline":  # pragma: no cover - offline evaluation / exploitation
        # Offline mode: exploit previously learned Q-tables on a separate dataset.
        offline_cfg = config.get("offline", {}) or {}
        offline_run = config.get("offline_run", None)

        if not offline_run:
            raise ValueError(
                "offline_run must be set in configs/default.yaml when mode=offline "
            )

        # Reuse the training loop but with overridden parameters:
        # - episodes: offline.episodes (typically 1)
        # - dataset: offline.dataset
        # - epsilon: constant, offline.epsilon
        # The agents should be initialized with Q-tables loaded from
        # results/checkpoints/<offline_run>/ by their own logic.
        sim_cfg = config.get("simulation", {})
        sim_cfg = dict(sim_cfg)  # shallow copy to avoid mutating original

        # Override dataset and episodes for offline evaluation
        if "dataset" in offline_cfg:
            sim_cfg["dataset"] = offline_cfg["dataset"]
        if "episodes" in offline_cfg:
            sim_cfg["episodes"] = offline_cfg["episodes"]

        # Build a constant epsilon schedule for offline exploitation
        eps_value = float(offline_cfg.get("epsilon", 0.01))
        sim_cfg["epsilon"] = {
            "schedule": "constant",
            "start": eps_value,
            "end": eps_value,
            "min": eps_value,
        }

        # Attach back the patched simulation config and offline_run hint,
        # then call run_training which will behave deterministically given
        # the fixed windowing logic and constant epsilon.
        offline_config = dict(config)
        offline_config["simulation"] = sim_cfg
        offline_config["offline_run"] = offline_run

        # NOTE: we do NOT clear directories here to preserve training results;
        # offline runs should accumulate under their own run_id.
        run_training(offline_config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":  # pragma: no cover
    main()