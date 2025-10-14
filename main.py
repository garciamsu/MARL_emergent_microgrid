"""Entry point for MARL emergent microgrid experiments."""

import agents  # noqa: F401  (populate registry via side effects)
import core.policies  # noqa: F401
import core.rewards  # noqa: F401

from analysis_tools.utils import clear_directories
from core.simulation import run_training
from configs.loader import load_config


def main(config_path: str = "configs/default.yaml") -> None:
    """Orchestrate a run based on configuration file.

    Args:
        config_path (str): Path to YAML config.
    """
    clear_directories()
    config = load_config(config_path)

    mode = config.get("mode", "train")
    if mode == "train":
        run_training(config)
    elif mode == "offline":  # pragma: no cover - placeholder
        # TODO: implement offline analysis (e.g., evaluation only / plotting)
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":  # pragma: no cover
    main()
