"""Entrypoint for running MARL emergent microgrid simulations."""

import agents
import core.policies  # ensure policy registry is populated
import core.rewards   # ensure reward registry is populated
from analysis_tools.utils import clear_directories
from core.simulation import run_training
from configs.loader import load_config  # función simple que abre el YAML

if __name__ == "__main__":
    # Clean old results
    clear_directories()

    # Load configuration
    config = load_config("configs/default.yaml")

    if config["mode"] == "train":
        run_training(config)
    elif config["mode"] == "offline":
        # TODO: implement offline analysis
        pass
