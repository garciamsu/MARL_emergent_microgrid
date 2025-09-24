import os
from analysis_tools.utils import clear_directories
from analysis_tools.metrics import check_stability
from analysis_tools.plotting import plot_metric, plot_coordination
from core.simulation import run_training
from core.environment import MultiAgentEnv
from configs.loader import load_config  # función simple que abre el YAML

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/evolution", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Clean old results
    clear_directories()

    # Load configuration
    config = load_config("configs/default.yaml")
    print(config)

    if config["mode"] == "train":
        run_training(config)
    elif config["mode"] == "offline":
        # TODO: implement offline analysis
        pass
