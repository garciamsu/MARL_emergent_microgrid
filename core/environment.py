import os
import pandas as pd
import numpy as np
from utils.discretization import digitize_clip

class MultiAgentEnv:
    """
    Environment that loads dataset, discretizes signals and advances step by step.
    """

    def __init__(self, config):
        """
        Initialize the environment with dataset and discretization parameters.
        """
        # --- Configuration parameters ---
        csv_filename = config["simulation"]["dataset"]
        self.num_power_bins = config["discretization"]["bins_power"]

        # --- Load dataset ---
        self.dataset = self._load_data(csv_filename)
        self.max_steps = len(self.dataset)

        # --- Compute maximum value for discretization ---
        self.max_value = (
            self.dataset.drop(columns="price")
            .apply(pd.to_numeric, errors="coerce")
            .max()
            .max()
        )

        # --- Define discretization bins ---
        self.power_bins = np.linspace(0, self.max_value, self.num_power_bins)

        # --- Initialize state ---
        self.reset()

    def reset(self) -> None:
        """
        Reset environment attributes to their initial values and discretized states.
        """
        # Continuous variables
        self.renewable_potential = 0
        self.renewable_power = 0
        self.demand_power = 0
        self.total_power = 0
        self.price = 0
        self.delta_power = 0

        # Discretized states
        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.power_bins)
        self.renewable_power_idx = digitize_clip(self.renewable_power, self.power_bins)
        self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
        self.total_power_idx = digitize_clip(self.total_power, self.power_bins)
        self.delta_power_idx = "surplus"

        # Auxiliaries
        self.scale_demand = 1

        # Global state
        self.state = None

    def _load_data(self, filename: str, offsets: dict = None) -> pd.DataFrame:
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep="[;,]", engine="python")

        if offsets is not None:
            for col, offset_value in offsets.items():
                if col in df.columns:
                    df[col] += offset_value

        if "demand" in df.columns:
            df["demand"] = df["demand"].clip(lower=0)

        return df

    def get_value(self, field: str, index: int) -> None:
        """
        Update environment attributes with values and discretized states 
        from the dataset row at the given index.
        """
        
        # Extract values from dataset row
        row = self.dataset.iloc[index]
        
        # Compute discretized states
        return digitize_clip(row[field], self.power_bins)