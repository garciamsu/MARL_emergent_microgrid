import os
import pandas as pd
import numpy as np
from utils.discretization import digitize_clip

class MultiAgentEnv:
    """
    Environment that loads dataset, discretizes signals and advances step by step.
    """

    def __init__(self, config):
        csv_filename = config["simulation"]["dataset"]
        num_power_bins = config["discretization"]["bins_power"]

        # Load dataset
        self.dataset = self._load_data(csv_filename)

        self.max_steps = len(self.dataset)
        self.max_value = self.dataset.drop(columns="price").apply(
            pd.to_numeric, errors="coerce"
        ).max().max()

        self.demand_bins = np.linspace(0, self.max_value, num_power_bins)
        self.renewable_bins = np.linspace(0, self.max_value, num_power_bins)

        self.num_power_bins = num_power_bins
        self.reset()

    def reset(self):
        self.renewable_potential = 0
        self.renewable_potential_idx = digitize_clip(
            self.renewable_potential, self.renewable_bins
        )
        self.renewable_power = 0
        self.renewable_power_idx = digitize_clip(
            self.renewable_power, self.renewable_bins
        )
        self.demand_power = 0
        self.demand_power_idx = digitize_clip(self.demand_power, self.renewable_bins)
        self.total_power = 0
        self.total_power_idx = digitize_clip(self.total_power, self.renewable_bins)
        self.price = 0
        self.delta_power = 0
        self.delta_power_idx = "surplus"
        self.scale_demand = 1
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

    def get_discretized_state(self, index):
        row = self.dataset.iloc[index]
        self.demand_power = row["demand"] * self.scale_demand
        self.renewable_potential = row["solar_power"] + row["wind_power"]
        self.price = row["price"]
        self.time = row["Datetime"]

        self.demand_power_idx = digitize_clip(self.demand_power, self.demand_bins)
        self.renewable_potential_idx = digitize_clip(
            self.renewable_potential, self.renewable_bins
        )

        return (self.demand_power_idx, self.renewable_potential_idx)
