import os
import pandas as pd
import numpy as np
from utils.discretization import digitize_clip


class MultiAgentEnv:
    """Multi-agent microgrid environment.

    Responsibilities:
        - Load a dataset containing demand, price and resource potentials.
        - Create discretization bins for power values.
        - Act as a lightweight container whose fields are mutated by the
          simulation loop (no formal step API yet).
    """

    def __init__(self, config):
        """Instantiate the environment.

        Args:
            config (dict): Must contain ``simulation.dataset`` and
                ``discretization.bins_power``.
        """
        csv_filename = config["simulation"]["dataset"]
        self.num_power_bins = config["discretization"]["power_bins"]

        # Simulation time step in hours (used for SOC integration)
        self.dt_h = config.get("simulation", {}).get("dt_h", 1.0)

        # Load dataset and derive meta info
        self.dataset = self._load_data(csv_filename)
        self.max_steps = len(self.dataset)

        # Excluir columnas no deseadas
        excluded_columns = ["price", "demand", "Datetime"]

        # Calcular la suma fila por fila, descartando las columnas excluidas
        row_sums = (
            self.dataset.drop(columns=excluded_columns)
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1)
        )

        # Obtener el valor máximo de las sumas de filas
        self.max_value = row_sums.max()
        print(f"Máximo valor calculado en dataset: {self.max_value}")

        self.power_bins = np.linspace(0, self.max_value, self.num_power_bins)
        self.reset()

    def reset(self) -> None:
        """Reset continuous and discretized attributes for a new episode."""
        self.renewable_potential = 0
        self.renewable_power = 0
        self.demand_power = 0
        self.total_power = 0
        self.price = 0
        self.energy_balance = 0
        self.soc_idx = 0

        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.power_bins)
        self.renewable_power_idx = digitize_clip(self.renewable_power, self.power_bins)
        self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
        self.total_power_idx = digitize_clip(self.total_power, self.power_bins)
        self.delta_power_idx = "surplus"

        self.scale_demand = 1
        self.state = None

    def _load_data(self, filename: str, offsets: dict | None = None) -> pd.DataFrame:
        """Load dataset CSV and apply optional per-column offsets."""
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep="[;,]", engine="python")

        if offsets:
            for col, offset_value in offsets.items():
                if col in df.columns:
                    df[col] += offset_value

        if "demand" in df.columns:
            df["demand"] = df["demand"].clip(lower=0)
        return df

    def get_dataset(self, field: str, index: int) -> int:
        """Return discretized value for ``field`` at ``index`` updating env if needed."""
        row = self.dataset.iloc[index]
        if field == "demand":
            self.demand_power = row[field] * self.scale_demand
            self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
            self.price = row["price"]

        return digitize_clip(row[field], self.power_bins)

    def get_value(self, var: str) -> int:
        """Return discretized index for a known global variable name."""
        if var == "potential":
            self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.power_bins)
            return self.renewable_potential_idx
        if var == "renewable":
            self.renewable_power_idx = digitize_clip(self.renewable_power, self.power_bins)
            return self.renewable_power_idx
        if var == "demand":
            self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
            return self.demand_power_idx
        if var == "total":
            self.total_power_idx = digitize_clip(self.total_power, self.power_bins)
            return self.total_power_idx
        return 0