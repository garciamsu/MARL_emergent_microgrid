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
        self.num_price_bins = config["discretization"].get("price_bins", 5)
        
        # Power scaling factor from configuration (kW/kWh to W/Wh)
        self.power_scale_factor = config["simulation"].get("power_scale_factor", 1000.0)

        # Simulation time step in hours (used for SOC integration)
        self.dt_h = config.get("simulation", {}).get("dt_h", 1.0)

        # Load dataset and derive meta info
        self.dataset = self._load_data(csv_filename)
        self.max_steps = len(self.dataset)

        # Excluir columnas no deseadas
        excluded_columns = ["price", "demand", "Datetime"]

        # Calcular la suma fila por fila, descartando las columnas excluidas
        row_sums = (
            self.dataset.drop(columns=excluded_columns, errors='ignore')
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1)
        )

        # Obtener el valor máximo de las sumas de filas
        self.max_value = row_sums.max()
        print(f"Máximo valor calculado en dataset: {self.max_value}")

        self.power_bins = np.linspace(0, self.max_value, self.num_power_bins)
        
        # Create price bins if price column exists
        if "price" in self.dataset.columns:
            price_min = self.dataset["price"].min()
            price_max = self.dataset["price"].max()
            self.price_bins = np.linspace(price_min, price_max, self.num_price_bins)
        else:
            self.price_bins = np.linspace(0, 1, self.num_price_bins)  # Default bins
        
        # Store full dataset for random window selection
        self.full_dataset = self.dataset.copy()
        
        # Initialize with full dataset (will be replaced per episode)
        self.episode_data = None
        self.current_step = 0
        self.reset()

    def reset(self, episode_data=None, initial_soc=None) -> None:
        """Reset continuous and discretized attributes for a new episode.
        
        Args:
            episode_data (pd.DataFrame, optional): Contiguous window from dataset.
                Window size is configured via simulation.episode_window_hours.
                If None, uses full dataset (for backward compatibility).
            initial_soc (float, optional): Initial SOC for battery agent(s).
                If None, battery agents maintain their configured initial SOC.
        """
        # Update episode data if provided
        if episode_data is not None:
            self.episode_data = episode_data
            self.max_steps = len(episode_data)
        else:
            # Fallback to full dataset for backward compatibility
            self.episode_data = self.full_dataset
            self.max_steps = len(self.full_dataset)
        
        # Reset step counter
        self.current_step = 0
        
        # Store initial SOC for battery agents to use
        # This will be picked up by simulation.py when setting battery SOC
        if initial_soc is not None:
            self.initial_soc = initial_soc
        
        # Reset continuous variables
        self.renewable_potential = 0
        self.renewable_power = 0
        self.demand_power = 0
        self.total_power = 0
        self.price = 0
        self.energy_balance = 0
        self.soc_idx = 0

        # Reset discretized indices
        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.power_bins)
        self.renewable_power_idx = digitize_clip(self.renewable_power, self.power_bins)
        self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
        self.total_power_idx = digitize_clip(self.total_power, self.power_bins)
        self.energy_balance_idx = digitize_clip(self.energy_balance, self.power_bins)
        self.delta_power_idx = "surplus"

        self.state = None

    def _load_data(self, filename: str, offsets: dict | None = None) -> pd.DataFrame:
        """Load dataset CSV and apply optional per-column offsets.
        
        Power values are scaled from kW/kWh to Watts (W/Wh) using power_scale_factor.
        Scaling is applied to columns containing 'power' or matching 'demand'.
        """
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep="[;,]", engine="python")

        # Scale power values from kW/kWh to Watts using configured factor
        for col in df.columns:
            # Scale all power columns except price and datetime
            if col not in ["price", "Datetime", "datetime"]:
                if "power" in col.lower() or col.lower() == "demand":
                    df[col] = df[col] * self.power_scale_factor

        if offsets:
            for col, offset_value in offsets.items():
                if col in df.columns:
                    df[col] += offset_value

        if "demand" in df.columns:
            df["demand"] = df["demand"].clip(lower=0)
        return df

    def get_dataset(self, field: str, index: int) -> int:
        """Return discretized value for ``field`` at ``index`` updating env if needed.
        
        Uses episode_data if available (from random contiguous window), otherwise falls back to full dataset.
        """
        # Use episode_data if available, otherwise fall back to full dataset
        data_source = self.episode_data if self.episode_data is not None else self.dataset
        row = data_source.iloc[index]
        
        if field == "demand":
            self.demand_power = row[field]
            self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
            self.price = row["price"]
        
        # Use appropriate bins based on field type
        if field == "price":
            return digitize_clip(row[field], self.price_bins)
        else:
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