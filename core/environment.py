import os
import logging
import pandas as pd
import numpy as np
from utils.discretization import digitize_clip
from core.csv_handler import read_dataset_csv

# Module-level logger
logger = logging.getLogger("marl")


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

        self.num_delta_bins = config["discretization"].get("delta_bins", 3)
        self.num_power_bins = config["discretization"].get("power_bins", 2)
        self.num_price_bins = config["discretization"].get("price_bins", 2)
        self.num_soc_bins = config["discretization"].get("soc_bins", 3)

        # Power scaling factor from configuration (kW/kWh to W/Wh)
        self.power_scale_factor = config["simulation"].get("power_scale_factor", 1000.0)

        # Enabled renewable sources (driven by YAML agent counts)
        agents_cfg = config.get("agents", {}) or {}
        self.solar_count = int(agents_cfg.get("solar", {}).get("count", 0) or 0)
        self.wind_count = int(agents_cfg.get("wind", {}).get("count", 0) or 0)
        self.solar_enabled = self.solar_count > 0
        self.wind_enabled = self.wind_count > 0

        # Simulation time step in hours (used for SOC integration)
        self.dt_h = config.get("simulation", {}).get("dt_h", 1.0)

        # Load dataset and derive meta info
        self.dataset = self._load_data(csv_filename)
        self.max_steps = len(self.dataset)

        # Precompute demand and price statistics for dynamic normalization in rewards
        if "demand" in self.dataset.columns:
            self.demand_max = float(self.dataset["demand"].max())
        else:
            self.demand_max = 0.0

        if "price" in self.dataset.columns:
            self.price_min = float(self.dataset["price"].min())
            self.price_max = float(self.dataset["price"].max())
        else:
            self.price_min = 0.0
            self.price_max = 0.0

        # Get maximum power value from config (design limit / physical constraint)
        self.max_value = float(config.get("simulation", {}).get("max_power", 300000.0))
        logger.info("Maximum power value from config: %.1f", self.max_value)

        # Get maximum price value from config (design limit / physical constraint)
        self.max_price = float(config.get("simulation", {}).get("max_price", 200.0))
        logger.info("Maximum price value from config: %.1f", self.max_price)

        self.price_bins = np.linspace(0, 1, self.num_price_bins)
        self.power_bins = np.linspace(0, 1, self.num_power_bins)
        self.delta_bins = np.linspace(-1, 1, self.num_delta_bins)

        # Create price bins using comfort_threshold from load agent config (if available)
        # If comfort_threshold is defined and load agent is active, create asymmetric bins
        # Otherwise, use default linear bins for price discretization
        comfort_threshold = config.get("agents", {}).get("load", {}).get("limits", {}).get("comfort_threshold")
        load_count = config.get("agents", {}).get("load", {}).get("count", 0)
        
        if comfort_threshold is not None and load_count > 0:
            # Asymmetric bins: [0, threshold, max_price] for binary "cheap/expensive"
            self.price_bins = np.array([0, float(comfort_threshold), self.max_price])
            logger.info("Price bins (asymmetric): [0, %.1f, %.1f] EUR/MWh", comfort_threshold, self.max_price)
        else:
            # Default linear bins when no load agent or no comfort_threshold
            self.price_bins = np.linspace(0, self.max_price, self.num_price_bins + 1)
            logger.info("Price bins (linear): %s EUR/MWh", self.price_bins)

        # Q-table initialization value (optimistic initialization)
        # Higher values encourage exploration of unvisited state-action pairs
        self.q_init_value = config.get("simulation", {}).get("q_init_value", 0.0)
        logger.info("Q-table initialization value: %.2f", self.q_init_value)

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
        self.base_demand = 0  # Base demand from dataset (before load agent modulation)
        self.total_power = 0
        self.price = 0
        self.price_norm = 0
        self.price_idx = 0
        self.energy_balance = 0
        self.soc_idx = 0
        self.soc = 0
        self.soc_state = 0
        self.grid_power = 0

        # Reset discretized indices
        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.power_bins)
        self.renewable_power_idx = digitize_clip(self.renewable_power, self.power_bins)
        self.demand_power_idx = digitize_clip(self.demand_power, self.power_bins)
        self.total_power_idx = digitize_clip(self.total_power, self.power_bins)
        self.energy_balance_idx = digitize_clip(self.energy_balance, self.power_bins)
        self.grid_power_idx = 0
        self.delta_power_idx = "surplus"
        self.delta_ph = 0
        self.delta_ph_norm = 0
        self.delta_ph_idx = 0
        # Real balance: actual power (renewable_power - demand) for battery/grid rewards
        self.real_balance = 0
        self.real_balance_norm = 0
        self.real_balance_idx = 0
        self.solar_potential = 0
        self.wind_potential = 0
        self.solar_potential_norm = 0
        self.wind_potential_norm = 0
        self.solar_potential_idx = 0
        self.wind_potential_idx = 0


        self.state = None

    def _load_data(self, filename: str, offsets: dict | None = None) -> pd.DataFrame:
        """Load dataset CSV and apply optional per-column offsets.
        
        Power values are scaled from kW/kWh to Watts (W/Wh) using power_scale_factor.
        Scaling is applied to columns containing 'power' or matching 'demand'.
        """
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = read_dataset_csv(file_path)

        # Scale power values from kW/kWh to Watts using configured factor
        for col in df.columns:
            # Scale all power columns except price and datetime
            if col not in ["price", "Datetime", "datetime"]:
                if "potential" in col.lower() or col.lower() == "demand":
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
            self.base_demand = row[field]  # Store original demand before load agent modulation
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

    def load_timestep_data(self, index: int) -> None:
        """Load base data from dataset for the current timestep.
        
        This method MUST be called at the START of each timestep, BEFORE any
        agent observes state or takes action. It initializes:
        - Base demand and price from dataset
        - Solar and wind potentials from dataset
        - Initial renewable_potential (sum of solar + wind)
        - Initial delta_ph calculation
        
        The renewable_potential will be dynamically reduced as each renewable
        agent consumes its portion (stigmergic paradigm).
        
        Args:
            index: Current timestep index within episode_data.
        """
        from utils.discretization import discretize_ternary
        
        data_source = self.episode_data if self.episode_data is not None else self.dataset
        row = data_source.iloc[index]
        
        # Load base values from dataset
        self.base_demand = row["demand"]
        self.demand_power = row["demand"]
        self.price = row["price"]
        self.price_norm = self.price / self.max_price if self.max_price > 0 else 0
        self.price_idx = digitize_clip(self.price, self.price_bins)
        
        # Load renewable potentials from dataset (respect agent enable flags)
        raw_solar_potential = row.get("solar_potential", 0.0)
        self.solar_potential = raw_solar_potential if self.solar_enabled else 0.0
        self.solar_potential_norm = self.solar_potential / self.max_value if self.max_value > 0 else 0
        self.solar_potential_idx = 1 if self.solar_potential_norm > 0.0 else 0
        
        raw_wind_potential = row.get("wind_potential", 0.0)
        self.wind_potential = raw_wind_potential if self.wind_enabled else 0.0
        self.wind_potential_norm = self.wind_potential / self.max_value if self.max_value > 0 else 0
        self.wind_potential_idx = 1 if self.wind_potential_norm > 0.0 else 0
        
        # Initialize renewable_potential as sum of all renewable sources
        # This is the STIGMERGIC variable that will be consumed by agents
        self.renewable_potential = self.solar_potential + self.wind_potential
        self.renewable_potential_idx = digitize_clip(
            self.renewable_potential / self.max_value if self.max_value > 0 else 0,
            self.power_bins
        )
        
        # Calculate initial delta_ph (stigmergic balance signal)
        self.update_delta_ph()
        
        logger.debug(
            "Timestep %d loaded: demand=%.1f, price=%.2f, solar_pot=%.1f, wind_pot=%.1f, delta_ph=%.1f",
            index, self.demand_power, self.price, self.solar_potential, self.wind_potential, self.delta_ph
        )

    def update_delta_ph(self, power_consumed: float = 0.0) -> None:
        """Recalculate delta_ph based on current renewable_potential and demand.
        
        delta_ph = renewable_potential - demand_power
        
        This is the core STIGMERGIC signal that agents observe. As renewable
        agents consume their potential, this value changes, affecting decisions
        of subsequent agents in the execution order.
        
        The delta_ph represents:
        - Positive: Potential surplus (more renewable potential than demand)
        - Zero: Balanced (potential matches demand)
        - Negative: Potential deficit (demand exceeds renewable potential)
        
        Additionally updates real_balance which represents the actual power balance
        (renewable_power - demand_power) for use by battery and grid agents.
        """
        from utils.discretization import discretize_ternary
        
        # Stigmergic signal: uses remaining POTENTIAL (for renewable coordination)
        self.delta_ph = (self.renewable_potential - self.demand_power) - power_consumed
        self.delta_ph_norm = self.delta_ph / self.max_value if self.max_value > 0 else 0
        self.delta_ph_idx = discretize_ternary(self.delta_ph_norm, threshold=0.01)
        
        # Real balance: uses actual POWER injected (for battery/grid rewards)
        # This is the actual surplus/deficit based on power already committed to the system
        self.real_balance = self.renewable_power - self.demand_power
        self.real_balance_norm = self.real_balance / self.max_value if self.max_value > 0 else 0
        self.real_balance_idx = discretize_ternary(self.real_balance_norm, threshold=0.01)

    def consume_renewable_potential(self, power_consumed: float, source: str = "unknown") -> None:
        """Reduce renewable_potential after an agent consumes its portion.
        
        This implements the stigmergic consumption model: when a renewable agent
        injects power, that portion is no longer available as "potential" for
        subsequent agents to consider in their decision-making.
        
        Args:
            power_consumed: Amount of power consumed/injected by the agent (W).
            source: Name of the agent consuming (for logging).
        """
        
        #self.renewable_potential -= power_consumed
        #self.renewable_potential = max(0.0, self.renewable_potential)  # Cannot go negative
        #self.renewable_potential_idx = digitize_clip(
        #    self.renewable_potential / self.max_value if self.max_value > 0 else 0,
        #    self.power_bins,
        #)

        # Recalculate delta_ph with updated potential
        self.update_delta_ph(power_consumed)
        
        logger.debug(
            "Agent %s consumed %.1f W, remaining potential=%.1f, new delta_ph=%.1f",
            source, power_consumed, self.renewable_potential, self.delta_ph
        )