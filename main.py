import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import copy
from tabulate import tabulate
from itertools import cycle
from analysis_tools import compute_q_diff_norm, plot_metric, check_stability, load_latest_evolution_csv, process_evolution_data, plot_coordination, clear_results_directories, digitize_clip

# Global constants
C_CONFORT = 0.5   # Comfort threshold for market cost
BINS = 7          # Defines how many intervals are used to discretize the power variables (renewables, no-renewables and demand).
SOC_INITIAL = 0.9
EPSILON_MIN = 0

# Creates files if they do not exist
os.makedirs("results", exist_ok=True)
os.makedirs("results/evolution", exist_ok=True)
os.makedirs("results/q_tables", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# -----------------------------------------------------
# Define the environment
# -----------------------------------------------------
class MultiAgentEnv:
    """
    Environment that:
    - Loads data from a CSV (irradance, wind speed, demand, price, etc.).
    - Discretizes these variables into bins to create states (solar_bins, wind_bins, etc.).
    - Advances step by step through the rows of the dataset (or randomly).
    - Can be managed in an episode loop.
    """

    def __init__(self, csv_filename, num_demand_bins=7, num_renewable_bins=7):
        """
        Parameters:
        - num_*_bins: Defines how many bins are used to discretize each variable.
        """
        
        # Load the DataFrame with offsets
        offsets_dict = {"demand": 0, "price": 0, "solar_power": 0, "wind_power": 0}
        self.dataset = self._load_data(csv_filename, offsets_dict)
        self.max_steps = len(self.dataset)
        self.max_value = self.dataset.apply(pd.to_numeric, errors='coerce').max().max()
        
        # Uniform Quantization Discretization
        # Define the bins to discretize each variable of interest
        # Adjust the ranges according to your actual dataset
        self.demand_bins = np.linspace(0, self.max_value, num_demand_bins)
        self.renewable_bins = np.linspace(0, self.max_value, num_renewable_bins)

        self.num_demand_bins = num_demand_bins
        self.num_renewable_bins = num_renewable_bins
        
        self.renewable_potential = 0
        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.renewable_bins)
        self.renewable_power = 0
        self.renewable_power_idx = digitize_clip(self.renewable_power, self.renewable_bins)
        self.demand_power = 0        
        self.demand_power_idx = digitize_clip(self.demand_power, self.renewable_bins)
        self.total_power = 0
        self.total_power_idx = digitize_clip(self.total_power, self.renewable_bins)
        self.price = 0
        self.delta_power = 0
        self.delta_power_idx = "surplus"
        self.scale_demand = 1
        
        # Initial state (discretized)
        self.state = None

    def _load_data(self, filename: str, offsets: dict = None) -> pd.DataFrame:
        """
        Loads a CSV file using pandas, applies offsets to the specified columns, and sets negative values in the 'demand' column to zero (0).

        Parameters:
        -----------
        filename: str
        Name of the CSV file to load, searched for in the specified path.
        offsets: dict, optional
        A dictionary with {column_name: offset_value} pairs.
        For example: {"Load": 5, "PV_Power": -10}.
        If None or empty, no offsets are applied.

        Return:
        --------
        df: pd.DataFrame
        DataFrame with the file contents and the specified transformations.
        """
        # Path to file
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep='[;,]', engine='python')

        # 1. Applies offsets to the indicated columns
        if offsets is not None:
            for col, offset_value in offsets.items():
                if col in df.columns:
                    df[col] += offset_value
                else:
                    print(f"Advertencia: La columna '{col}' no existe en el DataFrame. No se aplicó offset.")

        # 2. Replace negative values in 'demand' with 0
        # Adjust the 'demand' column name according to your CSV file.
        if 'demand' in df.columns:
            df['demand'] = df['demand'].clip(lower=0)

        return df

    def get_discretized_state(self, index):
        """
        It takes real values (irradance, wind, demand, price, etc.) and discretizes them into bins,
        returning a tuple like (idx_solar, idx_wind, idx_battery, idx_demand, idx_price).
        """
        row = self.dataset.iloc[index]
        
        self.demand_power = row["demand"]*self.scale_demand
        self.renewable_potential = row["solar_power"] + row["wind_power"]
        self.price = row["price"]
        self.time = row["Datetime"]

        self.demand_power_idx = digitize_clip(self.demand_power, self.demand_bins)
        self.renewable_potential_idx = digitize_clip(self.renewable_potential, self.renewable_bins)
        
        # The discretized state tuple is returned
        return (self.demand_power_idx, self.renewable_potential_idx)

# -----------------------------------------------------
# We define the Agent base class with Q-Table
# -----------------------------------------------------
class BaseAgent:
    """
    Base class for agents with Q-table.
    """
    def __init__(self, name, actions, alpha=0.1, gamma=0.9, kappa=10, sigma=10, mu=10, nu=10, beta=10):
        self.name = name
        self.actions = actions
        self.action = 0
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.mu = mu
        self.nu = nu
        self.beta = beta
        self.q_table = {}   
        self.power = 0.0
        self.idx = 0
    
    def choose_action(self, state, epsilon=0.1):
        """
        Select action with epsilon-greedy policy.
        """

        if random.random() < epsilon:
            self.action = random.choice(self.actions)
        else:
            # Choose the action with maximum Q
            q_values = self.q_table.get(state, {a: 0.0 for a in self.actions})
            self.action = max(q_values, key=q_values.get)
        
        return self.action

    def update_q_table(self, state, action, reward, next_state):

        """
        Update the Q-table according to Q-Learning:
          Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        # Choose the action with maximum Q
        q_values = self.q_table[state]
        current_q = q_values[action]
        
        # If next_state is not in the Q-table (edge case), we assume Q=0
        next_q_values = self.q_table.get(next_state, {a: 0.0 for a in self.actions})
        max_next_q = max(next_q_values.values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

# -----------------------------------------------------
# Specialized Agents (Solar, Wind, Battery, Grid, Load)
# Inherit from BaseAgent and add its rewards
# -----------------------------------------------------
class SolarAgent(BaseAgent):
    """
    Agent representing a solar power source.
    It uses a Q-table to decide whether to produce energy based on
    environmental conditions and system-level signals.
    
    State space: 
        (solar_potential_idx, total_power_idx, renewable_power_idx, demand_power_idx)
    """

    def __init__(self, env: MultiAgentEnv):
        super().__init__(name="solar", actions=[0, 1], alpha=0.1, gamma=0.9)

        # Discretization bins for potential generation (same for solar)
        self.solar_power_bins = np.linspace(0, env.max_value, env.num_renewable_bins)

        self.solar_state = 0
        self.potential = 0.0
        
    def get_discretized_state(self, env: MultiAgentEnv, index: int) -> tuple:
        """
        Returns the discretized state tuple for the solar agent:
        (solar_potential_idx, total_power_idx, demand_power_idx)

        Args:
            env (MultiAgentEnv): the shared environment.
            index (int): the row index of the current simulation step.

        Returns:
            tuple: Discretized state representation.
        """
        row = env.dataset.iloc[index]

        solar_potential = row["solar_power"]
        self.potential = solar_potential

        solar_potential_idx = digitize_clip(solar_potential, self.solar_power_bins)
        total_power_idx = digitize_clip(env.total_power, env.renewable_bins)
        demand_power_idx = env.demand_power_idx

        self.idx = solar_potential_idx

        return (solar_potential_idx, total_power_idx, demand_power_idx)

    def initialize_q_table(self, env: MultiAgentEnv) -> None:
        """
        Initializes the Q-table for all possible combinations of discretized states.
        """
        states = []
        for solar_idx in range(len(self.solar_power_bins)):
            for total_idx in range(len(env.renewable_bins)):
                for demand_idx in range(len(env.demand_bins)):
                    states.append((solar_idx, total_idx, demand_idx))

        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in states
        }

    def calculate_reward(
        self,
        solar_potential_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the solar agent based on its current action, solar potential,
        total system power (stigmergic signal), and demand.

        Parameters:
            solar_potential_idx (int): Discretized index of solar irradiance.
            total_power_idx (int): Discretized index of total power generated by the system (renewables + storage + grid).
            demand_power_idx (int): Discretized index of demand.

        Returns:
            float: Reward signal guiding the agent's learning.
        """
        power_gap = total_power_idx - demand_power_idx

        # Case 1: Action = produce (self.action == 1)
        if self.action == 1:
            if solar_potential_idx == 0:
                # Trying to produce without sun  → penalty
                return -self.mu * 100
            elif solar_potential_idx > 0 and power_gap >= 0:
                # With potential available and demand met → reward
                return self.kappa * 10
            else:
                # With available potential and energy deficit → reward
                return self.kappa

        # Case 2: Action = idle (self.action == 0)
        else:
            if solar_potential_idx == 0:
                # No produce without sun  → reward
                return self.kappa * 10
            elif solar_potential_idx > 0 and power_gap >= 0:
                # With available potential and satisfied demand → penalize
                return -self.mu * 10
            else:
                # With available potential and energy deficit → penalize
                return -self.mu * 100
            
class WindAgent(BaseAgent):
    """
    Agent representing a wind energy source.
    It uses a Q-table to decide whether to produce power based on
    local wind conditions and global system signals.

    State space:
        (wind_potential_idx, total_power_idx, demand_power_idx)
    """

    def __init__(self, env: MultiAgentEnv):
        super().__init__(name="wind", actions=[0, 1], alpha=0.1, gamma=0.9)

        # Discretization bins for wind and solar potential (same scale)
        self.wind_power_bins = np.linspace(0, env.max_value, env.num_renewable_bins)

        self.wind_state = 0
        self.potential = 0.0

    def get_discretized_state(self, env: MultiAgentEnv, index: int) -> tuple:
        """
        Returns the discretized state tuple for the wind agent:
        (wind_potential_idx, total_power_idx, demand_power_idx)

        Args:
            env (MultiAgentEnv): the shared environment.
            index (int): the row index of the current simulation step.

        Returns:
            tuple: Discretized state representation.
        """
        row = env.dataset.iloc[index]

        wind_potential = row["wind_power"]
        self.potential = wind_potential

        wind_potential_idx = digitize_clip(wind_potential, self.wind_power_bins)
        total_power_idx = digitize_clip(env.total_power, env.renewable_bins)
        demand_power_idx = env.demand_power_idx

        self.idx = wind_potential_idx

        return (wind_potential_idx, total_power_idx, demand_power_idx)

    def initialize_q_table(self, env: MultiAgentEnv) -> None:
        """
        Initializes the Q-table for all possible combinations of discretized states.
        """
        states = []
        for wind_idx in range(len(self.wind_power_bins)):
            for total_idx in range(len(env.renewable_bins)):
                for demand_idx in range(len(env.demand_bins)):
                    states.append((wind_idx, total_idx, demand_idx))

        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in states
        }

    def calculate_reward(
        self,
        wind_potential_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the wind agent based on its action, wind potential,
        total system power output, and current demand.

        Parameters:
            wind_potential_idx (int): Discretized index of available wind resource.
            total_power_idx (int): Discretized index of total system power (all agents).
            demand_power_idx (int): Discretized index of system demand.

        Returns:
            float: Reward signal guiding the wind agent's learning.
        """
        power_gap = total_power_idx - demand_power_idx

        # Case 1: Action = produce (self.action == 1)
        if self.action == 1:
            if wind_potential_idx == 0:
                # Trying to produce without sun  → penalty
                return -self.mu * 100
            elif wind_potential_idx > 0 and power_gap >= 0:
                # With potential available and demand met → reward
                return self.kappa * 10
            else:
                # With available potential and energy deficit → reward
                return self.kappa

        # Case 2: Action = idle (self.action == 0)
        else:
            if wind_potential_idx == 0:
                # No produce without sun  → reward
                return self.kappa * 10
            elif wind_potential_idx > 0 and power_gap >= 0:
                # With available potential and satisfied demand → penalize
                return -self.mu * 10
            else:
                # With available potential and energy deficit → penalize
                return -self.mu * 100

class BatteryAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, capacity_ah= 30, num_battery_soc_bins=3):
        super().__init__("battery", [0, 1, 2], alpha=0.1, gamma=0.9)
        
        # ["idle", "charge", "discharge"] -> [0, 1, 2]
        
        """
        Initializes the battery with a fixed capacity in Ah and an initial SOC of 50%.
        :param capacity_ah: Capacidad de la batería en Amperios-hora (Ah).
        """
        self.capacity_ah = capacity_ah  # Fixed capacity in Ah

        self.soc = SOC_INITIAL  # Initial state of charge in %
        self.battery_state = 0  # Initial operation state
        self.max_soc_idx = 2

        # Discretization by uniform quantization
        # Define the bins to discretize each variable of interest
        self.battery_soc_bins = np.linspace(0, 1, num_battery_soc_bins)

    def update_soc(
            self,
            power_w: float,
            dt_h: float = 1.0,
            nominal_voltage: float = 48.0  #default value in volts
        ) -> None:
        """
        Updates the battery's state of charge (SOC).

        Parameters
        ----------
        power_w : float
        Instantaneous power (W).
        + discharge → SOC ↓
        – charge → SOC ↑
        dt_h : float, default 1.0
        Simulation step duration in hours.
        nominal_voltage : float, default 48.0
        Nominal battery voltage (V). This can be overridden if you want to use a different value for a specific call.
        """
        # Capacity in Wh using nominal voltage
        capacity_wh = self.capacity_ah * nominal_voltage

        # Energy transferred during the time step

        energy_wh = power_w * dt_h
        capacity_wh_new = self.soc*capacity_wh - energy_wh

        # Integrate and saturate in [0, 1]
        new_soc = capacity_wh_new/capacity_wh
        self.soc = max(0.0, min(1.0, new_soc))

        # Discrete index (optional, for your agent)

        self.idx = digitize_clip(
            self.soc, self.battery_soc_bins
        )

    def initialize_q_table(self, env: MultiAgentEnv) -> None:
        """
        Initializes the Q-table for all possible combinations of discretized states:
        (SOC_idx, renewable_potential_idx, total_power_idx, demand_power_idx)
        """
        states = []
        for soc_idx in range(len(self.battery_soc_bins)):
            for renewable_idx in range(len(env.renewable_bins)):
                for total_power_idx in range(len(env.renewable_bins)):
                    for demand_idx in range(len(env.demand_bins)):
                        states.append((soc_idx, renewable_idx, total_power_idx, demand_idx))

        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in states
        }

    def get_discretized_state(self, env: MultiAgentEnv, index: int) -> tuple:
        """
        Returns the discretized state of the battery agent:
        (SOC_idx, renewable_potential_idx, total_power_idx, demand_power_idx)

        Assumes env.total_power and env.demand_power_idx are already computed before calling this.
        """
        # Update and discretize SOC
        self.idx = digitize_clip(self.soc, self.battery_soc_bins)

        # Discretized state components from environment
        renewable_potential_idx = env.renewable_potential_idx
        total_power_idx = digitize_clip(env.total_power, env.renewable_bins)
        demand_power_idx = env.demand_power_idx

        return (self.idx, renewable_potential_idx, total_power_idx, demand_power_idx)

    def calculate_reward(
        self,
        renewable_potential_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the battery agent based on its SOC, system balance,
        and current action (self.action ∈ {0: idle, 1: charge, 2: discharge}).

        Parameters:
            renewable_potential_idx (int): Discretized index of available renewable generation.
            total_power_idx (int): Discretized index of total system power output.
            demand_power_idx (int): Discretized index of demand.

        Returns:
            float: Reward signal guiding the battery agent.
        """
        power_gap = total_power_idx - demand_power_idx  # surplus if > 0, shortage if < 0

        # Action = discharge
        if self.action == 2:
            if self.idx == 0:
                # Trying to discharge with empty battery → strong penalty
                return -self.mu
            elif power_gap <= 0:
                # Discharging to help system under deficit → reward
                return self.kappa * abs(power_gap or 1)
            else:
                # Discharging when system has excess → penalty
                return -self.sigma * self.idx

        # Action = charge
        elif self.action == 1:
            if self.idx == self.max_soc_idx:
                # Trying to charge at full SOC → penalty
                return -self.mu
            elif renewable_potential_idx > demand_power_idx and power_gap > 0:
                # Charging with renewable surplus → reward
                return self.kappa
            else:
                # Charging when there's no surplus → mild penalty
                return -self.sigma

        # Action = idle
        else:
            if self.idx > 0 and power_gap < 0:
                # Battery could discharge to help system but stays idle → penalty
                return -self.beta
            elif self.idx < self.max_soc_idx and power_gap > 0 and renewable_potential_idx > demand_power_idx:
                # Battery could charge from renewable surplus but stays idle → penalty
                return -self.beta
            else:
                # Acceptable inaction → neutral or small reward
                return self.nu

class GridAgent(BaseAgent):
    """
    GridAgent represents the utility grid in the microgrid system.
    It decides whether to supply energy based on the system's battery SOC,
    total delivered power, and current demand.
    """

    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        """
        Initialize the GridAgent.

        Args:
            env (MultiAgentEnv): The environment instance.
            ess (BatteryAgent): Reference to the battery agent for SOC access.
        """
        super().__init__("grid", [0, 1], alpha=0.1, gamma=0.9)  # 0=idle, 1=produce
        self.ess = ess  # Battery agent reference

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Initializes the Q-table using the discretized state space:
        (battery_soc_idx, total_power_idx, demand_power_idx).
        """
        states = []
        for soc_idx in range(len(self.ess.battery_soc_bins)):           # battery SOC index
            for total_power_idx in range(len(env.renewable_bins)):  # total power output index
                for demand_idx in range(len(env.demand_bins)):      # demand index
                    states.append((soc_idx, total_power_idx, demand_idx))

        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in states
        }

    def get_discretized_state(self, env: MultiAgentEnv, index=None):
        """
        Returns the current discretized state tuple.

        Args:
            env (MultiAgentEnv): The environment instance.
            index (optional): Ignored, present for compatibility.

        Returns:
            tuple: (battery_soc_idx, total_power_idx, demand_power_idx)
        """
        return (
            self.ess.idx,
            env.total_power_idx,
            env.demand_power_idx
        )

    def calculate_reward(
        self,
        battery_soc_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the GridAgent based on the current system state
        and the selected action stored in self.action (0: idle, 1: produce).
        The reward encourages the grid to support the microgrid only when strictly
        necessary, considering battery availability, demand and total system output.

        Parameters:
            battery_soc_idx (int): Discretized index of the battery's state of charge (SOC).
            total_power_idx (int): Discretized index of the total power delivered by all agents.
            demand_power_idx (int): Discretized index of the total system demand.

        Returns:
            float: Reward signal guiding the grid agent.
        """
        power_gap = total_power_idx - demand_power_idx  # >0 = surplus, <0 = shortage

        # Action = produce
        if self.action == 1:
            if power_gap < 0 and battery_soc_idx == 0:
                # Grid is supplying during shortage and battery is empty → necessary → strong reward
                return self.kappa * 100
            elif power_gap >= 0 and battery_soc_idx == 0:
                # The battery has no energy but the demand is satisfied → penalize
                return -self.sigma 
            elif power_gap < 0 and battery_soc_idx > 0:
                # Grid is helping, but battery could have helped → mild penalty
                return -self.sigma
            else:
                # Grid produces despite sufficient system power → wasteful → strong penalty
                return -self.sigma*100

        # Action = idle
        elif self.action == 0:
            if power_gap < 0 and battery_soc_idx == 0:
                # Grid is not supplying during shortage and battery is empty → necessary → strong penalty
                return -self.sigma * 100
            elif power_gap >= 0 and battery_soc_idx == 0:
                # The battery has no energy but the demand is satisfied → reward
                return self.kappa 
            elif power_gap < 0 and battery_soc_idx > 0:
                # Grid is not helping, but battery could have helped → reward
                return self.kappa
            else:
                # The grid does not produce despite having sufficient system power and energy in the batteries → excess → strong reward
                return self.kappa*100

class LoadAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        """
        Controllable Load Agent that learns when to consume energy based on battery level,
        renewable potential, system-wide power balance, and market price comfort.

        :param env: Reference to the simulation environment.
        :param ess: Reference to the battery agent providing SoC index.
        """
        super().__init__("load", [0, 1], alpha=0.1, gamma=0.9)

        self.env = env
        self.ess = ess
        self.comfort = 5  # User-defined maximum acceptable market price
        self.comfort_idx = "acceptable"

    def get_discretized_state(self, env: MultiAgentEnv, index: int) -> tuple:
        """
        Constructs the discretized state tuple using battery SoC, renewable index, 
        power balance and market price acceptability.

        :param index: Index in the environment's dataset for the current timestep.
        :return: Tuple representing the discrete state.
        """
        row = self.env.dataset.iloc[index]
        market_price = row["price"]

        # Get components from environment and battery agent
        battery_soc_idx = self.ess.idx
        renewable_potential_idx = env.renewable_potential_idx
        delta_power_idx = env.delta_power_idx  # Should be set externally in env
        self.comfort_idx = 'acceptable' if market_price <= self.comfort else 'expensive'

        return (battery_soc_idx, renewable_potential_idx, delta_power_idx, self.comfort_idx)

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Initializes the Q-table with all possible combinations of discrete state variables.
        """
        battery_bins = self.ess.battery_soc_bins
        renewable_bins = self.env.renewable_bins
        delta_power_labels = ['deficit', 'surplus']
        comfort_labels = ['acceptable', 'expensive']

        states = []
        for b in range(len(battery_bins)):
            for r in range(len(renewable_bins)):
                for d in delta_power_labels:
                    for c in comfort_labels:
                        states.append((b, r, d, c))

        self.q_table = {
            state: {action: 0.0 for action in self.actions}
            for state in states
        }

    def calculate_reward(self, battery_soc_idx: int, renewable_potential_idx: int, 
                        delta_power_idx: str) -> float:
        """
        Computes reward based on full state context and agent's action.
        Penalizes consumption during grid surplus and encourages use of local/green energy.

        :param battery_soc_idx: Discretized battery SoC index.
        :param renewable_potential_idx: Discretized renewable generation index.
        :param delta_power_idx: 'surplus' or 'deficit'.
        :param comfort_idx: 'acceptable' or 'expensive'.
        :return: Reward value based on symbolic context.
        """
        # Thresholds to define "low" and "high"
        low_soc = battery_soc_idx <= 1
        low_renewables = renewable_potential_idx <= 1

        if self.action == 1:  # Turn ON
            if delta_power_idx == 'surplus':
                if low_soc and low_renewables:
                    # Grid surplus: penalize unnecessary consumption
                    return -self.kappa
                else:
                    # Surplus from battery or renewables: reward
                    return self.kappa
            elif delta_power_idx == 'deficit':
                if self.comfort_idx == 'acceptable':
                    return self.kappa / 2  # Small reward for affordable power
                else:
                    return -self.kappa  # Expensive + deficit → penalize

        elif self.action == 0:  # Turn OFF
            if delta_power_idx == 'deficit' and self.comfort_idx == 'expensive':
                return self.kappa  # Wise decision to save cost and avoid overloading
            elif delta_power_idx == 'surplus':
                if low_soc and low_renewables:
                    return self.kappa  # Grid surplus → good to avoid it
                else:
                    return -self.kappa / 2  # Missed opportunity to use green/local energy

        return 0.0

# -----------------------------------------------------
# Training simulation
# -----------------------------------------------------
class Simulation:
    def __init__(self, num_episodes=10, epsilon=1, learning=True, filename="Case1.csv"):
        self.num_episodes = num_episodes
        self.instant = {} 
        self.evolution = []
        self.df = pd.DataFrame()
        self.df_episode_metrics = pd.DataFrame()
        self.prev_q_tables = {} # Dictionary to store the previous Q-table of each agent

        
        # Create the environment that loads the CSV and discretizes
        self.env = MultiAgentEnv(csv_filename=filename, num_demand_bins=BINS, num_renewable_bins=BINS)
        
        # Obtains points automatically from the database
        self.max_steps = self.env.max_steps
        
        # The storage agent is instantiated
        bat_ag = BatteryAgent(self.env)

        # The set of agents is defined
        self.agents = [
            SolarAgent(self.env),
            WindAgent(self.env),
            bat_ag,
            GridAgent(self.env, bat_ag),
            LoadAgent(self.env, bat_ag)
        ]
        
        # Training parameters
        self.epsilon = epsilon  # Exploration \epsilon (0=exploitation, 1=exploration)

        
        # Initialize Q-tables
        if learning:
            for agent in self.agents:
                agent.initialize_q_table(self.env)

    def step(self, index):
        self.env.get_discretized_state(index)
        agent_states = {}
        
        for agent in self.agents:
            agent_type = type(agent).__name__  # Gets the name of the agent class
            agent_states[agent_type] = agent.get_discretized_state(self.env, index)
            agent.idx = agent_states[agent_type][0]        
       
        return agent_states
        
    def run(self):

        # Create the loop for the episodes
        for ep in range(self.num_episodes):
            
            # For each episode the power values ​​are initialized
            self.env.total_power = 0
            self.env.renewable_power  = 0.0
            self.env.renewable_power_idx = 0                
            bat_power = 0.0
            grid_power = 0.0
            solar_power = 0.0
            wind_power = 0.0
            loadc_power = 0.0
            battery_agent = None

            # Incorporates randomness in the demand so that the training
            self.env.scale_demand = random.uniform(0.1, 7)
            
            # Save snapshot of previous Q-tables (only if not the first episode)
            if ep > 0:
                for agent in self.agents:
                    agent_key = type(agent).__name__
                    self.prev_q_tables[agent_key] = copy.deepcopy(agent.q_table)
            else:
                for agent in self.agents:
                    agent_key = type(agent).__name__
                    self.prev_q_tables[agent_key] = {state: {a:0.0 for a in agent.actions} for state in agent.q_table}
            
            # Initialization of the evaluation by episode
            self.evolution = []

            # Initializes the soc at each episode start
            for agent in self.agents:
                if isinstance(agent, BatteryAgent):
                    # Incorporates randomness in the demand so that the training
                    agent.soc = random.uniform(0, 1)
                    # Stops the loop upon finding the battery agent
                    break  

            for i in range(self.max_steps-1):
                
                # We reset the environment and agents at the start of each episode.
                state = self.step(i)                

                self.instant["time"] = self.env.time                

                # Select the action
                for agent in self.agents:
                    if isinstance(agent, SolarAgent):
                        agent.choose_action(state['SolarAgent'], self.epsilon)

                        agent.solar_state = agent.action
                        agent.power = agent.potential*agent.action
                        solar_power = agent.power

                        self.instant["solar_potential"] = agent.potential
                        self.instant["solar_discrete"] = agent.idx
                        self.instant["solar"] = agent.power
                        self.instant["solar_state"] = agent.solar_state
                    elif isinstance(agent, WindAgent):
                        agent.choose_action(state['WindAgent'], self.epsilon)

                        agent.wind_state = agent.action
                        agent.power = agent.potential*agent.action
                        wind_power = agent.power

                        self.instant["wind_potential"] = agent.potential
                        self.instant["wind_discrete"] = agent.idx
                        self.instant["wind"] = agent.power
                        self.instant["wind_state"] = agent.wind_state
                    elif isinstance(agent, BatteryAgent):
                        agent.choose_action(state['BatteryAgent'], self.epsilon)
                        agent.battery_state = agent.action

                        # Charging
                        if agent.action == 1:
                            agent.power = -abs(self.env.demand_power - (solar_power + wind_power))
                        # Discharging
                        elif agent.action == 2:
                            agent.power = abs(self.env.demand_power - (solar_power + wind_power))
                        # Idle
                        else:                        
                            agent.power = 0.0

                        bat_power = agent.power
                        battery_agent = agent

                        self.instant["bat"] = agent.power
                        self.instant["bat_soc"] = agent.soc
                        self.instant["bat_soc_discrete"] = agent.idx
                        self.instant["bat_state"] = agent.battery_state
                        
                        agent.update_soc(power_w=agent.power)

                    elif isinstance(agent, GridAgent):
                        agent.choose_action(state['GridAgent'], self.epsilon)
                        
                        agent.grid_state = agent.action
                        # Sell
                        if agent.action == 1: 
                            agent.power = abs(self.env.demand_power - (solar_power + wind_power) - bat_power)
                        # Idle
                        else: 
                            agent.power = 0
                        
                        grid_power = agent.power
                        agent.idx = digitize_clip(agent.power, self.env.renewable_bins)

                        self.instant["grid"] = agent.power
                        self.instant["grid_state"] = agent.grid_state
                        self.instant["grid_discrete"] = agent.idx
                    else:
                        agent.choose_action(state['LoadAgent'], self.epsilon)
                        agent.load_state = agent.action

                        # Turn ON
                        if agent.action == 1: 
                            agent.power = 0
                        # Turn OFF
                        else:    
                            # The power that is reduced with the management of controllable loads is known
                            agent.power = -15
                        
                        self.instant["load_state"] = agent.load_state
                        self.instant["load_comfort_idx"] = agent.comfort_idx
                        loadc_power = agent.power

                # Updates variables in environment
                self.env.renewable_power = wind_power + solar_power
                self.env.renewable_power_idx = digitize_clip(self.env.renewable_power, self.env.renewable_bins)

                self.env.total_power = self.env.renewable_power + bat_power + (grid_power + loadc_power)
                self.env.total_power_idx = digitize_clip(self.env.total_power, self.env.renewable_bins)

                self.env.demand_power = self.env.demand_power + loadc_power
                self.env.demand_power_idx = digitize_clip(self.env.demand_power, self.env.demand_bins)

                self.delta_power = self.env.total_power - self.env.demand_power
                self.delta_power_idx = 'deficit' if self.delta_power <= 0 else 'surplus'

                self.instant["renewable_potential"] = self.env.renewable_potential
                self.instant["renewable_potential_discrete"] = self.env.renewable_potential_idx
                self.instant["renewable_power"] = self.env.renewable_power
                self.instant["renewable_power_discrete"] = self.env.renewable_power_idx
                self.instant["total"] = self.env.total_power
                self.instant["demand"] = self.env.demand_power
                self.instant["dif"] = self.delta_power
                self.instant["dif_idx"] = self.delta_power_idx
                self.instant["total_discrete"] = self.env.total_power_idx
                self.instant["demand_discrete"] = self.env.demand_power_idx
                self.instant["price"] = self.env.price

                # Now we calculate the individual reward per agent and update the Q-table               
                next_state = self.step(i + 1)  # We advance the environment by an index
                
                for agent in self.agents:
                    agent_type = type(agent).__name__  # Gets the name of the agent class
                    
                    # We calculate the reward according to the type of agent
                    if isinstance(agent, SolarAgent):
                        reward = agent.calculate_reward(
                            solar_potential_idx=agent.potential,
                            total_power_idx=self.env.total_power_idx,
                            demand_power_idx=self.env.demand_power_idx
                        )
                        self.instant["reward_solar"] = reward
                    elif isinstance(agent, WindAgent):
                        reward = agent.calculate_reward(
                            wind_potential_idx=agent.potential,
                            total_power_idx=self.env.total_power_idx,
                            demand_power_idx=self.env.demand_power_idx
                        )
                        self.instant["reward_wind"] = reward
                    elif isinstance(agent, BatteryAgent):
                        
                        reward = agent.calculate_reward(
                                renewable_potential_idx=self.env.renewable_potential_idx,
                                total_power_idx=self.env.total_power_idx,
                                demand_power_idx=self.env.demand_power_idx
                                )       

                        self.instant["bat_soc"] = agent.soc
                        self.instant["reward_bat"] = reward
                        battery_agent = agent
                    elif isinstance(agent, GridAgent):
                        reward = agent.calculate_reward(
                            battery_soc_idx=battery_agent.idx,
                            total_power_idx=self.env.total_power_idx,
                            demand_power_idx=self.env.demand_power_idx)
                        self.instant["reward_grid"] = reward                        
                    else:
                        reward = agent.calculate_reward(
                            battery_soc_idx=battery_agent.idx,
                            renewable_potential_idx=self.env.renewable_potential_idx,
                            delta_power_idx= self.delta_power_idx)

                        self.instant["reward_demand"] = reward
                    
                    # We updated Q-table
                    agent.update_q_table(state[agent_type], agent.action, reward, next_state[agent_type])

                # We update the current status
                state = next_state
                self.evolution.append(copy.deepcopy(self.instant))
              
            # The learning relationship changes with each episode
            if self.num_episodes > 1:
                self.epsilon = max(EPSILON_MIN, 1 - (ep / (self.num_episodes - 1)))
            else:
                self.epsilon = self.epsilon                
            
            self.df = pd.DataFrame(self.evolution)
            self.df.to_csv(f"results/evolution/learning_{ep}.csv", index=False)
            
            self.update_episode_metrics(ep, self.df)      
                        
            # Save current Q-table to Excel
            for agent in self.agents:
                df_q = pd.DataFrame([
                    {
                        "State": str(state),
                        "Action": action,
                        "Q_value": q_value
                    }
                    for state, actions in agent.q_table.items()
                    for action, q_value in actions.items()
                ])

                # Filename
                filename_q = f"results/q_tables/qtable_{agent.name}_ep{ep}.xlsx"

                df_q.to_excel(filename_q, index=False, engine="openpyxl")

            # Calculate episode metrics
            iae = self.calculate_iae()
            var_dif = self.df['dif'].var()

            # Calculate Q-difference norms
            q_norms = {
                type(agent).__name__: compute_q_diff_norm(agent.q_table, self.prev_q_tables[type(agent).__name__])
                for agent in self.agents
            }

            # Calculate average rewards
            mean_rewards = {}
            for agent in self.agents:
                col_name = f"reward_{agent.name.lower()}"
                if col_name in self.df.columns:
                    mean_rewards[agent.name] = self.df[col_name].mean()
                else:
                    mean_rewards[agent.name] = 0.0

            # Record in DataFrame
            row = {
                "Episode": ep,
                "IAE": iae,
                "Var_dif": var_dif,
            }
            row.update({f"Q_Norm_{k}": v for k, v in q_norms.items()})
            row.update({f"Mean_Reward_{k}": v for k, v in mean_rewards.items()})

            self.df_episode_metrics = pd.concat(
                [self.df_episode_metrics, pd.DataFrame([row])],
                ignore_index=True
            )

            # Save to Excel UTF-8
            self.df_episode_metrics.to_excel(
                "results/metrics_episode.xlsx",
                index=False,
                engine="openpyxl"
            )
            
            print(f"End of episode  {ep+1}/{self.num_episodes} with epsilon {self.epsilon}")

        # Interactive power charts
        self.plot_data_interactive(
            df=self.df,
            columns_to_plot=["solar_potential", "demand", "bat_soc", "grid", "wind_potential", "price"],
            title="Environment variables",
            save_static_plot=True,
            static_format="svg",  # o "png", "pdf"
            static_filename="results/plots/env_plot"
        )              
        
        # Interactive graphs of actions and df
        self.plot_data_interactive(
            df=self.df,
            columns_to_plot=["dif"],
            title="Energy balance",
            save_static_plot=True,
            static_format="svg",  # o "png", "pdf"
            static_filename="results/plots/actions_plot"
        ) 

        self.plot_metric('Total Reward')  # Puedes usar 'Total Reward' u otra métrica
        
        # Graph IAE
        plot_metric(
            self.df_episode_metrics,
            field="IAE",
            ylabel="Integral Absolute Error",
            filename_svg="results/plots/IAE_over_episodes.svg"
        )

        # Graph Variance of Diff
        plot_metric(
            self.df_episode_metrics,
            field="Var_dif",
            ylabel="Variance of dif",
            filename_svg="results/plots/Var_dif_over_episodes.svg"
        )

        # Graphing Q norms by agent
        for agent in self.agents:
            agent_key = type(agent).__name__
            field_q = f"Q_Norm_{agent_key}"
            plot_metric(
                self.df_episode_metrics,
                field=field_q,
                ylabel=f"Q Norm Difference ({agent_key})",
                filename_svg=f"results/plots/Q_Norm_{agent_key}.svg"
            )        
        
        # Calculate IAE threshold as median of first 50 episodes ±10%
        iae_median = self.df_episode_metrics[self.df_episode_metrics['Episode'] < 50]['IAE'].median()
        iae_threshold = iae_median * 1.10  # +10%

        # Check stability
        stability = check_stability(self.df_episode_metrics, iae_threshold=iae_threshold)

        print("\n=== Stability Check ===")
        print(f"IAE Threshold: {iae_threshold:.3f}")
        print(f"Mean IAE (last 200 eps): {stability['IAE_mean']:.3f} -> {'OK' if stability['IAE_stable'] else 'NOT STABLE'}")
        print(f"Mean Var (last 200 eps): {stability['Var_mean']:.3f} -> {'OK' if stability['Var_stable'] else 'NOT STABLE'}")

        if stability['IAE_stable'] and stability['Var_stable']:
            print("SYSTEM DECLARED STABLE ✅")
        else:
            print("SYSTEM NOT STABLE ⚠️")        
       
        
        self.show_performance_metrics()

        return self.agents

    def calculate_ise(self) -> float:
        """
        Calculates the ISE (Integral Square Error) over the 'dif' column

.

        :return: Valor de ISE.
        """
        ise = (self.df['dif'] ** 2).sum()
        return ise

    def calculate_mean(self) -> float:
        """
        Calculates the ISE (Integral Square Error) over the 'dif' column

.

        :return: Valor de ISE.
        """
        mean= self.df['dif'].mean()
        return mean

    def calculate_iae(self) -> float:
        """
        Calculates the IAE (Integral Absolute Error) over the 'dif' column.

        :return: Valor de IAE.
        """
        iae = self.df['dif'].abs().sum()
        return iae

    def calculate_rep(self) -> float:
        """
        Calculates the REP (Renewable Energy Penetration), percentage of renewable energy over the total.

        :return: Valor de REP como porcentaje.
        """
        total_renewable_energy = self.df['solar_state'].sum()
        total_energy = self.df.shape[0]
        if total_energy == 0:
            return 0.0  # avoid division by zero

        rep = (total_renewable_energy / total_energy) * 100
        return rep

    def calculate_grid(self) -> float:
        """
        Calculates the REP (Renewable Energy Penetration), percentage of renewable energy over the total.

        :return: Valor de REP como porcentaje.
        """
        total_grid_energy = self.df['grid_state'].sum()
        total_energy = self.df.shape[0]
        if total_energy == 0:
            return 0.0  # avoid division by zero

        rep = (total_grid_energy / total_energy) * 100
        return rep

    def show_performance_metrics(self):
        """
        Displays a table with the calculated performance metrics: ISE, IAE and REP.
        """
        results = [
            ["MEAN (Mean)", f"{self.calculate_mean():.3f}"],
            ["ISE (Integral Square Error)", f"{self.calculate_ise():.3f}"],
            ["IAE (Integral Absolute Error)", f"{self.calculate_iae():.3f}"],
            ["REP (Renewable Energy Penetration)", f"{self.calculate_rep():.2f}%"],
            ["GEP (Grid Energy Penetration)", f"{self.calculate_grid():.2f}%"]
        ]
        print(tabulate(results, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    
    def compute_reward_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = []

        for agent in ['solar', 'wind', 'bat', 'grid', 'demand']:
            rewards = df[f'reward_{agent}']
            metrics = {
                'Agent': agent,
                'Rewards (count)': (rewards > 0).sum(),
                'Penalties (count)': (rewards < 0).sum(),
                'Total Reward': rewards.sum(),
                'Average Reward': rewards.mean(),
                'Max Reward': rewards.max(),
                'Min Reward': rewards.min(),
                'Variance': rewards.var(),
                'Interpretation': ''
            }

            if metrics['Average Reward'] > 0.1:
                metrics['Interpretation'] = 'Consistently positive learning'
            elif metrics['Average Reward'] > 0:
                metrics['Interpretation'] = 'Acceptable learning'
            else:
                metrics['Interpretation'] = 'High level of penalties, review needed'

            summary.append(metrics)

        return pd.DataFrame(summary)

    def update_episode_metrics(self, episode: int, df_episode: pd.DataFrame):
        df_metrics = self.compute_reward_metrics(df_episode)
        df_metrics['Episode'] = episode
        self.df_episode_metrics = pd.concat([self.df_episode_metrics, df_metrics], ignore_index=True)

    def plot_metric(self, metric_field='Total Reward', output_format='svg', filename='results/plots/metric_plot'):
        """
        Genera una gráfica de métricas por agente y la guarda como archivo vectorial o de High resolution.
        
        Parámetros:
            metric_field (str): Nombre del campo de métrica a graficar.
            output_format (str): 'svg' para vectorial, 'png' para imagen de High resolution.
            filename (str): Nombre base del archivo sin extensión.
        """
        plt.figure(figsize=(10, 6))
        for agent in self.df_episode_metrics['Agent'].unique():
            agent_df = self.df_episode_metrics[self.df_episode_metrics['Agent'] == agent]
            plt.plot(agent_df['Episode'], agent_df[metric_field], label=f'{agent} - {metric_field}')

        plt.xlabel("Episode")
        plt.ylabel(metric_field)
        plt.title(f"{metric_field} per Episode by Agent")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save in high quality


        if output_format == 'svg':
            plt.savefig(f"{filename}.svg", format='svg')
        elif output_format == 'png':
            plt.savefig(f"{filename}.png", format='png', dpi=300)  # High resolution
        else:
            raise ValueError("output_format debe ser 'svg' o 'png'")

        plt.close()  # Closes the figure to free memory




        return self.df_episode_metrics     

    def plot_data_interactive(
            self,
            df: pd.DataFrame,                     # now receives a DataFrame

            columns_to_plot: list[str] | None = None,
            title: str = "Environment variables",
            save_static_plot: bool = False,
            static_format: str = "svg",
            static_filename: str = "interactive_plot_export",
            soc_keyword: str = "soc",             # pattern that detects SOC columns

            soc_scale: float = 100.0              # 0-1  → 0-100 %
        ):
        # ---------- 1. Validaciones -------------------------------
        if df.empty:
            print("No hay datos para graficar (DataFrame vacío).")
            return

        if columns_to_plot is None:
            columns_to_plot = df.columns.tolist()

        valid_cols = [c for c in columns_to_plot if c in df.columns]
        if not valid_cols:
            print("Las columnas indicadas no existen en el DataFrame.")
            return

        # ---------- 2. Clasificación ------------------------------
        soc_cols  = [c for c in valid_cols if soc_keyword.lower() in c.lower()]
        prim_cols = [c for c in valid_cols if c not in soc_cols]

        # ---------- 3. Figura y ejes ------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx() if soc_cols else None
        if ax2:
            ax2.patch.set_visible(False)          # transparent background


        base_colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle1 = cycle(base_colors)
        color_cycle2 = cycle(base_colors[1:]+base_colors[:1])

        lines, labels = [], []

        # ---------- 4. Potencias (eje primario) -------------------
        for col in prim_cols:
            l, = ax1.plot(df[col], label=col,
                        color=next(color_cycle1), zorder=3)
            lines.append(l); labels.append(col)

        # ---------- 5. SOC (eje secundario) ----------------------
        if ax2:
            for col in soc_cols:
                data  = df[col] * soc_scale
                label = f"{col} [%]"
                l, = ax2.plot(data, '--', lw=2,
                            color=next(color_cycle2), label=label, zorder=4)
                lines.append(l); labels.append(label)
            ax2.set_ylabel("State of Charge [%]")
            ax2.set_ylim(0, 100)

        # ---------- 6. Estilo ------------------------------------
        ax1.set_title(title)
        ax1.set_xlabel("Hours")
        ax1.set_ylabel("Power [kW]")
        ax1.grid(True, zorder=0)
        ax1.legend(lines, labels, loc="upper right")

        # ---------- 7. Check-boxes -------------------------------
        rax = fig.add_axes([0.05, 0.4, 0.17, 0.2])
        checks = CheckButtons(rax, labels, [True]*len(labels))

        def toggle(label: str):
            idx = labels.index(label)
            lines[idx].set_visible(not lines[idx].get_visible())
            plt.draw()

        checks.on_clicked(toggle)

        # ---------- 8. Exportación estática (sin check-boxes) ----
        if save_static_plot:
            valid_formats = {"svg", "png", "pdf"}
            if static_format.lower() not in valid_formats:
                raise ValueError(f"Formato '{static_format}' no soportado ({valid_formats})")

            # Temporarily hide the checkbox axis

            rax.set_visible(False)
            dpi = 300 if static_format.lower() == "png" else None
            fig.savefig(f"{static_filename}.{static_format}",
                        format=static_format, dpi=dpi)
            print(f"Gráfico guardado como {static_filename}.{static_format}")
            # Show it again for the interactive view
            rax.set_visible(True)

        plt.show()

# -----------------------------------------------------
# Main program
# -----------------------------------------------------
if __name__ == "__main__":
    
    # Cleans up files contained in temporary directories
    clear_results_directories()

    # Simulation setup
    sim = Simulation(num_episodes=300, epsilon=1, learning=True, filename="Case1_1.csv")
    sim.run()

    # Graphs with the results of the interaction when the agents have completed the learning
    df_raw = load_latest_evolution_csv()
    df_clean = process_evolution_data(df_raw)
    plot_coordination(df_clean)