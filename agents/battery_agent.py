import numpy as np
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip


@register_agent("battery")
class BatteryAgent(BaseAgent):
    """Battery agent with three discrete actions.

    Actions (convention):
        0 -> idle
        1 -> charge
        2 -> discharge
    """

    def __init__(self, env,  name="battery",capacity_ah=3, num_battery_soc_bins=5, state_space=None, **kwargs):
        super().__init__(env, name, [0, 1, 2], state_space=state_space, **kwargs)

        # Limits and electrical parameters
        limits = kwargs.get("limits", {})
        self.capacity_ah = limits.get("capacity_ah", capacity_ah)
        self.v_nom = limits.get("v_nom", 48.0)
        self.soc_min = limits.get("soc_min", 0.0)
        self.soc_max = limits.get("soc_max", 1.0)
        self.p_charge_max = limits.get("p_charge_max", 100.0)      # W (negative when charging)
        self.p_discharge_max = limits.get("p_discharge_max", 100.0)  # W (positive when discharging)

        # State of charge (continuous [0,1]) and discretization bins
        self.soc = 0.5
        self.battery_soc_bins = np.linspace(self.soc_min, self.soc_max, num_battery_soc_bins)
        # Initialize discrete index consistent with initial SOC
        self.idx = digitize_clip(self.soc, self.battery_soc_bins)

    def update_power(self, env):
        """Map action to battery power with correct sign convention.

        Action semantics:
            0 -> idle       -> power = 0
            1 -> charge     -> power < 0 (consumption from grid/renewables)
            2 -> discharge  -> power > 0 (generation to the system)

        Notes:
            - Magnitudes are clipped by p_charge_max / p_discharge_max.
            - A richer policy could modulate power with potentials or prices.
        """
        if self.action == 1:  # charge
            self.power = -abs(self.p_charge_max)
        elif self.action == 2:  # discharge
            self.power = +abs(self.p_discharge_max)
        else:  # idle
            self.power = 0.0

        # 
        self.update_soc(power_w=self.power)


    def update_soc(
            self,
            power_w: float,
            dt_h: float = 1.0,
            nominal_voltage: float = 48.0  # default value in volts
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
        capacity_wh_new = self.soc * capacity_wh - energy_wh

        # Integrate and saturate in [0, 1]
        new_soc = capacity_wh_new / capacity_wh if capacity_wh > 0 else self.soc
        self.soc = max(self.soc_min, min(self.soc_max, new_soc))

        # Discrete index (optional, for your agent)
        self.idx = digitize_clip(self.soc, self.battery_soc_bins)

    def initialize_q_table(self, env):
        states = [
            (soc, dem, tot)
            for soc in range(len(self.battery_soc_bins))
            for dem in range(len(env.power_bins))
            for tot in range(len(env.power_bins))
        ]
        self.q_table = {s: {a: 0.0 for a in self.actions} for s in states}

    def calculate_reward(self, state):
        """
        Calculate the reward for the Battery Agent based on its internal state (SOC),
        current action, and the system power balance.

        Reward structure follows the equation:

            R_i = {
                ψ * |ΔP| * SOC,                       if s_BAT = 2 and ΔP < 0 and SOC > 0
                -σ,                                   if s_BAT = 2 and (ΔP >= 0 or SOC = 0)
                ν * ΔP * (SOC_max - SOC),             if s_BAT = 1 and ΔP > 0
                -β * |ΔP|,                            if s_BAT = 1 and ΔP ≤ 0
                -ξ * |ΔP|,                            if s_BAT = 0 and |ΔP| > 0
                0,                                    Otherwise
            }

        Parameters
        ----------
        demand_idx : float
            Instantaneous demand index of the system.
        total_idx : float
            Instantaneous total generation index of the system.

        Returns
        -------
        float
            Reward value for the current action and state.
        """

        # --- Constants (can be tuned or read from configuration) ---
        psi = 1.0     # Weight for discharge reward
        sigma = 5.0   # Penalty for invalid discharge
        nu = 1.0      # Weight for valid charge
        beta = 1.0    # Penalty for invalid charge
        xi = 1.0      # Penalty for being idle during imbalance

        soc, demand_idx, total_idx = state

        # --- Derived variables ---
        # soc = self.idx
        delta_p = total_idx - demand_idx

        # --- Case 1: Discharge (s_BAT = 2) ---
        if self.action == 2 and delta_p < 0 and soc > 0:
            return psi * abs(delta_p) * soc

        elif self.action == 2 and (delta_p >= 0 or soc == 0):
            return -sigma

        # --- Case 2: Charge (s_BAT = 1) ---
        elif self.action == 1 and delta_p > 0:
            return nu * delta_p * (self.soc_max - soc)

        elif self.action == 1 and delta_p <= 0:
            return -beta * abs(delta_p)

        # --- Case 3: Idle (s_BAT = 0) ---
        elif self.action == 0 and abs(delta_p) > 0:
            return -xi * abs(delta_p)

        # --- Otherwise ---
        else:
            return 0.0

