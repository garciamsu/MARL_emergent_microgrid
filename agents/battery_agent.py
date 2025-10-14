import numpy as np
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip


@register_agent("battery")
class BatteryAgent(BaseAgent):
    def __init__(self, env,  name="battery",capacity_ah=3, num_battery_soc_bins=5, state_space=None, **kwargs):
        super().__init__(env, name, [0, 1, 2], state_space=state_space, **kwargs)
        self.capacity_ah = capacity_ah
        self.soc = 0.5
        self.soc_max = 1
        self.battery_soc_bins = np.linspace(0, self.soc_max, num_battery_soc_bins)

    def update_power(self, env):
        self.power = self.potential * self.action

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

        print(state)
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

