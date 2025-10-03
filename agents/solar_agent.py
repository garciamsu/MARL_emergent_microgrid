import numpy as np
import math
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip

@register_agent("solar")
class SolarAgent(BaseAgent):
    def __init__(self, env,  name="solar", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1])
        self.solar_power_bins = np.linspace(0, self.env.max_value, self.env.num_power_bins)
        self.state_space = state_space        

    def initialize_q_table(self, env):
        states = [(s, t, d)
                  for s in range(len(self.solar_power_bins))
                  for t in range(len(env.power_bins))
                  for d in range(len(env.power_bins))]
        self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}

    def calculate_reward(self, solar_idx, total_idx, demand_idx):
        sigma, kappa, mu, nu, beta, xi = 15, 3, 12, 1, 5, 8
        power_gap = total_idx - demand_idx
        if self.action == 1:
            if solar_idx == 0:
                return -sigma * math.log(demand_idx + 1)
            elif solar_idx > 0 and power_gap >= 0:
                return kappa * solar_idx
            else:
                return mu * np.tanh(abs(power_gap))
        else:
            if solar_idx == 0:
                return nu
            elif solar_idx > 0 and power_gap >= 0:
                return -beta * solar_idx
            else:
                return max(-xi * abs(power_gap), -50)
