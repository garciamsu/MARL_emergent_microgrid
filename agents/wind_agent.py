from agents.base_agent import BaseAgent
from core.registry import register_agent
import numpy as np

@register_agent("wind")
class WindAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(name="wind", actions=[0, 1], **kwargs)
        self.env = env
        self.wind_power_bins = np.linspace(0, env.max_value, env.num_power_bins)
        self.potential = 0.0
        self.idx = 0

def initialize_q_table(self, env):
    states = []
    for wind_idx in range(len(self.wind_power_bins)):
        for demand_idx in range(len(env.demand_bins)):
            states.append((wind_idx, demand_idx))
            self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}


def get_discretized_state(self, env, index):
    row = env.dataset.iloc[index]
    self.potential = row["wind_power"]
    self.idx = np.digitize(self.potential, self.wind_power_bins) - 1
    demand_idx = np.digitize(env.demand_power, env.demand_bins) - 1
    return (self.idx, demand_idx)

def calculate_reward(self, wind_idx, demand_idx):
    if self.action == 1:
        if wind_idx == 0:
            return -10 # trying to produce with no wind
        return 5 # reward for producing with wind available
    else:
        if wind_idx > 0:
            return -2 # penalty for idling when wind available
        return 1 # neutral if no wind