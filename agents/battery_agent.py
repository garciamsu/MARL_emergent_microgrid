import numpy as np
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip


@register_agent("battery")
class BatteryAgent(BaseAgent):
    def __init__(self, env, capacity_ah=3, num_battery_soc_bins=5, **kwargs):
        super().__init__("battery", [0, 1, 2])
        self.capacity_ah = capacity_ah
        self.soc = 0.5
        self.battery_soc_bins = np.linspace(0, 1, num_battery_soc_bins)

    def get_discretized_state(self, env, index):
        self.idx = digitize_clip(self.soc, self.battery_soc_bins)
        renewable_idx = env.renewable_potential_idx
        total_idx = digitize_clip(env.total_power, env.power_bins)
        demand_idx = env.demand_power_idx
        return (self.idx, renewable_idx, total_idx, demand_idx)

    def initialize_q_table(self, env):
        states = [
            (soc, ren, tot, dem)
            for soc in range(len(self.battery_soc_bins))
            for ren in range(len(env.power_bins))
            for tot in range(len(env.power_bins))
            for dem in range(len(env.power_bins))
        ]
        self.q_table = {s: {a: 0.0 for a in self.actions} for s in states}

    def calculate_reward(self, soc_idx, renewable_idx, total_idx, demand_idx):
        if self.action == 0:
            return -1
        elif self.action == 1:
            return 1 if renewable_idx > demand_idx else -1
        elif self.action == 2:
            return 1 if soc_idx > 0 else -5
