from agents.base_agent import BaseAgent
from core.registry import register_agent

@register_agent("grid")
class GridAgent(BaseAgent):
    def __init__(self, env, battery=None, **kwargs):
        super().__init__(name="grid", actions=[0, 1], **kwargs)
        self.env = env
        self.battery = battery

    def initialize_q_table(self, env):
        states = []
        for soc_idx in range(5):
            for demand_idx in range(len(env.demand_bins)):
                states.append((soc_idx, demand_idx))
        self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}


    def get_discretized_state(self, env, index):
        demand_idx = np.digitize(env.demand_power, env.demand_bins) - 1
        soc_idx = self.battery.idx if self.battery else 0
        return (soc_idx, demand_idx)


    def calculate_reward(self, soc_idx, demand_idx):
        if self.action == 1:
            if soc_idx == 0 and demand_idx > 0:
                return 10 # supplying when battery empty and demand > 0
            return -2 # penalize unnecessary grid use
        else:
            if demand_idx > 0 and soc_idx == 0:
                return -10 # penalty for not helping when necessary
            return 2 # reward for saving costs