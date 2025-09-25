from agents.base_agent import BaseAgent
from core.registry import register_agent


@register_agent("load")
class LoadAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(name="load", actions=[0, 1], **kwargs)
        self.env = env
        self.comfort_price = 0.5
        self.market_price = 0


    def initialize_q_table(self, env):
        states = []
        comfort_labels = ["acceptable", "expensive"]
        for demand_idx in range(len(env.demand_bins)):
            for c in comfort_labels:
                states.append((demand_idx, c))
            self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}


    def get_discretized_state(self, env, index):
        row = env.dataset.iloc[index]
        self.market_price = row["price"]
        comfort_idx = "acceptable" if self.market_price <= self.comfort_price else "expensive"
        demand_idx = np.digitize(env.demand_power, env.demand_bins) - 1
        return (demand_idx, comfort_idx)


    def calculate_reward(self, demand_idx, comfort_idx):
        if self.action == 1: # turn ON
            if comfort_idx == "acceptable":
                return 5 # reward for using energy when cheap
            return -5 # penalty if energy is expensive
        else: # turn OFF
            if comfort_idx == "expensive":
                return 5 # reward for saving money
            return -2 # small penalty if off when cheap