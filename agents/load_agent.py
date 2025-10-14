from agents.base_agent import BaseAgent
from core.registry import register_agent


@register_agent("load")
class LoadAgent(BaseAgent):
    def __init__(self, env,  name="load", state_space=None, **kwargs):
        super().__init__(env,  name, actions=[0, 1], state_space=state_space, **kwargs)
        self.env = env
        # Nuevo: extraer límites desde el config
        self.limits = kwargs.get("limits", {})
        # comfort_threshold definido en configs/default.yaml -> agents.load.limits.comfort_threshold
        self.comfort_threshold = self.limits.get("comfort_threshold", 1)
        self.market_price = 1

    def initialize_q_table(self, env):
        states = [(s, d, t)
                  for s in range(len(self.battery_soc_bins))
                  for d in range(len(env.power_bins))
                  for t in range(len(env.power_bins))]
        self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}

    def update_power(self, env):
        self.power = self.potential * self.action

    def calculate_reward(self, state):
        """
        Calculates the reward Ri for the LoadAgent based on system state and action.

        Parameters
        ----------
        state : tuple
            (soc_idx, renewable_idx, demand_idx, market_cost, comfort_cost)
            where:
                soc_idx       : float, State of Charge (SOC)
                renewable_idx : float, renewable power available (P_H)
                demand_idx    : float, demand power (P_L)
                market_cost   : float, market cost (C_M)
                comfort_cost  : float, comfort cost (C_confort)

        Returns
        -------
        float
            Reward value according to:
                Ri = +σ·C_M                        if S_D=1 and (SOC>0 or P_H>P_L)
                Ri = -ψ/C_M                        if S_D=1 and C_confort < C_M
                Ri = -ν·SOC·P_H                    if S_D=0 and (SOC>0 or P_H>P_L)
                Ri = β                             Otherwise
        """
        soc_idx, demand_idx, renewable_idx = state
        market_cost = self.env.price
        # Reward parameters
        sigma = 1.0
        psi = 1.0
        nu = 1.0
        beta = -0.1

        # Conditional reward calculation
        if self.action == 1 and (soc_idx > 0 or renewable_idx > demand_idx):
            reward = sigma * market_cost
        elif self.action == 1 and self.comfort_threshold < market_cost:
            reward = -psi / market_cost
        elif self.action == 0 and (soc_idx > 0 or renewable_idx > demand_idx):
            reward = -nu * soc_idx * renewable_idx
        else:
            reward = beta

        return reward