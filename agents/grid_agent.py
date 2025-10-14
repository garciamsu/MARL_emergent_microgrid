from agents.base_agent import BaseAgent
from core.registry import register_agent

@register_agent("grid")
class GridAgent(BaseAgent):
    def __init__(self, env,  name="grid", battery=None, state_space=None, **kwargs):
        super().__init__(env,  name, actions=[0, 1], state_space=state_space, **kwargs)
        self.env = env
        self.battery = battery

    def initialize_q_table(self, env):
        states = []
        for soc_idx in range(5):
            for demand_idx in range(len(env.power_bins)):
                for total_idx in range(len(env.power_bins)):
                    states.append((soc_idx, demand_idx, total_idx))
        self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}

    def update_power(self, env):
        self.power = self.potential * self.action

    def calculate_reward(self, state):
        """
        Calculates the reward Ri based on the system state and the agent's action.

        Parameters
        ----------
        state : tuple
            (soc_idx, total_idx, demand_idx)
            where:
                soc_idx     : int or float, state of charge (SOC)
                total_idx   : int or float, total generated power
                demand_idx  : int or float, demand power

        Returns
        -------
        float
            Reward value according to:
                Ri = +ψ/CM      if sU=1 and ΔP<0 and SOC=0
                Ri = -σ·CM      if sU=1 and (ΔP≥0 or SOC>0)
                Ri = -ν·CM      if sU=0 and ΔP<0 and SOC=0
                Ri = -ξ         Otherwise
        """
        soc_idx, demand_idx, total_idx = state
        delta_P = total_idx - demand_idx
        print(state)
        # Reward parameters
        psi = 1.0
        sigma = 1.0
        nu = 1.0
        xi = 1.0
        C_M = 1.0

        if self.action == 1 and delta_P < 0 and soc_idx == 0:
            reward = psi / C_M
        elif self.action == 1 and (delta_P >= 0 or soc_idx > 0):
            reward = -sigma * C_M
        elif self.action == 0 and delta_P < 0 and soc_idx == 0:
            reward = -nu * C_M
        else:
            reward = -xi

        return reward
