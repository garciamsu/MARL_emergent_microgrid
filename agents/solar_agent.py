import numpy as np
import math
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip

@register_agent("solar")
class SolarAgent(BaseAgent):
    def __init__(self, env,  name="solar", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1], state_space=state_space, **kwargs)
        self.solar_power_bins = np.linspace(0, self.env.max_value, self.env.num_power_bins)

    def update_power(self, env):
        self.power = self.potential * self.action

    def initialize_q_table(self, env):
        states = [(s, t, d)
                  for s in range(len(self.solar_power_bins))
                  for t in range(len(env.power_bins))
                  for d in range(len(env.power_bins))]
        self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}

    def calculate_reward(self, state):
        """
        Calculate the reward for the solar agent based on the energy balance and action taken.

        Parameters
        ----------
        state : tuple
            A tuple containing:
            - solar_idx (int or float): Discretized solar power index.
            - demand_idx (int or float): Discretized demand index.
            - renewable_idx (int or float): Discretized total renewable power index.

        Returns
        -------
        float
            Reward value computed according to the energy balance and agent's action.

        Description
        -----------
        The reward function encourages the solar agent to contribute energy
        when the system is under-supplied (renewables < demand) and discourages
        overproduction when the system is already meeting or exceeding demand.
        The weight factor `wi` adjusts the impact of solar generation relative
        to total renewable contribution.

        Reward logic:
        - If action == 1 (produce):
            - If renewable < demand: penalize shortage (negative reward)
            - Else: reward proportional to surplus (positive reward)
        - If action == 0 (do nothing):
            - If renewable > demand: penalize missed opportunity (negative reward)
            - Else: reward inaction that prevents overproduction
        """

        solar_idx, demand_idx, renewable_idx = state
        theta, beta, eta, xi = 1.0, 1.0, 1.0, 1.0

        # Avoid division by zero in the weight factor
        wi = (solar_idx / renewable_idx) if renewable_idx != 0 else 1.0

        delta_abs = abs(renewable_idx - demand_idx)

        if self.action == 1:
            # Agent decides to produce
            if renewable_idx <= demand_idx:
                # Penalize shortage: renewables not enough to meet demand
                reward = -theta * wi * delta_abs
            else:
                # Reward surplus (renewables exceed demand)
                reward = beta * wi * delta_abs
        else:
            # Agent decides not to produce
            if renewable_idx > demand_idx:
                # Penalize missed opportunity (underutilized renewables)
                reward = -eta * wi * delta_abs
            else:
                # Reward correct inaction during shortage
                reward = xi * wi

        return reward
