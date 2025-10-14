from agents.base_agent import BaseAgent
from core.registry import register_agent
import numpy as np

@register_agent("wind")
class WindAgent(BaseAgent):
    """Binary action wind generation agent (structure mirrors SolarAgent)."""

    def __init__(self, env,  name="wind", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1], state_space=state_space, **kwargs)
        self.solar_power_bins = np.linspace(0, self.env.max_value, self.env.num_power_bins)

    def update_power(self, env):
        """Compute instantaneous wind power from potential and chosen action."""
        self.power = self.potential * self.action

    def initialize_q_table(self, env):
        states = []
        for wind_idx in range(len(self.wind_power_bins)):
            for demand_idx in range(len(env.power_bins)):
                states.append((wind_idx, demand_idx))
                self.q_table = {state: {a: 0.0 for a in self.actions} for state in states}

    def calculate_reward(self, state):
        """
        Calculate the reward for the wind agent based on the current energy balance and the action taken.

        Parameters
        ----------
        state : tuple
            A tuple containing:
            - wind_idx (int or float): Discretized wind power index.
            - demand_idx (int or float): Discretized demand index.
            - renewable_idx (int or float): Discretized total renewable power index.

        Returns
        -------
        float
            Reward value computed based on the system balance and the agent's decision.

        Description
        -----------
        The reward function encourages the wind agent to contribute energy when
        total renewable generation is below demand and discourages overproduction
        when renewable generation exceeds demand.

        The weight factor `wi` scales the reward according to the agent's relative
        contribution within total renewable power.

        Reward logic:
        - If action == 1 (produce):
            - Penalize if total renewables < demand (shortage)
            - Reward if total renewables > demand (helping balance)
        - If action == 0 (do nothing):
            - Penalize if total renewables > demand (missed opportunity)
            - Reward if total renewables < demand (preventing surplus)
        """

        wind_idx, demand_idx, renewable_idx = state
        theta, beta, eta, xi = 1.0, 1.0, 1.0, 1.0

        # Avoid division by zero in weight factor
        wi = (wind_idx / renewable_idx) if renewable_idx != 0 else 1.0

        delta_abs = abs(renewable_idx - demand_idx)

        if self.action == 1:
            # Agent decides to produce
            if renewable_idx <= demand_idx:
                # Penalize shortage (the agent contributes but the system still lacks energy)
                reward = -theta * wi * delta_abs
            else:
                # Reward surplus contribution (helped exceed demand)
                reward = beta * wi * delta_abs
        else:
            # Agent decides not to produce
            if renewable_idx > demand_idx:
                # Penalize missed opportunity to help during surplus control
                reward = -eta * wi * delta_abs
            else:
                # Reward inaction when production is unnecessary or would worsen deficit
                reward = xi * wi

        return reward
