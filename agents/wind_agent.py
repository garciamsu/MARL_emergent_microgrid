from agents.base_agent import BaseAgent
from core.registry import register_agent
import numpy as np
from utils.discretization import digitize_clip

@register_agent("wind")
class WindAgent(BaseAgent):
    """Binary action wind generation agent (structure mirrors SolarAgent)."""

    def __init__(self, env,  name="wind", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1], state_space=state_space, **kwargs)

    def update_power(self, env):
        """Compute instantaneous wind power from potential and chosen action."""
        self.potential = env.wind_potential
        self.power = self.potential * self.action
        self.idx = env.wind_potential_idx

    # Q-table se inicializa en BaseAgent.initialize_q_table según state_space
