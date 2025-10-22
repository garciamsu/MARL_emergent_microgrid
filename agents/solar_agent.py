import numpy as np
import math
from agents.base_agent import BaseAgent
from core.registry import register_agent
from utils.discretization import digitize_clip


@register_agent("solar")
class SolarAgent(BaseAgent):
    """Binary action solar generation agent.

    Action semantics:
        0 -> do not inject power
        1 -> inject full potential (current discretized potential)
    """

    def __init__(self, env,  name="solar", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1], state_space=state_space, **kwargs)
        self.solar_power_bins = np.linspace(0, self.env.max_value, self.env.num_power_bins)

    def update_power(self, env):
        """Compute instantaneous solar power from potential and chosen action."""
        self.power = self.potential * self.action
        self.idx = digitize_clip(self.power, env.power_bins)

    # Q-table se inicializa en BaseAgent.initialize_q_table según state_space
