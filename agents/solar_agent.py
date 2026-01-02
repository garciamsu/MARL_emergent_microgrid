"""Solar agent implementing binary action for PV power injection."""

from agents.base_agent import BaseAgent
from core.registry import register_agent


@register_agent("solar")
class SolarAgent(BaseAgent):
    """Binary action solar generation agent.

    Action semantics:
        0 -> do not inject power
        1 -> inject full potential (current discretized potential)
    """

    def __init__(self, env,  name="solar", state_space=None, **kwargs):
        super().__init__(env, name, [0, 1], state_space=state_space, **kwargs)

    def update_power(self, env):
        """Compute instantaneous solar power from potential and chosen action."""
        self.potential = env.solar_potential
        self.power = self.potential * self.action
        self.idx = env.solar_potential_idx


    # Q-table se inicializa en BaseAgent.initialize_q_table según state_space
