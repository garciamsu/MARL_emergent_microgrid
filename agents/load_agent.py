from agents.base_agent import BaseAgent
from core.registry import register_agent


@register_agent("load")
class LoadAgent(BaseAgent):
    """Controllable demand agent with binary action (consume vs shed/idle)."""

    def __init__(self, env,  name="load", state_space=None, **kwargs):
        super().__init__(env,  name, actions=[0, 1], state_space=state_space, **kwargs)
        self.env = env
        self.limits = kwargs.get("limits", {})
        self.comfort_threshold = self.limits.get("comfort_threshold", 1)
        self.p_load = self.limits.get("p_load", 200.0)  # Controllable load in W
        self.market_price = 1

    # Q-table se inicializa en BaseAgent.initialize_q_table según state_space

    def update_power(self, env):
        """Compute load consumption (positive demand contribution).

        Action semantics:
            0 -> Shed controllable load (reduce consumption by p_load)
            1 -> Full demand (consume all base demand from dataset)

        The potential is the base demand from dataset.
        Power is negative (consumption convention).
        """
        # Base demand from dataset (stored in env.base_demand)
        base_demand = getattr(env, 'base_demand', 0.0)
        
        # Calculate actual demand based on action
        if self.action == 1:
            # Full demand: base + controllable
            self.potential = base_demand
            self.power = -base_demand
        else:  # action == 0
            # Shed controllable load: only critical base load
            self.potential = base_demand
            controllable_demand = max(0, base_demand - self.p_load)
            self.power = -controllable_demand