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
        
        Safety: Prevents negative demand by ensuring p_load <= base_demand.
        """
        # Base demand from dataset (stored in env.base_demand)
        base_demand = getattr(env, 'base_demand', 0.0)
        
        # Calculate actual demand based on action
        if self.action == 1:
            # Action 1: Full demand (load ON)
            self.potential = base_demand
            self.power = -base_demand
        else:  # action == 0
            # Action 0: Shed controllable load (load OFF)
            # Safety check: prevent negative demand
            if base_demand < self.p_load:
                # If base_demand < p_load, only shed what's available
                # This prevents physical inconsistency
                actual_shed = base_demand
                controllable_demand = 0
            else:
                # Normal operation: shed p_load from base_demand
                actual_shed = self.p_load
                controllable_demand = base_demand - self.p_load
            
            self.potential = base_demand
            self.power = -controllable_demand