from agents.base_agent import BaseAgent
from core.registry import register_agent

@register_agent("grid")
class GridAgent(BaseAgent):
    """Grid import agent with binary action (import vs idle/export placeholder)."""

    def __init__(self, env,  name="grid", battery=None, state_space=None, **kwargs):
        super().__init__(env,  name, actions=[0, 1], state_space=state_space, **kwargs)
        self.env = env
        self.battery = battery
        self.limits = kwargs.get("limits", {})
        self.p_max = self.limits.get("p_max", 1000.0)  # Maximum import power in W
        self.export_allowed = self.limits.get("export_allowed", False)

    # Q-table se inicializa en BaseAgent.initialize_q_table según state_space

    def update_power(self, env):
        """Compute power drawn from grid (only imports, no exports).

        Action semantics:
            0 -> do not import (grid disconnected)
            1 -> import to cover deficit (up to p_max)

        The grid acts as the last resort to balance the system.
        It calculates the residual deficit after renewables, load, and battery have acted.
        """
        # Calculate current system balance
        # total_power includes: renewables + battery discharge + grid (will be added)
        # demand_power includes: load consumption + battery charging
        current_deficit = env.demand_power - env.total_power
        
        # Grid can only import (cover deficit), not export
        if self.action == 1:
            # Import power to cover deficit, limited by p_max
            self.potential = max(0, current_deficit)
            self.power = min(self.potential, self.p_max)
        else:  # action == 0
            # Do not import
            self.potential = max(0, current_deficit)
            self.power = 0.0
