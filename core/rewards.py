from core.registry import register_reward


class RewardFn:
    """Base reward function interface.

    Subclasses should implement :meth:`compute` returning a scalar reward given
    (agent, env, state_tuple). The current reward functions are placeholders;
    they return constant values and should be replaced with domain logic.
    """

    def compute(self, agent, env, state_tuple):  # pragma: no cover - interface
        raise NotImplementedError


@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):
    """Placeholder solar reward returning a constant value.

    Args:
        theta (float): Constant reward baseline.
        beta (float): Unused parameter placeholder for future shaping.
    """

    def __init__(self, theta=3, beta=3):
        self.theta = theta
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        return self.theta

@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Placeholder wind reward returning a constant value."""

    def __init__(self, theta=3, beta=3):
        self.theta = theta
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        return self.theta

@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    """Placeholder battery reward returning a constant value."""

    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma

@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    """Placeholder grid reward returning a constant value."""

    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma
    
@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Placeholder load reward returning a constant value."""

    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma