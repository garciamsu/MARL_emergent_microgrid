from core.registry import register_reward


class RewardFn:
    def compute(self, agent, env, state_tuple):
        raise NotImplementedError


@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):
    def __init__(self, theta=3, beta=3):
        self.theta = theta
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        return self.theta

@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    def __init__(self, theta=3, beta=3):
        self.theta = theta
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        return self.theta

@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma

@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma
    
@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    def __init__(self, sigma=10, mu=5):
        self.sigma = sigma
        self.mu = mu

    def compute(self, agent, env, state_tuple):
        return self.sigma