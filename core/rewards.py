from core.registry import register_reward


class RewardFn:
    """Base reward function interface.

    Implementa compute(agent, env, state_tuple) → float.
    """

    def compute(self, agent, env, state_tuple):  # pragma: no cover - interface
        raise NotImplementedError


@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):
    """Replica la lógica de SolarAgent.calculate_reward."""

    def __init__(self, theta=1.0, beta=1.0, eta=1.0, xi=1.0, **kwargs):
        self.theta = theta
        self.beta = beta
        self.eta = eta
        self.xi = xi

    def compute(self, agent, env, state_tuple):
        solar_idx, demand_idx, renewable_idx = state_tuple
        # wi = (solar_idx / renewable_idx) if renewable_idx != 0 else 1.0
        wi = 1
        delta_abs = abs(renewable_idx - demand_idx)
        if agent.action == 1:
            if renewable_idx < demand_idx:
                return -self.theta * wi * delta_abs
            else:
                return self.beta * wi * delta_abs
        else:
            if renewable_idx > demand_idx:
                return self.eta * wi * delta_abs
            else:
                return -self.xi


@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Replica la lógica de WindAgent.calculate_reward."""

    def __init__(self, theta=1.0, beta=1.0, eta=1.0, xi=1.0, **kwargs):
        self.theta = theta
        self.beta = beta
        self.eta = eta
        self.xi = xi

    def compute(self, agent, env, state_tuple):
        wind_idx, demand_idx, renewable_idx = state_tuple
        # wi = (wind_idx / renewable_idx) if renewable_idx != 0 else 1.0
        wi = 1
        delta_abs = abs(renewable_idx - demand_idx)
        if agent.action == 1:
            if renewable_idx < demand_idx:
                return -self.theta * wi * delta_abs
            else:
                return self.beta * wi * delta_abs
        else:
            if renewable_idx > demand_idx:
                return self.eta * wi * delta_abs
            else:
                return -self.xi

@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    """Replica la lógica de BatteryAgent.calculate_reward."""

    def __init__(self, psi=1.0, sigma=5.0, nu=1.0, beta=1.0, xi=1.0, **kwargs):
        self.psi = psi
        self.sigma = sigma
        self.nu = nu
        self.beta = beta
        self.xi = xi

    def compute(self, agent, env, state_tuple):
        
        soc, demand_idx, total_idx = state_tuple
        delta_p = total_idx - demand_idx
        if agent.action == 2 and delta_p < 0 and soc > 0:
            return self.psi * abs(delta_p) * soc
        elif agent.action == 2 and (delta_p >= 0 or soc == 0):
            return -self.sigma
        elif agent.action == 1 and delta_p > 0:
            return self.nu * delta_p * (agent.soc_max - soc)
        elif agent.action == 1 and delta_p <= 0:
            return -self.beta * abs(delta_p)
        elif agent.action == 0 and abs(delta_p) > 0:
            return -self.xi * abs(delta_p)
        else:
            return 0.0


@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    """Replica la lógica de GridAgent.calculate_reward."""

    def __init__(self, psi=1.0, sigma=1.0, nu=1.0, xi=1.0, C_M=1.0, **kwargs):
        self.psi = psi
        self.sigma = sigma
        self.nu = nu
        self.xi = xi
        self.C_M = C_M

    def compute(self, agent, env, state_tuple):
        soc_idx, demand_idx, total_idx = state_tuple
        delta_P = total_idx - demand_idx
        if agent.action == 1 and delta_P < 0 and soc_idx == 0:
            return self.psi / self.C_M
        elif agent.action == 1 and (delta_P >= 0 or soc_idx > 0):
            return -self.sigma * self.C_M
        elif agent.action == 0 and delta_P < 0 and soc_idx == 0:
            return -self.nu * self.C_M
        else:
            return -self.xi


@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Replica la lógica de LoadAgent.calculate_reward."""

    def __init__(self, sigma=1.0, psi=1.0, nu=1.0, beta=-0.1, **kwargs):
        self.sigma = sigma
        self.psi = psi
        self.nu = nu
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        soc_idx, demand_idx, renewable_idx = state_tuple
        market_cost = env.price
        if agent.action == 1 and (soc_idx > 0 or renewable_idx > demand_idx):
            return self.sigma * market_cost
        elif agent.action == 1 and getattr(agent, 'comfort_threshold', 0) < market_cost:
            return -self.psi / market_cost if market_cost else -self.psi
        elif agent.action == 0 and (soc_idx > 0 or renewable_idx > demand_idx):
            return -self.nu * soc_idx * renewable_idx
        else:
            return self.beta