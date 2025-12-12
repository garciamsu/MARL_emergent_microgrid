from core.registry import register_reward


class RewardFn:
    """Base reward function interface.

    Implementa compute(agent, env, state_tuple) → float.
    """

    def compute(self, agent, env, state_tuple):  # pragma: no cover - interface
        raise NotImplementedError

@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):

    def __init__(self, theta=1.0, beta=1.0, nu=0.3, **kwargs):
        self.theta = theta  # escala dinámica principal
        self.beta = beta    # castigos dinámicos
        self.nu = nu        # balance suave

    def compute(self, agent, env, state_tuple):
        delta_ph, pv = state_tuple

        if delta_ph == 1 and agent.action == 1 and pv == 1:
            reward = +self.theta * abs(env.delta_ph_norm)

        elif delta_ph == 1 and agent.action == 0:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif delta_ph == -1 and agent.action == 0:
            reward = +self.theta * abs(env.delta_ph_norm)

        elif delta_ph == -1 and agent.action == 1:
            reward = -self.beta * abs(env.delta_ph_norm)

        else:
            reward = +self.nu

        return max(min(reward, 1.0), -1.0)


@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):

    def __init__(self, theta=1.0, beta=1.0, nu=0.3, **kwargs):
        self.theta = theta
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        delta_ph, pw = state_tuple

        if delta_ph == 1 and agent.action == 1 and pw == 1:
            reward = +self.theta * abs(env.delta_ph_norm)

        elif delta_ph == 1 and agent.action == 0:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif delta_ph == -1 and agent.action == 0:
            reward = +self.theta * abs(env.delta_ph_norm)

        elif delta_ph == -1 and agent.action == 1:
            reward = -self.beta * abs(env.delta_ph_norm)

        else:
            reward = +self.nu

        return max(min(reward, 1.0), -1.0)


@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):

    def __init__(self, psi=1.0, beta=1.0, nu=0.3, **kwargs):
        self.psi = psi
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        delta_ph, soc = state_tuple
        soc_norm = agent.soc

        if delta_ph == -1 and agent.action == 2 and soc > 0:
            reward = +self.psi * abs(env.delta_ph_norm) * soc_norm

        elif delta_ph == -1 and agent.action == 1:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif delta_ph == -1 and agent.action == 0:
            reward = -self.nu * abs(env.delta_ph_norm)

        elif delta_ph == 1 and agent.action == 1:
            reward = +self.psi * abs(env.delta_ph_norm) * (1 - soc_norm)

        elif delta_ph == 1 and agent.action == 2:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif delta_ph == 0 and agent.action == 0:
            reward = +self.nu

        else:
            reward = -self.nu

        return max(min(reward, 1.0), -1.0)


@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):

    def __init__(self, psi=1.0, beta=1.0, nu=0.4, **kwargs):
        self.psi = psi
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        delta_ph, soc = state_tuple
        soc_norm = env.soc

        if delta_ph == -1 and soc == 0 and agent.action == 1:
            reward = +self.psi * abs(env.delta_ph_norm)

        elif delta_ph == -1 and soc == 0 and agent.action == 0:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif (delta_ph == 1 or soc > 0) and agent.action == 1:
            reward = -self.beta * abs(env.delta_ph_norm)

        elif (delta_ph == 1 or soc > 0) and agent.action == 0:
            reward = +self.nu

        elif delta_ph == 0 and agent.action == 0:
            reward = +self.nu

        else:
            reward = -self.nu

        return max(min(reward, 1.0), -1.0)


@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):

    def __init__(self, psi=1.0, beta=1.0, nu=0.3, **kwargs):
        self.psi = psi
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        pu, cm = state_tuple

        if agent.action == 1 and pu == 1 and cm == 0:
            reward = +self.psi

        elif agent.action == 1 and pu == 1 and cm == 1:
            reward = -self.beta

        elif agent.action == 0 and pu == 0:
            reward = -self.nu

        elif agent.action == 0 and pu == 1 and cm == 0:
            reward = -self.nu

        else:
            reward = +self.nu * 0.5

        return max(min(reward, 1.0), -1.0)

