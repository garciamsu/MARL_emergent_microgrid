from core.registry import register_reward


class RewardFn:
    """Base reward function interface.

    Implementa compute(agent, env, state_tuple) → float.
    """

    def compute(self, agent, env, state_tuple):  # pragma: no cover - interface
        raise NotImplementedError

@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):
    """Reward function for solar agent.
    
    Note: Currently uses fixed rewards based on state-action pairs.
    Parameters theta, beta, nu are reserved for future dynamic scaling.
    """

    def __init__(self, theta=1.0, beta=1.0, nu=1.0, **kwargs):
        # Reserved for future use - currently rewards are fixed
        self.theta = theta
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        delta_ph_idx, pv_idx = state_tuple
        reward = 0.0

        if delta_ph_idx > 0 and agent.action == 1:
            reward = 1 * (1 if pv_idx > 0 else -1)
        elif delta_ph_idx > 0 and agent.action == 0:
            reward = -1 
        elif delta_ph_idx < 0 and agent.action == 0:
            reward = 1
        elif delta_ph_idx < 0 and agent.action > 0:
            reward = -1
        else:
            reward = 1

        return reward




@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Reward function for wind agent.
    
    Note: Currently uses fixed rewards based on state-action pairs.
    Parameters theta, beta, nu are reserved for future dynamic scaling.
    """

    def __init__(self, theta=1.0, beta=1.0, nu=1.0, **kwargs):
        # Reserved for future use - currently rewards are fixed
        self.theta = theta
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        delta_ph_idx, pw_idx = state_tuple
        reward = 0.0

        if delta_ph_idx > 0 and agent.action == 1:
            reward = 1 * (1 if pw_idx > 0 else -1)
        elif delta_ph_idx > 0 and agent.action == 0:
            reward = -1
        elif delta_ph_idx < 0 and agent.action == 0:
            reward = 1
        elif delta_ph_idx < 0 and agent.action > 0:
            reward = -1
        else:
            reward = 1

        return reward


@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    """Reward function for battery agent.
    
    Uses REAL BALANCE (renewable_power - demand) instead of stigmergic delta_ph.
    This ensures the battery is rewarded/penalized based on actual power availability,
    not on remaining potential after stigmergic consumption.
    
    Parameters:
        psi: Scale factor for correct behavior rewards (discharge on deficit, charge on surplus)
        beta: Scale factor for incorrect behavior penalties
        nu: Scale factor for neutral/idle behavior
    """

    def __init__(self, psi=1.0, beta=1.0, nu=0.3, **kwargs):
        self.psi = psi
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        # Use REAL BALANCE instead of stigmergic delta_ph from state
        # This ensures battery is evaluated on actual power availability
        # Note: Use _reward_real_balance_* saved before loading next timestep data
        real_balance_idx = getattr(env, '_reward_real_balance_idx', getattr(env, 'real_balance_idx', 0))
        real_balance_norm = getattr(env, '_reward_real_balance_norm', getattr(env, 'real_balance_norm', 0.0))
        
        _, soc = state_tuple  # Still use SOC from state tuple
        soc_norm = agent.soc
        reward = 0.0

        # Deficit scenario: discharge is correct, charge is wrong
        if real_balance_idx < 0 and agent.action == 2 and soc > 0:
            reward = +self.psi * abs(real_balance_norm) * soc_norm

        elif real_balance_idx < 0 and agent.action == 1:
            reward = -self.beta * 1

        elif real_balance_idx < 0 and agent.action == 0:
            reward = -self.nu * 1

        # Surplus scenario: charge is correct, discharge is wrong
        elif real_balance_idx > 0 and agent.action == 1:
            reward = +self.psi * 1 * (1 - soc_norm)

        elif real_balance_idx > 0 and agent.action == 2:
            reward = -self.beta * 1

        # Balanced scenario: idle is correct
        elif real_balance_idx == 0 and agent.action == 0:
            reward = +self.nu

        else:
            reward = -self.nu

        return reward


@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    """Reward function for grid agent.
    
    Uses REAL BALANCE (renewable_power - demand) instead of stigmergic delta_ph.
    This ensures grid import decisions are evaluated based on actual power availability,
    not on remaining potential after stigmergic consumption.
    
    Parameters:
        psi: Scale factor for correct import behavior (import only when needed)
        beta: Scale factor for incorrect import behavior penalties
        nu: Scale factor for correct non-import behavior
    """

    def __init__(self, psi=1.0, beta=1.0, nu=1.0, **kwargs):
        self.psi = psi
        self.beta = beta
        self.nu = nu

    def compute(self, agent, env, state_tuple):
        # Use REAL BALANCE instead of stigmergic delta_ph from state
        # Grid should import only when there's actual deficit after all renewables
        # Note: Use _reward_real_balance_* saved before loading next timestep data
        real_balance_idx = getattr(env, '_reward_real_balance_idx', getattr(env, 'real_balance_idx', 0))
        
        _, soc_idx = state_tuple  # Still use SOC from state tuple
        soc = env.soc_state
        reward = 0.0

        # True deficit AND no battery charge: import is correct
        if real_balance_idx < 0 and soc == 0 and agent.action == 1:
            reward = self.psi * 1
        # True deficit AND no battery charge: not importing is wrong
        elif real_balance_idx < 0 and soc == 0 and agent.action == 0:
            reward = -self.beta * 1
        # Surplus OR battery has charge: importing is wrong
        elif (real_balance_idx > 0 or soc > 0) and agent.action == 1:
            reward = -self.beta * 1
        # Surplus OR battery has charge: not importing is correct
        elif (real_balance_idx > 0 or soc > 0) and agent.action == 0:
            reward = self.nu * 1
        else:
            reward = -0.25
        
        return reward


@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Reward function for controllable load agent.
    
    Parameters:
        psi: Scale factor for correct behavior (ON when cheap/available, OFF when expensive)
        beta: Scale factor for incorrect behavior penalties
        nu: Scale factor for neutral/edge case rewards
    """

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

        return reward

