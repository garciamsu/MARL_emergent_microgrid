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

        renewable_idx = env.renewable_power_idx
        demand_idx = env.demand_power_idx
        solar_idx = agent.idx
        
        delta_abs = max(abs(renewable_idx - demand_idx), 1)
        if agent.action == 1:
            if renewable_idx < demand_idx:
                return -self.theta * delta_abs
            else:
                return self.beta * delta_abs
        else:
            if renewable_idx > demand_idx:
                return self.eta * delta_abs
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

            
        demand_idx = env.demand_power_idx
        wind_idx = agent.idx
        
        delta_abs = max(abs(renewable_idx - demand_idx), 1)
        if agent.action == 1:
            if renewable_idx < demand_idx:
                return -self.theta * delta_abs
            else:
                return self.beta * delta_abs
        else:
            if renewable_idx > demand_idx:
                return self.eta * delta_abs
            else:
                return -self.xi

@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    """Replica la lógica de BatteryAgent.calculate_reward."""

    def __init__(self, psi=1.0, sigma=1.0, nu=1.0, beta=1.0, xi=1.0, **kwargs):
        self.psi = psi
        self.sigma = sigma
        self.nu = nu
        self.beta = beta
        self.xi = xi

    def compute(self, agent, env, state_tuple):

        soc, demand_idx, total_idx = state_tuple
        delta_p = total_idx - demand_idx

        print(f"DEBUG: action={agent.action}, soc={soc}, demand_idx={demand_idx}, total_idx={total_idx}, delta_p={delta_p}")
        agent.soc_max = 4
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
        # --- 1. Obtener Variables de Estado y Entorno ---
        soc_idx = state_tuple[0]
        demand_idx = env.demand_power_idx
        # Corrección: Usar la generación renovable para el déficit
        renewable_idx = env.renewable_power_idx
        self.C_M = env.price # Precio del mercado

        # --- 2. Calcular Déficit/Excedente Interno (Corrección) ---
        # (Excedente renovable antes de la acción de la red)
        delta_P = renewable_idx - demand_idx

        # --- 3. Lógica de Recompensa (Corregida) ---

        # CASO 1: Red ACTÚA (1) + Microred NECESITA energía
        # (Déficit renovable Y Batería vacía)
        if agent.action == 1 and delta_P <= 0 and soc_idx == 0:
            # PREMIO: por cumplir su deber.
            # (Inversamente proporcional al precio)
            reward = self.psi / self.C_M if self.C_M > 0 else self.psi

        # CASO 2: Red ACTÚA (1) + Microred NO NECESITA energía
        # (Excedente renovable O Batería con carga)
        elif agent.action == 1 and (delta_P > 0 or soc_idx > 0):
            # CASTIGO: por importar innecesariamente.
            # (Proporcional al precio)
            reward = -self.sigma * self.C_M

        # CASO 3: Red NO ACTÚA (0) + Microred NECESITA energía
        # (Déficit renovable Y Batería vacía)
        elif agent.action == 0 and delta_P <= 0 and soc_idx == 0:
            # CASTIGO: por fallar en su deber.
            # (Tu Opción B: Proporcional al déficit)
            reward = -self.nu * max(abs(delta_P), 1)

        # CASO 4 ("else"): Red NO ACTÚA (0) + Microred NO NECESITA energía
        # (Excedente renovable O Batería con carga)
        else:
            # PREMIO: por inacción correcta.
            # (Tu fórmula: Proporcional a la energía interna)
            reward = self.xi * max(delta_P, 1) * max(soc_idx, 1)

        return reward

@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Replica la lógica de LoadAgent.calculate_reward."""

    def __init__(self, sigma=1.0, psi=1.0, nu=1.0, beta=-0.1, **kwargs):
        self.sigma = sigma
        self.psi = psi
        self.nu = nu
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        soc_idx, demand_idx, renewable_idx, price = state_tuple

        # 1. PREMIO por usar energía interna/excedente
        if agent.action == 1 and (soc_idx > 0 or renewable_idx > demand_idx):
            # Tu Lógica 1 (corregida):
            reward = self.sigma * max((renewable_idx - demand_idx), 1) * max(soc_idx, 1)

        # 2. CASTIGO por comprar caro
        # Compare market price against the agent's comfort threshold (agent owns this parameter).
        elif agent.action == 1 and price > getattr(agent, 'comfort_threshold', 1):
            # Penalize buying when the market price exceeds the agent's comfort threshold
            reward = -self.psi * price

        # 3. CASTIGO por desperdiciar energía interna/excedente
        elif agent.action == 0 and (soc_idx > 0 or renewable_idx > demand_idx):
            # Tu corrección (simétrica a la Lógica 1):
            reward = -self.nu * max((renewable_idx - demand_idx), 1) * max(soc_idx, 1)

        # 4. RECOMPENSA NEUTRAL (Apagado correcto O Comprar barato)
        else:
            reward = self.beta

        print(f"debug: action={agent.action}, soc_idx={soc_idx}, demand_idx={demand_idx}, renewable_idx={renewable_idx}, price={price}, reward={reward}")

        return reward