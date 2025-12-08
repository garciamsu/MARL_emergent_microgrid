from core.registry import register_reward


class RewardFn:
    """Base reward function interface.

    Implementa compute(agent, env, state_tuple) → float.
    """

    def compute(self, agent, env, state_tuple):  # pragma: no cover - interface
        raise NotImplementedError


@register_reward("DefaultSolarReward")
class DefaultSolarReward(RewardFn):
    """Recompensa simple y estable para agentes solares.
    Lógica plana:
    - Sin sol: Recompensa neutral.
    - Déficit + Producir: Recompensa positiva (ayuda).
    - Déficit + Nada: Penalización (oportunidad perdida).
    - Excedente + Producir: Penalización (daña la red).
    - Excedente + Nada: Recompensa positiva (ayuda/curtailment).
    """

    def __init__(self, theta: float = 1.0, beta: float = 1.0,
                 nu: float = 1.0, xi: float = 1.0,
                 **kwargs) -> None:
        self.theta = theta # Premio por producir en excedente
        self.beta = beta   # Castigo por producir en déficit
        self.nu = nu       # Castigo por no producir en excedente
        self.xi = xi       # Premio por no producir en déficit

    def compute(self, agent, env, state_tuple) -> float:

        # --- 1. Variables de Estado ---
        delta_p = env.renewable_potential_idx - env.demand_power_idx # <0 Deficit, >0 Excedente
        
        # --- 2. Normalización Dinámica ---
        max_p = max(env.num_power_bins - 1, 1)

        imbalance_norm = abs(delta_p) / max_p
        solar_norm = state_tuple[0] / max_p

        # --- 3. Lógica Plana (Flat Logic) ---
        print(f"DEBUG: delta_p={delta_p}, max_p={max_p}, env.renewable_potential_idx={env.renewable_potential_idx}, env.demand_power_idx={env.demand_power_idx}, agent.action={agent.action}, solar_norm={solar_norm:.3f}, imbalance_norm={imbalance_norm:.3f}")

        # CASO A: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = PRODUCIR (1)
        if delta_p >= 0 and agent.action == 1:
            reward = self.theta * imbalance_norm

        # CASO B: DÉFICIT (Falta energía) y ACCIÓN = IDLE (0)
        elif delta_p < 0 and agent.action == 0:
            reward = self.xi * imbalance_norm

        # CASO C: DÉFICIT (Falta energía) y ACCIÓN = PRODUCIR (1)
        elif delta_p < 0 and agent.action == 1:
            reward = -self.beta * imbalance_norm

        # CASO D: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = IDLE (0)
        else:
            reward = -self.nu * imbalance_norm

        # --- 4. Clipping final ---
        return max(min(reward, 1.0), -1.0)

@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Recompensa simple y estable para agentes eolicos.
    Lógica plana:
    - Sin sol: Recompensa neutral.
    - Déficit + Producir: Recompensa positiva (ayuda).
    - Déficit + Nada: Penalización (oportunidad perdida).
    - Excedente + Producir: Penalización (daña la red).
    - Excedente + Nada: Recompensa positiva (ayuda/curtailment).
    """

    def __init__(self, theta: float = 1.0, beta: float = 1.0,
                 nu: float = 1.0, xi: float = 1.0,
                 **kwargs) -> None:
        self.theta = theta # Premio por producir en excedente
        self.beta = beta   # Castigo por producir en déficit
        self.nu = nu       # Castigo por no producir en excedente
        self.xi = xi       # Premio por no producir en déficit

    def compute(self, agent, env, state_tuple) -> float:

        # --- 1. Variables de Estado ---
        delta_p = env.renewable_potential_idx - env.demand_power_idx # <0 Deficit, >0 Excedente
        
        # --- 2. Normalización Dinámica ---
        max_p = max(env.num_power_bins - 1, 1)

        imbalance_norm = abs(delta_p) / max_p
        wind_norm = state_tuple[0] / max_p

        # --- 3. Lógica Plana (Flat Logic) ---

        # CASO A: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = PRODUCIR (1)
        if delta_p >= 0 and agent.action == 1:
            reward = self.theta * imbalance_norm

        # CASO B: DÉFICIT (Falta energía) y ACCIÓN = IDLE (0)
        elif delta_p < 0 and agent.action == 0:
            reward = self.xi * imbalance_norm

        # CASO C: DÉFICIT (Falta energía) y ACCIÓN = PRODUCIR (1)
        elif delta_p < 0 and agent.action == 1:
            reward = -self.beta * imbalance_norm

        # CASO D: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = IDLE (0)
        else:
            reward = -self.nu * imbalance_norm

        # --- 4. Clipping final ---
        return max(min(reward, 1.0), -1.0)

@register_reward("DefaultBatteryReward")
class DefaultBatteryReward(RewardFn):
    """Replica la lógica de BatteryAgent.calculate_reward."""

    def __init__(self, psi=1.0, sigma=1.0, nu=1.0, beta=1.0, xi=1.0,  mu=1.0, **kwargs):
        self.psi = psi
        self.sigma = sigma
        self.nu = nu
        self.beta = beta
        self.xi = xi
        self.mu = mu

    def compute(self, agent, env, state_tuple):

        # --- 1. Obtener Variables de Estado y Entorno ---
        # state_tuple[0] es el ÍNDICE discreto del SOC [0, num_soc_bins-1]
        soc_idx = state_tuple[0]
        demand_idx = state_tuple[1]
        # state_tuple[2] (renewable_idx) no se usa directamente

        # Corrección: Definir el desequilibrio ANTES de la acción de la batería
        renewable_idx = env.renewable_power_idx
        delta_p = renewable_idx - demand_idx

        # --- 2. Normalización Dinámica ---
        max_p = max(env.num_power_bins - 1, 1)
        max_soc_idx = max(env.num_soc_bins - 1, 1)

        imbalance_norm = delta_p / max_p
        # Normalizar el índice de SOC a [0, 1]
        soc_norm = soc_idx / max_soc_idx

        # --- 3. Lógica de Recompensa (Corregida) ---

        # CASO 1: Descarga Correcta (PREMIO)
        # (Acción=Descargar, Hay Déficit, Batería tiene carga)
        if agent.action == 2 and imbalance_norm < 0 and soc_norm > 0:
            # Premio por suplir la demanda
            reward = self.psi * abs(imbalance_norm) * soc_norm

        # CASO 2: Descarga Incorrecta (CASTIGO)
        # (Acción=Descargar, PERO hay Excedente O Batería vacía)
        elif agent.action == 2 and (imbalance_norm >= 0 or soc_norm == 0):
            # Castigo fijo por acción ilógica o innecesaria
            reward = -self.sigma * imbalance_norm

        # CASO 3: Carga Correcta (PREMIO)
        # (Acción=Cargar, Hay Excedente)
        elif agent.action == 1 and imbalance_norm > 0:
            # Premio por almacenar excedente (escala con espacio vacío)
            reward = self.nu * imbalance_norm * (agent.soc_max - soc_norm)

        # CASO 4: Carga Incorrecta (CASTIGO)
        # (Acción=Cargar, PERO hay Déficit)
        elif agent.action == 1 and imbalance_norm <= 0:
            # Castigo por empeorar el déficit
            reward = -self.beta * abs(imbalance_norm)

        # CASO 5: Inacción Incorrecta (CASTIGO)
        # (Acción=Inactivo, PERO hay Desequilibrio)
        elif agent.action == 0 and soc_norm > 0:
            # Castigo por no actuar (cargar o descargar)
            reward = -self.xi * abs(imbalance_norm)

        # CASO 6 ("else"): Inacción Correcta (PREMIO)
        # (Acción=Inactivo, Hay Equilibrio perfecto)
        else:
            # Tu corrección: Premio por inacción correcta
            reward = self.mu

        # --- 4. Clipping final ---
        return max(min(reward, 1.0), -1.0)

@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    """Grid como 'último recurso':
       - Premia importar cuando hay déficit y la batería está baja.
       - Castiga fuerte quedarse idle con déficit y batería baja.
       - Penaliza importar sin necesidad (sin déficit / con excedente).
       - Premia no usar el Grid cuando no hay déficit.
    """

    def __init__(self, psi=1.0, sigma=1.0, nu=1.0, xi=1.0, C_M=1.0, **kwargs):
        # psi   → factor de PREMIO al importar cuando realmente hace falta (déficit + batería baja)
        # sigma → factor de CASTIGO por importar sin necesidad (sin déficit / excedente)
        # nu    → factor de CASTIGO por quedarse idle con déficit crítico
        # xi    → factor de PREMIO por no usar el Grid cuando no hay déficit
        # C_M   → mantenido por compatibilidad (no se usa directamente en la escala del reward)
        self.psi = psi
        self.sigma = sigma
        self.nu = nu
        self.xi = xi
        self.C_M = C_M

    def compute(self, agent, env, state_tuple):
        # --- 1. Variables de estado ---
        soc_idx = state_tuple[0]

        # delta_p < 0 → déficit, delta_p > 0 → excedente
        delta_p = env.renewable_power_idx - env.demand_power_idx

        max_p = max(env.num_power_bins - 1, 1)
        max_soc_idx = max(env.num_soc_bins - 1, 1)

        # Normalización (en condiciones normales ya quedan en rangos controlados)
        imbalance_norm = delta_p / max_p         # ≈ [-1, 1]
        soc_norm = soc_idx / max_soc_idx         # [0, 1]

        # --- 2. Recompensa por partes (SIN anidación) ---

        # Caso A: Grid IMPORTA y hay DÉFICIT (imbalance_norm < 0)
        if agent.action == 1 and imbalance_norm < 0.0:
            # déficit normalizado = -imbalance_norm  (p.ej. -1 → 1, -0.2 → 0.2)
            # batería baja → (1 - soc_norm)
            reward = self.psi * (
                0.5 * (-imbalance_norm) +      # magnitud del déficit
                0.5 * (1.0 - soc_norm)         # batería vacía/baja
            )

        # Caso B: Grid IMPORTA pero NO hay DÉFICIT (imbalance_norm >= 0)
        elif agent.action == 1 and imbalance_norm >= 0.0:
            # excedente normalizado = max(imbalance_norm, 0)
            # batería cargada → soc_norm
            reward = -self.sigma * (
                0.5 * max(imbalance_norm, 0.0) +   # excedente o al menos no déficit
                0.5 * soc_norm                     # SOC alto → más despilfarro
            )

        # Caso C: Grid IDLE y hay DÉFICIT (imbalance_norm < 0)
        elif agent.action == 0 and imbalance_norm < 0.0:
            # déficit grande + batería vacía → inacción muy grave
            reward = -self.nu * (
                0.5 * (-imbalance_norm) +      # tamaño del déficit
                0.5 * (1.0 - soc_norm)         # batería baja/vacía
            )

        # Caso D: Grid IDLE y NO hay DÉFICIT
        else:
            # estabilidad: no déficit (o leve) + batería con SOC aceptable
            reward = self.xi * (
                0.5 * (1.0 - max(-imbalance_norm, 0.0)) +  # “no déficit”: 1 cuando no falta nada
                0.5 * soc_norm                             # batería con energía disponible
            )

        # --- 3. Clipping final para mantener escala consistente ---
        reward = max(min(reward, 1.0), -1.0)
        return reward

@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Dynamic reward for controllable load agent (demand-side management).
    
    Implements economic-based reward modulation aligned with emergent systems paradigm.
    Rewards reflect real economic impact: savings when turning OFF at high prices,
    costs when consuming unnecessarily.
    
    Decision logic:
    - Load turns OFF only if: price > comfort_threshold AND (no internal energy)
    - Load stays ON if: price acceptable OR internal energy available
    - Internal energy = (soc_idx > soc_threshold) OR renewable surplus
    
    Reward modulation:
    - economic_signal = (base_demand / demand_max) × (price / price_max)
    - Reflects instantaneous system cost in normalized scale [0, 1]
    - All rewards scaled by this signal to reflect economic impact
    """

    def __init__(self, sigma=1.0, psi=1.5, nu=1.0, beta=0.2, **kwargs):
        # sigma: Scale factor for correct behavior (OFF when expensive, ON with internal energy)
        self.sigma = sigma
        # psi: Scale factor for incorrect behavior (ON when expensive) - higher penalty
        self.psi = psi
        # nu: Scale factor for missed opportunity (OFF when could use internal energy)
        self.nu = nu
        # beta: Scale factor for acceptable grid import (ON with low price, no internal)
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        # --- 1. Extract State Variables ---
        soc_idx, demand_idx, renewable_idx, price_idx = state_tuple
        action = agent.action  # 1 = ON, 0 = OFF

        # Access continuous values from environment
        real_price = env.price
        base_demand = getattr(env, 'base_demand', 0.0)
        
        # --- 2. Compute Normalization Factors ---
        # demand_max: maximum demand observed in current dataset (after scaling)
        demand_max = getattr(env, "demand_max", None)

        # Fallback to avoid division by zero or missing attributes
        if not demand_max or demand_max <= 0:
            demand_max = 1.0

        # Normalize base demand to [0, 1]
        demand_norm = base_demand / demand_max

        # Normalize discretized price index to [0, 1]
        # Uses the same number of bins configured in the environment
        num_price_bins = getattr(env, "num_price_bins", 1)
        if num_price_bins > 1:
            price_norm = price_idx / (num_price_bins - 1)
        else:
            price_norm = 0.0
        
        # Economic signal: instantaneous system cost (normalized)
        # Represents: "How much is the system costing right now?"
        economic_signal = demand_norm * price_norm
        
        # --- 3. Define Decision Variables ---
        # Get SOC threshold from agent configuration (default to 0 if not set)
        soc_threshold = getattr(agent, 'soc_threshold_idx', 0)
        comfort_threshold = getattr(agent, "comfort_threshold", 21.0)
        
        # Internal energy available: battery has charge OR renewable surplus
        internal_energy = (soc_idx > soc_threshold) or (renewable_idx > demand_idx)
        
        # Price is expensive: exceeds comfort threshold
        expensive = (real_price > comfort_threshold)
        
        # --- 4. Flat Reward Logic (NO NESTING) ---
        
        # CASE 1: OFF when price expensive and no internal energy
        # CORRECT behavior: Demand response to high price signal
        if action == 0 and expensive and not internal_energy:
            reward = self.sigma * economic_signal
        
        # CASE 2: ON when price expensive
        # INCORRECT behavior: Ignoring price signal, consuming at high cost
        elif action == 1 and expensive:
            reward = -self.psi * economic_signal
        
        # CASE 3: ON when internal energy available (regardless of price)
        # CORRECT behavior: Using renewable/battery energy
        elif action == 1 and internal_energy:
            reward = self.sigma * economic_signal
        
        # CASE 4: OFF when internal energy available and price acceptable
        # INCORRECT behavior: Missing opportunity to use cheap/free energy
        elif action == 0 and internal_energy and not expensive:
            reward = -self.nu * economic_signal
        
        # CASE 5: ON when price acceptable but no internal energy
        # ACCEPTABLE behavior: Importing from grid at reasonable price
        # Small positive reward (user willing to pay)
        elif action == 1 and not expensive and not internal_energy:
            reward = self.beta * economic_signal
        
        # CASE 6 (else): OFF when price acceptable and no internal energy
        # NEUTRAL behavior: Conservative, avoiding import
        else:
            reward = 0.0
        
        # --- 5. Clipping for Q-Learning Stability ---
        return max(min(reward, 1.0), -1.0)
