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
        solar_norm = 0.05 if solar_norm == 0 else solar_norm

        # --- 3. Lógica Plana (Flat Logic) ---

        # CASO A: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = PRODUCIR (1)
        if agent.action == 1 and delta_p >= 0 and solar_norm > 0:
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
        wind_norm = 0.05 if wind_norm == 0 else wind_norm

        # --- 3. Lógica Plana (Flat Logic) ---

        # CASO A: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = PRODUCIR (1)
        if delta_p >= 0 and agent.action == 1:
            reward = self.theta * imbalance_norm * wind_norm

        # CASO B: DÉFICIT (Falta energía) y ACCIÓN = IDLE (0)
        elif delta_p < 0 and agent.action == 0:
            reward = self.xi * imbalance_norm* wind_norm

        # CASO C: DÉFICIT (Falta energía) y ACCIÓN = PRODUCIR (1)
        elif delta_p < 0 and agent.action == 1:
            reward = -self.beta * imbalance_norm* wind_norm

        # CASO D: EXCEDENTE o BALANCE (Sobra energía) y ACCIÓN = IDLE (0)
        else:
            reward = -self.nu * imbalance_norm* wind_norm

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
    """Recompensa discreta para la carga controlable.

    Usa dos entradas del estado discretizado: potencia de red disponible
    (`grid_power_idx`) y precio (`price_idx`). Acciones esperadas: 0=OFF, 1=ON.

    - OFF sin potencia de red → penalización fija (`-sigma`).
    - ON con potencia y precio barato → premio proporcional (`psi * grid_norm`).
    - ON con potencia y precio caro → penalización proporcional (`-nu * grid_norm`).
    - Cualquier otro caso → premio/base neutral (`beta`).
    """

    def __init__(self, sigma=0.2, psi=1.0, nu=1.0, beta=0.2, **kwargs):
        # sigma: Penaliza permanecer OFF cuando no hay potencia de red
        self.sigma = sigma
        # psi: Premia consumir cuando hay potencia y precio barato
        self.psi = psi
        # nu: Penaliza consumir cuando hay potencia pero el precio es caro
        self.nu = nu
        # beta: Recompensa base para casos neutros
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        # --- 1. Extraer variables de estado ---
        grid_power_idx = state_tuple[0] # Índice discreto de potencia de red disponible
        price_idx = state_tuple[1]      # Índice discreto de precio (0=barato, 1=caro)

        # --- 2. Normalización dinámica ---
        max_p = max(env.num_power_bins - 1, 1)
        grid_norm = env.grid_power_idx / max_p

        # --- 3. Lógica de recompensa ---
        if agent.action == 0 and grid_power_idx == 0:
            reward = -self.sigma
        elif agent.action == 1 and grid_power_idx > 0 and price_idx==0:
            reward = self.psi * grid_norm
        elif agent.action == 1 and grid_power_idx > 0 and price_idx==1:
            reward = -self.nu * grid_norm
        else:
            reward = self.beta

        print(f"DEBUG: LoadReward - Action: {agent.action}, GridIdx: {grid_power_idx}, PriceIdx: {price_idx}, GridNorm: {grid_norm:.2f}, Reward: {reward:.2f}")

        # --- 5. Clipping for Q-Learning Stability ---
        return max(min(reward, 1.0), -1.0)
