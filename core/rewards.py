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

    def __init__(self, theta=1.0, beta=1.0, eta=1.0, nu=1.0, xi=1.0, **kwargs):
        self.theta = theta
        self.beta = beta
        self.eta = eta
        self.nu = nu
        self.xi = xi

    def compute(self, agent, env, state_tuple):

        # --- 1. Obtener Variables de Estado y Entorno ---
        renewable_idx = env.renewable_power_idx
        demand_idx = env.demand_power_idx
        
        # La contribución potencial de ESTE agente
        solar_potential_idx = state_tuple[0] 
        
        # --- 2. Calcular Desequilibrio y Magnitud ---
        delta_p = renewable_idx - demand_idx
        delta_abs = max(abs(delta_p), 1)

        # --- 3. Lógica de Recompensa (Estructura Plana) ---

        # CASO 1: Suministro Correcto (PREMIO)
        # (Acción=Suministrar, Tenía potencial)
        if agent.action == 1 and solar_potential_idx > 0:
            reward =  self.theta * solar_potential_idx
        
        # CASO 2: Suministro Ilógico (CASTIGO)
        # (Acción=Suministrar, PERO no tenía potencial)
        # (Solo entra aquí si action==1 Y solar_potential_idx==0)
        elif agent.action == 1:
            reward =  -self.beta 

        # CASO 3: Inacción Correcta (Forzada) (PREMIO)
        # (Acción=No Suministrar, No tenía potencial)
        elif agent.action == 0 and solar_potential_idx == 0:
            reward =  self.eta
        
        # CASO 4: Inacción Correcta (Inteligente) (PREMIO)
        # (Acción=No Suministrar, Tenía potencial, Había excedente)
        # (Solo entra aquí si action==0, solar_potential_idx>0 Y delta_p>0)
        elif agent.action == 0 and delta_p > 0:
            reward =  self.nu * delta_abs 
        
        # CASO 5: Inacción Incorrecta (Fallo) (CASTIGO)
        # (Único caso restante: Acción=No Suministrar, Tenía potencial, Había déficit)
        else: 
            reward =  -self.xi * delta_abs

        return reward

@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Replica la lógica de WindAgent.calculate_reward."""

    def __init__(self, theta=1.0, beta=1.0, eta=1.0, nu=1.0, xi=1.0, **kwargs):
        self.theta = theta
        self.beta = beta
        self.eta = eta
        self.nu = nu
        self.xi = xi

    def compute(self, agent, env, state_tuple):

        # --- 1. Obtener Variables de Estado y Entorno ---
        renewable_idx = env.renewable_power_idx
        demand_idx = env.demand_power_idx
        
        # La contribución potencial de ESTE agente
        wind_potential_idx = state_tuple[0]  
        
        # --- 2. Calcular Desequilibrio y Magnitud ---
        delta_p = renewable_idx - demand_idx
        delta_abs = max(abs(delta_p), 1)

        # --- 3. Lógica de Recompensa (Estructura Plana) ---
        
        # CASO 1: Suministro Correcto (PREMIO)
        # (Acción=Suministrar, Tenía potencial)
        if agent.action == 1 and wind_potential_idx > 0:
            reward =  self.theta * wind_potential_idx

        # CASO 2: Suministro Ilógico (CASTIGO)
        # (Acción=Suministrar, PERO no tenía potencial)
        # (Solo entra aquí si action==1 Y wind_potential_idx==0)
        elif agent.action == 1:
            reward =  -self.beta 

        # CASO 3: Inacción Correcta (Forzada) (PREMIO)
        # (Acción=No Suministrar, No tenía potencial)
        elif agent.action == 0 and wind_potential_idx == 0:
            reward =  self.eta
        
        # CASO 4: Inacción Correcta (Inteligente) (PREMIO)
        # (Acción=No Suministrar, Tenía potencial, Había excedente)
        # (Solo entra aquí si action==0, wind_potential_idx>0 Y delta_pº>0)
        elif agent.action == 0 and delta_p > 0:
            reward =  self.nu * delta_abs 
        
        # CASO 5: Inacción Incorrecta (Fallo) (CASTIGO)
        # (Único caso restante: Acción=No Suministrar, Tenía potencial, Había déficit)
        else: 
            reward = -self.xi * delta_abs

        return reward

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
        soc, demand_idx, _ = state_tuple # total_idx no se usa

        # Corrección: Definir el desequilibrio ANTES de la acción de la batería
        renewable_idx = env.renewable_power_idx
        delta_p = renewable_idx - demand_idx

        # Asegurar que soc_max esté definido
        agent.soc_max = 4 # O el valor real

        # --- 3. Lógica de Recompensa (Corregida) ---

        # CASO 1: Descarga Correcta (PREMIO)
        # (Acción=Descargar, Hay Déficit, Batería tiene carga)
        if agent.action == 2 and delta_p < 0 and soc > 0:
            # Premio por suplir la demanda
            reward = self.psi * abs(delta_p) * soc

        # CASO 2: Descarga Incorrecta (CASTIGO)
        # (Acción=Descargar, PERO hay Excedente O Batería vacía)
        elif agent.action == 2 and (delta_p >= 0 or soc == 0):
            # Castigo fijo por acción ilógica o innecesaria
            reward = -self.sigma

        # CASO 3: Carga Correcta (PREMIO)
        # (Acción=Cargar, Hay Excedente)
        elif agent.action == 1 and delta_p > 0:
            # Premio por almacenar excedente (escala con espacio vacío)
            reward = self.nu * delta_p * (agent.soc_max - soc)

        # CASO 4: Carga Incorrecta (CASTIGO)
        # (Acción=Cargar, PERO hay Déficit)
        elif agent.action == 1 and delta_p <= 0:
            # Castigo por empeorar el déficit
            reward = -self.beta * abs(delta_p)

        # CASO 5: Inacción Incorrecta (CASTIGO)
        # (Acción=Inactivo, PERO hay Desequilibrio)
        elif agent.action == 0 and soc > 0:
            # Castigo por no actuar (cargar o descargar)
            reward = -self.xi * abs(delta_p)

        # CASO 6 ("else"): Inacción Correcta (PREMIO)
        # (Acción=Inactivo, Hay Equilibrio perfecto)
        else:
            # Tu corrección: Premio por inacción correcta
            reward = self.mu
        
        return reward


@register_reward("DefaultGridReward")
class DefaultGridReward(RewardFn):
    """Simple, stable Grid reward using:
       - discretized indices (renewable_idx, demand_idx, soc_idx)
       - binary SOC interpretation (soc_idx == 0 => battery empty)
       - hyperparameters psi, sigma, nu, xi controlling all magnitudes.
    """

    def __init__(self, psi=1.0, sigma=1.0, nu=1.0, xi=1.0, C_M=1.0, **kwargs):
        self.psi = psi     # reward: import when deficit + battery empty
        self.sigma = sigma # penalty: import without need
        self.nu = nu       # penalty: idle with deficit + battery empty
        self.xi = xi       # reward: idle when no deficit
        self.C_M = C_M     # (kept for compatibility but NOT used to scale reward)

    def compute(self, agent, env, state_tuple):
        # --- 1. Variables de estado en forma de índices discretizados ---
        soc_idx = state_tuple[0]
        demand_idx = env.demand_power_idx
        renewable_idx = env.renewable_power_idx
        
        # SOC binario
        battery_empty = (soc_idx == 0)

        # Deficit en espacio de índices
        # deficit_idx > 0 significa: demanda > renovables
        deficit_idx = (demand_idx - renewable_idx)
        has_deficit = (deficit_idx > 0)

        # ----------------------------------------------------------
        # NUEVA LÓGICA SIMPLE (versión A), usando hiperparámetros:
        # psi   → premio por importar cuando es necesario
        # sigma → castigo por importar sin necesidad
        # nu    → castigo por idle cuando hay déficit + bateria vacía
        # xi    → premio por idle cuando NO hay déficit
        # ----------------------------------------------------------

        # CASO 1: Grid actúa (importa)
        if agent.action == 1:
            if has_deficit and battery_empty:
                # Importación necesaria → premio controlado por psi
                reward = self.psi
            else:
                # Importa sin necesidad → castigo controlado por sigma
                reward = -self.sigma

        # CASO 2: Grid idle
        else:
            if has_deficit and battery_empty:
                # Inacción crítica → castigo controlado por nu
                reward = -self.nu
            else:
                # Idle correcto → premio controlado por xi
                reward = self.xi

        return reward

@register_reward("DefaultLoadReward")
class DefaultLoadReward(RewardFn):
    """Replica la lógica de LoadAgent.calculate_reward."""

    def __init__(self, sigma=1.0, psi=1.0, nu=1.0, beta=-1.0, **kwargs):
        self.sigma = sigma
        self.psi = psi
        self.nu = nu
        self.beta = beta

    def compute(self, agent, env, state_tuple):
        soc_idx, demand_idx, renewable_idx, price_idx = state_tuple
        action = agent.action  # 1 = ON, 0 = OFF

        # Access REAL continuous price from environment, not discretized index
        real_price = env.price
        
        surplus = (renewable_idx > demand_idx)
        expensive = (real_price > getattr(agent, "comfort_threshold", 1.0))
        internal = (soc_idx > 1 or surplus)
        
        print("DEBUG LoadReward:", "renewable_idx: ", renewable_idx, "demand_idx: ", demand_idx, "price_idx: ", price_idx, "real_price: ", real_price, "comfort_threshold: ", getattr(agent, "comfort_threshold"), "soc_idx: ", soc_idx)
        
        # ================================
        #   REWARD SIMPLE SIN ANIDACIÓN
        # ================================
        if action == 1 and internal:
            reward = self.sigma             # encender con energía interna → correcto

        elif action == 1 and expensive:
            reward = -self.psi              # encender caro sin interno → incorrecto

        elif action == 0 and internal:
            reward = -self.nu               # apagar con SOC/excedente → oportunidad perdida

        elif action == 0 and expensive:
            reward = self.sigma             # apagar caro → comportamiento racional

        else:
            reward = self.beta              # zona neutra

        return reward
