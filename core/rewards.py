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

        # --- 1. Obtener Variables de Estado y Entorno ---
        renewable_idx = env.renewable_power_idx
        demand_idx = env.demand_power_idx
        
        # La contribución potencial de ESTE agente
        solar_potential_idx = agent.solar_potential_idx 
        
        # --- 2. Calcular Desequilibrio y Magnitud ---
        # (Asumimos que renewable_idx es el total *DESPUÉS* de la acción)
        # (Si renewable_idx es ANTES, el cálculo de delta_p cambiaría)
        delta_p = renewable_idx - demand_idx
        delta_abs = max(abs(delta_p), 1)

        # --- 3. Lógica de Recompensa (Corregida) ---

        # === CASOS PARA action == 1 (Suministrar) ===

        if agent.action == 1:
            # CASO 1.A: Suministro Correcto (PREMIO)
            # (Acción=Suministrar, Tenía potencial para hacerlo)
            if solar_potential_idx > 0:
                # Tu "Opción A": Premio por el esfuerzo/contribución
                # (Usamos 'self.theta' como el ponderador)
                return self.theta * solar_potential_idx
            
            # CASO 1.B: Suministro Ilógico (CASTIGO)
            # (Acción=Suministrar, PERO no tenía potencial)
            else: 
                # Castigo por una acción ilógica/imposible
                # (Usamos 'self.sigma' como un nuevo hiperparámetro)
                return -self.sigma 

        # === CASOS PARA action == 0 (No Suministrar) ===

        else: 
            # CASO 2.A: Inacción Correcta (Forzada) (PREMIO)
            # (Acción=No Suministrar, No tenía potencial)
            if solar_potential_idx == 0:
                # Tu propuesta: Premio fijo por "inacción correcta forzada"
                # (Usamos 'self.eta' como el premio fijo)
                return self.eta
            
            # CASO 2.B: Inacción (Inteligente o Incorrecta) (CON POTENCIAL)
            # (Acción=No Suministrar, PERO sí tenía potencial)
            else:
                # En este caso, SÍ importa el estado de la red
                
                # CASO 2.B.i: Inacción Correcta (Inteligente) (PREMIO)
                # (No suministró, había potencial, PERO había excedente)
                if delta_p > 0:
                    # Premio por no empeorar el excedente
                    # (Usamos 'self.gamma' como el ponderador escalado)
                    return self.gamma * delta_abs 
                
                # CASO 2.B.ii: Inacción Incorrecta (Fallo) (CASTIGO)
                # (No suministró, había potencial, Y había déficit)
                else: 
                    # Tu propuesta: Castigo por no ayudar
                    # (Usamos 'self.xi' como el ponderador escalado)
                    return -self.xi * delta_abs

@register_reward("DefaultWindReward")
class DefaultWindReward(RewardFn):
    """Replica la lógica de WindAgent.calculate_reward."""

    def __init__(self, theta=1.0, beta=1.0, eta=1.0, xi=1.0, **kwargs):
        self.theta = theta
        self.beta = beta
        self.eta = eta
        self.xi = xi

    def compute(self, agent, env, state_tuple):

        # --- 1. Obtener Variables de Estado y Entorno ---
        renewable_idx = env.renewable_power_idx
        demand_idx = env.demand_power_idx
        
        # La contribución potencial de ESTE agente
        solar_potential_idx = agent.solar_potential_idx 
        
        # --- 2. Calcular Desequilibrio y Magnitud ---
        # (Asumimos que renewable_idx es el total *DESPUÉS* de la acción)
        # (Si renewable_idx es ANTES, el cálculo de delta_p cambiaría)
        delta_p = renewable_idx - demand_idx
        delta_abs = max(abs(delta_p), 1)

        # --- 3. Lógica de Recompensa (Corregida) ---

        # === CASOS PARA action == 1 (Suministrar) ===

        if agent.action == 1:
            # CASO 1.A: Suministro Correcto (PREMIO)
            # (Acción=Suministrar, Tenía potencial para hacerlo)
            if solar_potential_idx > 0:
                # Tu "Opción A": Premio por el esfuerzo/contribución
                # (Usamos 'self.theta' como el ponderador)
                return self.theta * solar_potential_idx
            
            # CASO 1.B: Suministro Ilógico (CASTIGO)
            # (Acción=Suministrar, PERO no tenía potencial)
            else: 
                # Castigo por una acción ilógica/imposible
                # (Usamos 'self.sigma' como un nuevo hiperparámetro)
                return -self.sigma 

        # === CASOS PARA action == 0 (No Suministrar) ===

        else: 
            # CASO 2.A: Inacción Correcta (Forzada) (PREMIO)
            # (Acción=No Suministrar, No tenía potencial)
            if solar_potential_idx == 0:
                # Tu propuesta: Premio fijo por "inacción correcta forzada"
                # (Usamos 'self.eta' como el premio fijo)
                return self.eta
            
            # CASO 2.B: Inacción (Inteligente o Incorrecta) (CON POTENCIAL)
            # (Acción=No Suministrar, PERO sí tenía potencial)
            else:
                # En este caso, SÍ importa el estado de la red
                
                # CASO 2.B.i: Inacción Correcta (Inteligente) (PREMIO)
                # (No suministró, había potencial, PERO había excedente)
                if delta_p > 0:
                    # Premio por no empeorar el excedente
                    # (Usamos 'self.gamma' como el ponderador escalado)
                    return self.gamma * delta_abs 
                
                # CASO 2.B.ii: Inacción Incorrecta (Fallo) (CASTIGO)
                # (No suministró, había potencial, Y había déficit)
                else: 
                    # Tu propuesta: Castigo por no ayudar
                    # (Usamos 'self.xi' como el ponderador escalado)
                    return -self.xi * delta_abs


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
            return self.psi * abs(delta_p) * soc

        # CASO 2: Descarga Incorrecta (CASTIGO)
        # (Acción=Descargar, PERO hay Excedente O Batería vacía)
        elif agent.action == 2 and (delta_p >= 0 or soc == 0):
            # Castigo fijo por acción ilógica o innecesaria
            return -self.sigma

        # CASO 3: Carga Correcta (PREMIO)
        # (Acción=Cargar, Hay Excedente)
        elif agent.action == 1 and delta_p > 0:
            # Premio por almacenar excedente (escala con espacio vacío)
            return self.nu * delta_p * (agent.soc_max - soc)

        # CASO 4: Carga Incorrecta (CASTIGO)
        # (Acción=Cargar, PERO hay Déficit)
        elif agent.action == 1 and delta_p <= 0:
            # Castigo por empeorar el déficit
            return -self.beta * abs(delta_p)

        # CASO 5: Inacción Incorrecta (CASTIGO)
        # (Acción=Inactivo, PERO hay Desequilibrio)
        elif agent.action == 0 and abs(delta_p) > 0:
            # Castigo por no actuar (cargar o descargar)
            return -self.xi * abs(delta_p)

        # CASO 6 ("else"): Inacción Correcta (PREMIO)
        # (Acción=Inactivo, Hay Equilibrio perfecto)
        else:
            # Tu corrección: Premio por inacción correcta
            return self.mu


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