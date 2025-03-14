import os
import sys
import numpy as np
import random
import pandas as pd

# Parámetros físicos y constantes
ETA = 0.15        # Eficiencia de conversión solar
SOLAR_AREA = 10   # Área de paneles solares en m^2
T_AMBIENT = 25    # Temperatura ambiente en °C
PHI = 1000        # Irradiancia solar en W/m^2
RHO = 1.225       # Densidad del aire en kg/m^3
BLADE_AREA = 5    # Área de los álabes de la turbina en m^2
C_P = 0.4         # Coeficiente de potencia
C_CONFORT = 0.5   # Umbral de confort para el costo del mercado

# -----------------------------------------------------
# 1) Definimos el entorno
# -----------------------------------------------------
class MultiAgentEnv:
    """
    Entorno que:
    - Carga datos desde un CSV (irradancia, velocidad de viento, demanda, precio...).
    - Discretiza estas variables en bins para crear estados (solar_bins, wind_bins, etc.).
    - Avanza step a step por filas del dataset (o de manera aleatoria).
    - Puede gestionarse en un bucle de episodios.
    """

    def __init__(self, csv_filename="Case1_energy_data_with_pv_power.csv", num_demand_bins=7, num_renewable_bins=7):
        """
        Parámetros:
          - num_*_bins: define cuántos intervalos se utilizan para discretizar cada variable.
        """

        self.renewable_power = 0
        self.total_power = 0
        self.demand_power = 0        
        self.dif_power = 0
        
        # Cargamos el dataset
        self.dataset = self._load_data(csv_filename)

        self.max_value = self.dataset.apply(pd.to_numeric, errors='coerce').max().max()
        
        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.demand_bins = np.linspace(0, self.max_value, num_demand_bins)
        self.renewable_bins = np.linspace(0, self.max_value, num_renewable_bins)

        # Índice actual dentro del dataset (simularemos step a step)
        self.current_index = 0
        
        # Estado inicial (discretizado)
        self.state = None

    def _load_data(self, filename):
        # Ruta al archivo
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep=';', engine='python')
        #print("Primeras filas del dataset:\n", df.head())
        
        return df

    def step(self):
        """
        Avanza al siguiente índice del dataset (o uno aleatorio) y retorna el nuevo estado.
        """
        self.current_index += 1
        if self.current_index >= len(self.dataset):
            self.current_index = 0  # En un entorno real, podría definirse done=True en vez de wrap-around
        
        self.state = self._get_discretized_state(self.current_index)
        return self.state

    def _get_discretized_state(self, index):
        """
        Toma valores reales (irradancia, viento, demanda, precio, etc.) y los discretiza en bins,
        devolviendo una tupla como (idx_solar, idx_wind, idx_battery, idx_demand, idx_price).
        """
        row = self.dataset.iloc[index]
        
        self.demand_power = row["demand"]

        # Discretizamos
        demand_power_idx = np.digitize([row["demand"]], self.demand_bins)[0] - 1
        renewable_power_idx = np.digitize([self.renewable_power], self.renewable_bins)[0] - 1
        
        # Retornamos la tupla de estado discretizado
        return (demand_power_idx, renewable_power_idx)

# -----------------------------------------------------
# 2) Definimos la clase base de Agente con Q-Table
# -----------------------------------------------------
class BaseAgent:
    """
    Clase base para agentes con Q-table.
    """
    def __init__(self, name, actions, alpha=0.1, gamma=0.9, isPower=True):
        self.name = name
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.isPower = isPower
        self.q_table = {}   
        self.current_power = 0.0

    def choose_action(self, state, epsilon=0.1):
        """
        Selecciona acción con política epsilon-greedy.
        """
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            # Escoge la acción con Q máximo
            q_values = self.q_table.get(state, {a: 0.0 for a in self.actions})
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza la Q-table según Q-Learning:
          Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        q_values = self.q_table[state]
        current_q = q_values[action]
        
        # Si next_state no está en la Q-table (caso borde), asumimos Q=0
        next_q_values = self.q_table.get(next_state, {a: 0.0 for a in self.actions})
        max_next_q = max(next_q_values.values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

# -----------------------------------------------------
# 3) Agentes Especializados (Solar, Wind, Battery, Grid, Load)
#    Heredan de BaseAgent y añaden sus recompensas
# -----------------------------------------------------
class SolarAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, num_solar_bins=7):
        super().__init__("solar", ["produce", "idle"], alpha=0.1, gamma=0.9, isPower=True)

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.solar_power_bins = np.linspace(0, env.max_value, num_solar_bins)
        self.solar_state_bins = [0, 1]
        self.solar_state = 0        

    def _get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
        row = env.dataset.iloc[index]
        self.current_power = row["solar_power"]
        
        # Discretizamos
        solar_power_idx = np.digitize([self.current_power], self.solar_power_bins)[0] - 1
        solar_state_idx = np.digitize([self.solar_state], self.solar_state_bins)[0] - 1
        
        print("*"*100)
        print(self.current_power)
        print(self.solar_power_bins)
        print(solar_power_idx)

        state_env = env._get_discretized_state(index)
        
        # Retornamos la tupla de estado discretizado
        print("*** solar_power_idx, solar_state_idx, renewable_power_idx, demand_power_idx ***")
        return (solar_power_idx, solar_state_idx, state_env[1], state_env[0])

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        """
        states = []
        for a in range(len(self.solar_power_bins)):
            for b in range(len(self.solar_state_bins)):
                for c in range(len(env.renewable_bins)):
                    for d in range(len(env.demand_bins)):
                        states.append((a, b, c, d))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def calculate_power(self, row):

        if self.isPower:
            return row["solar_power"]
        else:
            return ETA * SOLAR_AREA * row["irradiance"] * (1 - 0.005*(T_AMBIENT + 25))

        # Ejemplo muy simplificado

    def calculate_reward(self, P_H, P_L, SOC, active_agents):
        """
        P_H: potencia generada por el panel local
        P_L: demanda local (o parte de la demanda)
        SOC: estado de la batería
        active_agents: lista con otros agentes que puedan generar potencia, etc.
        """
        total_renewables = 0
        for ag in active_agents:
            if isinstance(ag, (SolarAgent, WindAgent)):
                total_renewables += ag.calculate_power(random.uniform(0,1))
        
        if P_H <= P_L:
            return 1.0 * (P_L / (P_H + total_renewables + 1e-6))
        elif P_H > P_L and SOC == 1:
            return 1.0 * (P_L - (P_H + total_renewables))
        elif P_H > P_L and SOC < 1:
            return -1.0 * ((P_H + total_renewables - P_L))
        return 0.0

class WindAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, num_wind_bins=7):
        super().__init__("wind", ["produce", "idle"], alpha=0.1, gamma=0.9, isPower=True)
        
        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.wind_power_bins = np.linspace(0, env.max_value, num_wind_bins)
        self.wind_state_bins = [0, 1]
        self.wind_state = 0
        

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        """
        states = []
        for a in range(len(self.wind_power_bins)):
            for b in range(len(self.wind_state_bins)):
                for c in range(len(env.renewable_bins)):
                    for d in range(len(env.demand_bins)):
                        states.append((a, b, c, d))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def _get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
        row = env.dataset.iloc[index]
        self.current_power = row["wind_power"]
        
        # Discretizamos
        wind_power_idx = np.digitize([self.current_power], self.wind_power_bins)[0] - 1
        wind_state_idx = np.digitize([self.wind_state], self.wind_state_bins)[0] - 1
        
        print("*"*100)
        print(self.current_power)
        print(self.wind_power_bins)
        print(wind_power_idx)

        state_env = env._get_discretized_state(index)
        
        # Retornamos la tupla de estado discretizado
        print("*** wind_power_idx, wind_state_idx, renewable_power_idx, demand_power_idx ***")
        return (wind_power_idx, wind_state_idx, state_env[1], state_env[0])

    def calculate_power(self, row):

        if self.isPower:
            return row["wind_power"]
        else:
            return 0.5 * RHO * BLADE_AREA * C_P * (row["wind speed"]**3)

    def calculate_reward(self, P_H, P_L, SOC, active_agents):
        # Ejemplo simplificado
        if P_H <= P_L:
            return 1.0 * (P_L / (P_H + 1e-6))
        elif P_H > P_L and SOC == 1:
            return 1.0 * (P_L - (P_H / len(active_agents)))
        elif P_H > P_L and SOC < 1:
            return -1.0 * ((P_H - P_L) / len(active_agents))
        return 0.0

class BatteryAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, capacity_ah= 10000, num_battery_soc_bins=4):
        super().__init__("battery", ["charge", "discharge", "idle"], alpha=0.1, gamma=0.9)
        """
        Inicializa la batería con una capacidad fija en Ah y un SOC inicial del 50%.
        :param capacity_ah: Capacidad de la batería en Amperios-hora (Ah).
        """
        self.capacity_ah = capacity_ah  # Capacidad fija en Ah
        self.soc = 50.0  # Estado de carga inicial en %
        self.battery_power = 0.0  # Potencia en W
        self.battery_state = "idle"  # Estado inicial de operación

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        self.battery_soc_bins = np.linspace(0, 100, num_battery_soc_bins)

    def update_soc(self, power_w: float):
        """
        Actualiza el SOC basado en la potencia suministrada o extraída en una unidad de tiempo de 1 hora.
        
        :param power_w: Potencia en Watts (positiva si está cargando, negativa si está descargando).
        """
        # Convertimos potencia en energía transferida en Wh
        energy_wh = power_w  # Asumimos 1 hora de tiempo fijo
        
        # Convertimos la energía en Wh a Ah usando la capacidad de la batería
        delta_soc = (energy_wh / (self.capacity_ah * 1)) * 100  # Expresado en porcentaje
        
        # Actualizamos el SOC asegurándonos de que se mantenga en los límites de 0% a 100%
        self.soc = max(0.0, min(100.0, self.soc + delta_soc))
        
        # Actualizamos la potencia y el estado de la batería
        self.battery_power = power_w
        
        if power_w > 0:
            self.battery_state = "charging"
        elif power_w < 0:
            self.battery_state = "discharging"
        else:
            self.battery_state = "idle"

    def get_state(self) -> str:
        """
        Retorna el estado actual de la batería.
        
        :return: "charging", "discharging" o "idle".
        """
        return self.battery_state
    
    def get_soc(self) -> float:
        """
        Retorna el estado de carga actual en porcentaje.
        
        :return: SOC en porcentaje (0% - 100%).
        """
        return self.soc

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        """
        states = []
        for a in range(len(self.battery_soc_bins)):
            for b in range(3):
                for c in range(len(env.renewable_bins)):
                    for d in range(len(env.demand_bins)):
                        states.append((a, b, c, d))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def _get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
       
        # Discretizamos
        battery_power_idx = np.digitize([self.soc], self.battery_soc_bins)[0] - 1
        
        print("*"*100)
        print(self.soc)
        print(self.battery_soc_bins)
        print(battery_power_idx)

        state_env = env._get_discretized_state(index)
        
        # Retornamos la tupla de estado discretizado
        print("*** battery_power_idx, self.battery_state, renewable_power_idx, demand_power_idx ***")
        return (battery_power_idx, self.battery_state, state_env[1], state_env[0])

    def calculate_reward_charge(self, P_T, P_L):
        if self.soc < 1 and P_T > P_L:
            return 1.0 * (1 - self.soc) * (P_T - P_L)
        elif self.soc <= 1 and P_T <= P_L:
            return -1.0 * self.soc * (P_L - P_T)
        return 0

    def calculate_reward_discharge(self, P_T, P_L):
        if self.soc >= 0.5 and P_T < P_L:
            return 1.0 * self.soc * (P_L - P_T)
        elif self.soc < 0.5 and P_T > P_L:
            return -1.0 * (1 - self.soc) * (P_T - P_L)
        elif self.soc < 0.5 and P_T < P_L:
            return -1.0 * (1 - self.soc) * (P_L - P_T)
        return 0

class GridAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        super().__init__("grid", ["sell", "idle"], alpha=0.1, gamma=0.9)

        self.grid_power = 0.0  # Potencia en W
        self.grid_state = "idle"  # Estado inicial de operación

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        #self.battery_soc_bins = np.linspace(0, 100, num_battery_soc_bins)

        self.ess =  ess

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        """
        states = []
        for a in range(2):
            for b in range(len(self.ess.battery_soc_bins)):
                for c in range(3):
                    for d in range(len(env.renewable_bins)):
                        for e in range(len(env.demand_bins)):
                            states.append((a, b, c, d, e))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def _get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
       
        # Discretizamos
        battery_power_idx = np.digitize([self.ess.soc], self.ess.battery_soc_bins)[0] - 1
        
        print("*"*100)
        print(self.ess.soc)
        print(self.ess.battery_soc_bins)
        print(battery_power_idx)

        state_env = env._get_discretized_state(index)
        
        # Retornamos la tupla de estado discretizado
        print("*** grid_state, battery_power_idx, self.battery_state, renewable_power_idx, demand_power_idx ***")
        return (self.grid_state, battery_power_idx, self.ess.battery_state, state_env[1], state_env[0])        

    def calculate_reward(self, action, P_T, P_L, SOC, C_mercado, S_UT):
        if action == "sell":
            if SOC < 0.5 and P_T < P_L and S_UT == 0:
                return 1.0 / (P_T * C_mercado + 1e-6)
            elif SOC >= 0.5 and P_T >= P_L and S_UT == 0:
                return -1.0 * P_T * C_mercado
            elif (SOC >= 0.5 or P_T >= P_L) and S_UT == 1:
                return -1.0 * P_T * C_mercado
        return 0.0

class LoadAgent(BaseAgent):
    def __init__(self):
        super().__init__("load", ["consume", "idle"], alpha=0.1, gamma=0.9)

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        (solar_bins, wind_bins, battery_soc_bins, demand_bins)
        """
        states = []
        for a in range(len(env.solar_power_bins)):
            for b in range(len(env.state_solar_bins)):
                for c in range(len(env.wind_power_bins)):
                    for d in range(len(env.state_wind_bins)):
                        for e in range(len(env.demand_bins)):
                            for f in range(len(env.battery_soc_bins)):
                                states.append((a, b, c, d, e, f))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def calculate_reward(self, action, P_T, P_L, SOC, C_mercado):
        if action == "consume":
            if P_T > P_L or SOC >= 0.5:
                return 1.0 * (1.0 / (P_T + C_mercado + 1e-6))
            elif C_mercado > C_CONFORT:
                return -1.0 * P_T * C_mercado
        elif action == "idle":
            if P_T > P_L or SOC >= 0.5:
                return -1.0 * P_T * C_mercado
        return 0.0

# -----------------------------------------------------
# 4) Simulación de entrenamiento
# -----------------------------------------------------
class Simulation:
    def __init__(self, num_episodes=10, max_steps=5):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Creamos el entorno que carga el CSV y discretiza
        self.env = MultiAgentEnv(csv_filename="Case1_energy_data_with_pv_power.csv", num_demand_bins=7)
        
        # Definimos un conjunto de agentes (ejemplo: 1 pv, 1 battery, 1 grid, 1 load)
        self.agents = [
            SolarAgent(self.env),
            WindAgent(self.env),
            BatteryAgent(self.env),
            GridAgent(self.env, BatteryAgent(self.env))
        ]
        
        # Parámetros de entrenamiento
        self.epsilon = 0.1  # Exploración \epsilon
        
        # Inicializamos Q-tables
        for agent in self.agents:
            agent.initialize_q_table(self.env)
                
        
    def run(self):
        for ep in range(self.num_episodes):
            # Reseteamos entorno y los agentes al inicio de cada episodio
            self.env._get_discretized_state(0)
            for agent in self.agents:
                if isinstance(agent, SolarAgent):
                    state_solar = agent._get_discretized_state(self.env, 0)
                    print(state_solar)
                elif isinstance(agent, WindAgent):
                    state_wind = agent._get_discretized_state(self.env, 0)
                    print(state_wind)
                elif isinstance(agent, BatteryAgent):
                    state_battery = agent._get_discretized_state(self.env, 0)
                    print(state_battery)
                elif isinstance(agent, GridAgent):
                    state_grid = agent._get_discretized_state(self.env, 0)
                    print(state_grid)

            for step in range(self.max_steps):

                # Obtenemos variables reales del entorno (por ejemplo, para recompensas)
                # row real:
                row = self.env.dataset.iloc[self.env.current_index]
                
                # Valor "total de potencia" (muy simplificado)
                # Suponemos que cada agente "produce" si su acción es "produce"
                # y 0 en otro caso. Podrías refinarlo según la acción de cada uno.
                self.env.renewable_power = 0.0
                self.env.total_power = 0
                for agent in self.agents:
                    if isinstance(agent, SolarAgent):
                        # Escoger acción
                        action = agent.choose_action(state_solar, self.epsilon)
                        print(action)
                        if action == "produce":
                            agent.solar_state = 1
                            self.env.renewable_power += agent.solar_state*agent.current_power
                        else:
                            self.env.renewable_power += 0.0
                        
                    elif isinstance(agent, WindAgent):
                        action = agent.choose_action(state_wind, self.epsilon)
                        print(action)
                        if action == "produce":
                            agent.solar_state = 1
                            self.env.renewable_power += agent.solar_state*agent.current_power
                        else:
                            self.env.renewable_power += 0.0
                    elif isinstance(agent, BatteryAgent):
                        action = agent.choose_action(state_battery, self.epsilon)
                        print(action)

                        if action == "charge":
                            agent.battery_state = "charging" 
                        elif action == "discharge":
                            agent.battery_state = "discharging" 
                        else:
                            agent.battery_state = "idle" 
                    elif isinstance(agent, GridAgent):
                        action = agent.choose_action(state_grid, self.epsilon)
                        print(action)
                            


                    else:
                        # GridAgent, LoadAgent no generan en este ejemplo
                        _ = agent.choose_action(state, self.epsilon)                
                
                self.env.total_power = self.env.renewable_power
                self.dif_power = self.env.total_power - self.env.demand_power
                
                sys.exit(0)
                # Ahora calculamos la recompensa individual por agente
                # y actualizamos la Q-table
                next_state = self.env.step()  # Avanzamos el entorno un índice
                '''for ag in self.agents:
                    # Recuperamos la acción que tomó este agente en este step
                    # Para un enfoque riguroso, cada agente debería almacenar su "acción" actual,
                    # aquí simplificamos volviendo a elegir la misma acción con choose_action(...) 
                    # o se podría guardar en un diccionario {agent:action} en el bucle anterior
                    # para no volver a generarla.
                    
                    # -- EJEMPLO: asumiremos la misma acción que generamos antes (guardándola):
                    # Para hacerlo rápido, repetimos la llamada (no es lo ideal).
                    action = ag.choose_action(state, self.epsilon)
                    
                    # Calculamos la recompensa según el tipo de agente
                    if isinstance(ag, SolarAgent):
                        # Asumimos que P_H es la parte de P_total que generó solar
                        # Para simplificar, lo aproximamos como P_total (no es lo ideal)
                        reward = ag.calculate_reward(
                            P_H=P_total, 
                            P_L=demand, 
                            SOC=0.5, 
                            active_agents=self.agents
                        )
                    elif isinstance(ag, WindAgent):
                        reward = ag.calculate_reward(
                            P_H=P_total, 
                            P_L=demand, 
                            SOC=0.5, 
                            active_agents=self.agents
                        )
                    elif isinstance(ag, BatteryAgent):
                        # Ejemplo simplificado: 
                        if action == "charge":
                            reward = ag.calculate_reward_charge(P_T=P_total, P_L=demand)
                        elif action == "discharge":
                            reward = ag.calculate_reward_discharge(P_T=P_total, P_L=demand)
                        else:
                            reward = 0.0
                        ag.update_soc(action)
                    elif isinstance(ag, GridAgent):
                        # S_UT aleatorio 0/1, C_mercado=price
                        S_UT = random.choice([0,1])
                        reward = ag.calculate_reward(action, P_T=P_total, P_L=demand, 
                                                     SOC=0.5, C_mercado=price, S_UT=S_UT)
                    elif isinstance(ag, LoadAgent):
                        reward = ag.calculate_reward(action, P_T=P_total, P_L=demand,
                                                     SOC=0.5, C_mercado=price)
                    else:
                        reward = 0.0
                    '''
                    # Actualizamos Q-table
                    #ag.update_q_table(state, action, reward, next_state)
                
                # Actualizamos el estado actual
                #state = next_state
                
            
            print(f"Fin episodio {ep+1}/{self.num_episodes}")

# -----------------------------------------------------
# 5) Punto de entrada principal
# -----------------------------------------------------
if __name__ == "__main__":
    sim = Simulation(num_episodes=5, max_steps=24)
    sim.run()
