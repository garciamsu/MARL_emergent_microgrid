import os
import sys
import matplotlib
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import copy
from tabulate import tabulate
from itertools import cycle
from analysis_tools import compute_q_diff_norm, plot_metric, check_stability, load_latest_evolution_csv, process_evolution_data, plot_coordination, clear_results_directories


# Parámetros físicos y constantes
ETA = 0.15        # Eficiencia de conversión solar
SOLAR_AREA = 10   # Área de paneles solares en m^2
T_AMBIENT = 25    # Temperatura ambiente en °C
PHI = 1000        # Irradiancia solar en W/m^2
RHO = 1.225       # Densidad del aire en kg/m^3
BLADE_AREA = 5    # Área de los álabes de la turbina en m^2
C_P = 0.4         # Coeficiente de potencia
C_CONFORT = 0.5   # Umbral de confort para el costo del mercado
BINS = 7          # Define cuántos intervalos se utilizan para discretizar las variables de potencia (renovables + demanda).
SOC_INITIAL = 0.6

os.makedirs("results", exist_ok=True)
os.makedirs("results/evolution", exist_ok=True)
os.makedirs("results/q_tables", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# -----------------------------------------------------
# Definimos el entorno
# -----------------------------------------------------
class MultiAgentEnv:
    """
    Entorno que:
    - Carga datos desde un CSV (irradancia, velocidad de viento, demanda, precio...).
    - Discretiza estas variables en bins para crear estados (solar_bins, wind_bins, etc.).
    - Avanza step a step por filas del dataset (o de manera aleatoria).
    - Puede gestionarse en un bucle de episodios.
    """

    def __init__(self, csv_filename, num_demand_bins=7, num_renewable_bins=7):
        """
        Parámetros:
          - num_*_bins: define cuántos intervalos se utilizan para discretizar cada variable.
        """

        self.renewable_power = 0
        self.renewable_power_idx = 0
        self.total_power = 0
        self.demand_power = 0        
        self.demand_power_idx = 0
        self.price = 0
        self.dif_power = 0
        self.total_power_idx = 0
        self.max_steps = 0
        self.num_demand_bins = num_demand_bins
        self.num_renewable_bins = num_renewable_bins
        
        # Cargamos el DataFrame con offsets
        offsets_dict = {"demand": 0, "price": 0, "solar_power": 0, "wind_power": 0}
        self.dataset = self._load_data(csv_filename, offsets_dict)

        self.max_steps = len(self.dataset)
        
        self.max_value = self.dataset.apply(pd.to_numeric, errors='coerce').max().max()
        
        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.demand_bins = np.linspace(0, self.max_value, num_demand_bins)
        self.renewable_bins = np.linspace(0, self.max_value, num_renewable_bins)

        # Estado inicial (discretizado)
        self.state = None

    def _load_data(self, filename: str, offsets: dict = None) -> pd.DataFrame:
        """
        Carga un archivo CSV usando pandas, aplica offsets (desplazamientos) a las columnas
        especificadas y coloca en cero (0) los valores negativos de la columna 'demand'.

        Parámetros:
        -----------
        filename : str
            Nombre del archivo CSV a cargar, buscado en la ruta indicada.
        offsets : dict, opcional
            Diccionario con pares {nombre_columna: valor_offset}.
            Por ejemplo: {"Load": 5, "PV_Power": -10}.
            Si es None o está vacío, no se aplican desplazamientos.

        Retorno:
        --------
        df : pd.DataFrame
            DataFrame con el contenido del archivo y las transformaciones indicadas.
        """
        # Ruta al archivo
        file_path = os.path.join(os.getcwd(), "assets", "datasets", filename)
        df = pd.read_csv(file_path, sep='[;,]', engine='python')

        # 1. Aplica offsets a las columnas indicadas
        if offsets is not None:
            for col, offset_value in offsets.items():
                if col in df.columns:
                    df[col] += offset_value
                else:
                    print(f"Advertencia: La columna '{col}' no existe en el DataFrame. No se aplicó offset.")

        # 2. Sustituir valores negativos en 'demand' con 0
        #    Ajusta el nombre de la columna 'demand' según tu archivo CSV.
        if 'demand' in df.columns:
            df['demand'] = df['demand'].clip(lower=0)

        return df

    def get_discretized_state(self, index):
        """
        Toma valores reales (irradancia, viento, demanda, precio, etc.) y los discretiza en bins,
        devolviendo una tupla como (idx_solar, idx_wind, idx_battery, idx_demand, idx_price).
        """
        row = self.dataset.iloc[index]
        
        self.demand_power = row["demand"]
        self.renewable_power = row["solar_power"] + row["wind_power"]
        self.price = row["price"]
        self.time = row["Datetime"]

        # Discretizamos
        self.demand_power_idx = self.digitize_clip(self.demand_power, self.demand_bins)
        self.renewable_power_idx = self.digitize_clip(self.renewable_power, self.renewable_bins)
        
        # Retornamos la tupla de estado discretizado
        return (self.demand_power_idx, self.renewable_power_idx)

    def digitize_clip(self, value: float, bins: np.ndarray) -> int:
        #discretización robusta y reutilizable
        idx = np.digitize([value], bins)[0] - 1
        idx = np.clip(idx, 0, len(bins)-2)        # evita -1 y último overflow
        return int(idx)

# -----------------------------------------------------
# Definimos la clase base de Agente con Q-Table
# -----------------------------------------------------
class BaseAgent:
    """
    Clase base para agentes con Q-table.
    """
    def __init__(self, name, actions, alpha=0.1, gamma=0.9, kappa=10, sigma=10, mu=10, nu=10, beta=10, isPower=True):
        self.name = name
        self.actions = actions
        self.action = 0
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.mu = mu
        self.nu = nu
        self.beta = beta
        self.isPower = isPower
        self.q_table = {}   
        self.current_power = 0.0
        self.idx = 0
    
    def choose_action(self, state, epsilon=0.1):
        """
        Selecciona acción con política epsilon-greedy.
        """

        if random.random() < epsilon:
            self.action = random.choice(self.actions)
        else:
            # Escoge la acción con Q máximo
            q_values = self.q_table.get(state, {a: 0.0 for a in self.actions})
            self.action = max(q_values, key=q_values.get)
        
        return self.action

    def update_q_table(self, state, action, reward, next_state):

        """
        Actualiza la Q-table según Q-Learning:
          Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        #print(state)
        q_values = self.q_table[state]
        current_q = q_values[action]
        
        # Si next_state no está en la Q-table (caso borde), asumimos Q=0
        next_q_values = self.q_table.get(next_state, {a: 0.0 for a in self.actions})
        max_next_q = max(next_q_values.values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def digitize_clip(self, value: float, bins: np.ndarray) -> int:
        #discretización robusta y reutilizable
        idx = np.digitize([value], bins)[0] - 1
        idx = np.clip(idx, 0, len(bins)-2)        # evita -1 y último overflow

        return int(idx)

# -----------------------------------------------------
# Agentes Especializados (Solar, Wind, Battery, Grid, Load)
#    Heredan de BaseAgent y añaden sus recompensas
# -----------------------------------------------------
class SolarAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv):
        super().__init__("solar", [0, 1], alpha=0.1, gamma=0.9, isPower=True)

        # ["idle", "produce"] -> [0, 1]

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.solar_power_bins = np.linspace(0, env.max_value, env.num_renewable_bins)
        self.solar_state_bins = [0, 1] # 0=idle, 1=producing
        self.solar_state = 0  
 
    def to_dataframe(self):
        """
        Convierte la Q-table en un DataFrame de pandas.
        """
        registros = []
        for estado, acciones in self.q_table.items():
            a, b, c, d = estado
            q0 = acciones.get(0, 0)
            q1 = acciones.get(1, 0)
            registros.append({
                'a (solar)': a,
                'b (panel)': b,
                'c (renov)': c,
                'd (dem)': d,
                'Q[0] (no produce)': q0,
                'Q[1] (produce)': q1,
                'mejor acción': 0 if q0 >= q1 else 1
            })
        return pd.DataFrame(registros)

    def get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
        row = env.dataset.iloc[index]
        self.current_power = row["solar_power"]

        # Discretizamos
        solar_power_idx = self.digitize_clip(self.current_power, self.solar_power_bins)
        solar_state_idx = self.digitize_clip(self.solar_state, self.solar_state_bins)
        self.idx = solar_power_idx
        
        # Retornamos la tupla de estado discretizado
        return (solar_power_idx, solar_state_idx, env.renewable_power_idx, env.demand_power_idx)

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
        
        # Para cada estado, creamos un diccionario de acción -> Q
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

    def calculate_reward(self, P_H, P_L, S_PV):
        """
        P_H: Potencia generada por las fuentes de energía renovables
        P_L: Demanda 
        S_PV: Estado del panel solar
        """
        
        if P_H <= P_L and S_PV == 1:
            return - self.kappa * (P_L - P_H)
        elif P_H > P_L and S_PV == 1:
            return self.sigma * (P_H - P_L)
        elif P_H > P_L and S_PV == 0:
            return - self.sigma * (P_H - P_L)
        elif P_H <= P_L and S_PV == 0:
            return self.sigma * (P_L - P_H)
        return -self.sigma

class WindAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv):
        super().__init__("wind", [0, 1], alpha=0.1, gamma=0.9, isPower=True)

        # ["idle", "produce"] -> [0, 1]

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.wind_power_bins = np.linspace(0, env.max_value, env.num_renewable_bins)
        self.wind_state_bins = [0, 1] # 0=idle, 1=producing
        self.wind_state = 0  
 
    def to_dataframe(self):
        """
        Convierte la Q-table en un DataFrame de pandas.
        """
        registros = []
        for estado, acciones in self.q_table.items():
            a, b, c, d = estado
            q0 = acciones.get(0, 0)
            q1 = acciones.get(1, 0)
            registros.append({
                'a (solar)': a,
                'b (panel)': b,
                'c (renov)': c,
                'd (dem)': d,
                'Q[0] (no produce)': q0,
                'Q[1] (produce)': q1,
                'mejor acción': 0 if q0 >= q1 else 1
            })
        return pd.DataFrame(registros)

    def get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_wind).
        """
        row = env.dataset.iloc[index]
        self.current_power = row["wind_power"]

        # Discretizamos
        wind_power_idx = self.digitize_clip(self.current_power, self.wind_power_bins)
        wind_state_idx = self.digitize_clip(self.wind_state, self.wind_state_bins)
        self.idx = wind_power_idx
        
        # Retornamos la tupla de estado discretizado
        return (wind_power_idx, wind_state_idx, env.renewable_power_idx, env.demand_power_idx)

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
        
        # Para cada estado, creamos un diccionario de acción -> Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def calculate_power(self, row):

        if self.isPower:
            return row["wind_power"]
        else:
            return 0.5 * RHO * BLADE_AREA * C_P * (row["wind speed"]**3)

    def calculate_reward(self, P_H, P_L, S_WD):
        """
        P_H: potencia generada por la turbina eólica
        P_L: demanda 
        S_WD: estado de la turbina eólica
        """
        
        if P_H <= P_L and S_WD == 1:
            return - self.kappa * (P_L - P_H)
        elif P_H > P_L and S_WD == 1:
            return self.sigma * (P_H - P_L)
        elif P_H > P_L and S_WD == 0:
            return - self.sigma * (P_H - P_L)
        if P_H <= P_L and S_WD == 0:
            return self.kappa * (P_L - P_H)
        return -self.kappa

class BatteryAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, capacity_ah= 30, num_battery_soc_bins=4):
        super().__init__("battery", [0, 1, 2], alpha=0.1, gamma=0.9)
        
        # ["idle", "charge", "discharge"] -> [0, 1, 2]
        
        """
        Inicializa la batería con una capacidad fija en Ah y un SOC inicial del 50%.
        :param capacity_ah: Capacidad de la batería en Amperios-hora (Ah).
        """
        self.capacity_ah = capacity_ah  # Capacidad fija en Ah
        self.soc = SOC_INITIAL  # Estado de carga inicial en %
        self.battery_power = 0.0  # Potencia en W
        self.battery_state = 0  # Estado inicial de operación
        self.battery_soc_idx = 0 # Estado SOC discretizado
        self.max_soc_idx = 2

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        self.battery_soc_bins = np.linspace(0, 1, num_battery_soc_bins)

    def update_soc(
            self,
            power_w: float,
            dt_h: float = 1.0,
            nominal_voltage: float = 48.0  # ← valor por defecto en voltios
        ) -> None:
        """
        Actualiza el estado de carga (SOC) de la batería.

        Parameters
        ----------
        power_w : float
            Potencia instantánea (W).  
            +  descarga  → SOC ↓  
            –  carga     → SOC ↑
        dt_h : float, default 1.0
            Duración del paso de simulación en horas.
        nominal_voltage : float, default 48.0
            Tensión nominal de la batería (V).  Se puede sobreescribir si
            se desea usar otro valor en alguna llamada concreta.
        """
        # Capacidad en Wh usando el voltaje nominal
        capacity_wh = self.capacity_ah * nominal_voltage

        # Energía transferida en el paso de tiempo
        energy_wh = power_w * dt_h
        capacity_wh_new = self.soc*capacity_wh - energy_wh

        # Integrar y saturar en [0, 1]
        new_soc = capacity_wh_new/capacity_wh
        self.soc = max(0.0, min(1.0, new_soc))

        # Índice discreto (opcional, para tu agente)
        self.battery_soc_idx = self.digitize_clip(
            self.soc, self.battery_soc_bins
        )

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

    def get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
       
        # Discretizamos
        self.battery_soc_idx = self.digitize_clip(self.soc, self.battery_soc_bins)
        self.idx = self.battery_soc_idx       
        
        # Retornamos la tupla de estado discretizado
        return (self.battery_soc_idx, self.battery_state, env.renewable_power_idx, env.demand_power_idx)

    def calculate_reward(self, P_T, P_L):
        """
        Calculates the reward based on grid and battery state.
        Battery states (battery_state):
        - 0: Standby
        - 1: Charging
        - 2: Discharging
        """
        power_surplus = P_T - P_L
        normalized_soc = self.battery_soc_idx / self.max_soc_idx

        # --- Positive Incentives (Rewards) ---

        # 1. Reward for discharging during a power deficit
        if power_surplus < 0 and self.battery_state == 2 and self.battery_soc_idx > 0:
            return self.kappa * normalized_soc * abs(power_surplus)

        # 2. Reward for charging during a power surplus
        if power_surplus > 0 and self.battery_state == 1 and self.battery_soc_idx < self.max_soc_idx:
            return self.nu * (1 - normalized_soc) * power_surplus

        # --- Negative Incentives (Penalties) ---

        # 3. Penalty for attempting an impossible action (e.g., discharging when empty)
        if (self.battery_state == 2 and self.battery_soc_idx == 0) or \
        (self.battery_state == 1 and self.battery_soc_idx == self.max_soc_idx):
            return -100 # High, fixed penalty

        # 4. Penalty for doing the opposite of what is needed
        if power_surplus > 0 and (self.battery_state == 2 or self.battery_state == 0):
            return -self.sigma * power_surplus

        if power_surplus < 0 and (self.battery_state == 1 or self.battery_state == 0):
            return -self.mu * abs(power_surplus)

        # 5. Small penalty for degradation or operating without need
        if self.battery_state != 0:
            return -0.1 # Small fixed cost for cycling

        # 6. Default reward (inactive state and balanced grid)
        return 0

class GridAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        super().__init__("grid", [0, 1], alpha=0.1, gamma=0.9)

        # ["idle", "produce"] -> [0, 1]
        self.grid_power = 0.0  # Potencia en W
        self.grid_state = 0  # Estado inicial de operación

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
                for c in range(len(env.renewable_bins)):
                    for d in range(len(env.demand_bins)):
                        states.append((a, b, c, d))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx_solar).
        """
       
        # Discretizamos
        battery_soc_idx = self.digitize_clip(self.ess.soc, self.ess.battery_soc_bins)
        self.idx = self.grid_state
        
        # Retornamos la tupla de estado discretizado
        return (self.grid_state, battery_soc_idx, env.renewable_power_idx, env.demand_power_idx)

    def calculate_reward(self, P_H, P_L, SOC, C_mercado):
        
        if SOC == 0 and P_H < P_L and self.grid_state == 1:
            return self.kappa / C_mercado
        elif SOC > 0 and P_H < P_L and self.grid_state == 1:
            return -self.mu * C_mercado
        elif (SOC > 0 or P_H > P_L) and self.grid_state == 1:
            return -self.sigma * C_mercado
        elif (SOC == 0 and P_H <= P_L) and self.grid_state == 0:
            return -self.nu * C_mercado
        else:
            return 0.0

class LoadAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        super().__init__("load", [0, 1], alpha=0.1, gamma=0.9)

        # Acciones: ["apagar", "encender"] -> [0, 1]
        self.comfort = 5
        self.ess =  ess
        self.load_state_bins = [0, 1] # 0=no encender, 1=encender
        self.load_state = 1
        self.controllable_demand = 0.0 #kW

    def get_discretized_state(self, env: MultiAgentEnv, index):
        """
        Toma valores reales y los discretiza en bins,
        devolviendo (idx).
        """
        row = env.dataset.iloc[index]
        self.current_power = row["demand"]

        # Discretizamos
        battery_soc_idx = self.digitize_clip(self.ess.soc, self.ess.battery_soc_bins)
        self.demand_power_idx = self.digitize_clip(self.current_power, env.demand_bins)
        self.idx = self.demand_power_idx
        
        # Retornamos la tupla de estado discretizado
        return (env.demand_power_idx, self.load_state, env.renewable_power_idx, battery_soc_idx)

    def initialize_q_table(self, env: MultiAgentEnv):
        """
        Crea la Q-table para todos los posibles estados discretizados.
        """
        states = []
        for a in range(len(env.demand_bins)):
            for b in range(len(self.load_state_bins)):
                for c in range(len(env.renewable_bins)):
                    for d in range(len(self.ess.battery_soc_bins)):
                        states.append((a, b, c, d))
        
        # Para cada estado, creamos un diccionario de acción->Q
        self.q_table = {
            state: {action: 0 for action in self.actions} 
            for state in states
        }

    def calculate_reward(self, action, P_T, P_L, SOC, C_mercado):
        if action == 1: #Encender
            if (P_T > P_L or SOC > 0):
                return self.kappa
            elif C_mercado < self.comfort:
                return self.kappa
            else:
                return - self.kappa
        elif action == 0: #Apagar
            if (P_T > P_L or SOC > 0):
                return - self.kappa
            elif C_mercado < self.comfort:
                return - self.kappa
            else:
                return self.kappa
        return 0.0

# -----------------------------------------------------
# Simulación de entrenamiento
# -----------------------------------------------------
class Simulation:
    def __init__(self, num_episodes=10, epsilon=1, learning=True, filename="Case1.csv"):
        self.num_episodes = num_episodes
        self.instant = {} 
        self.evolution = []
        self.df = pd.DataFrame()
        self.df_episode_metrics = pd.DataFrame()
        self.prev_q_tables = {} # Diccionario para almacenar la Q-table previa de cada agente
        
        # Creamos el entorno que carga el CSV y discretiza
        self.env = MultiAgentEnv(csv_filename=filename, num_demand_bins=BINS, num_renewable_bins=BINS)
        
        # Obtiene los puntos de manera automatica de la base de datos
        self.max_steps = self.env.max_steps
        
        bat_ag = BatteryAgent(self.env)

        # Definimos un conjunto de agentes
        self.agents = [
            SolarAgent(self.env),
            WindAgent(self.env),
            bat_ag,
            GridAgent(self.env, bat_ag),
            LoadAgent(self.env, bat_ag)
        ]
        
        # Parámetros de entrenamiento
        self.epsilon = epsilon  # Exploración \epsilon (0=explotación, 1=exploración)
        
        # Inicializamos Q-tables
        if learning:
            for agent in self.agents:
                agent.initialize_q_table(self.env)

    def step(self, index):
        self.env.get_discretized_state(index)
        agent_states = {}
        
        for agent in self.agents:
            agent_type = type(agent).__name__  # Obtiene el nombre de la clase del agente
            agent_states[agent_type] = agent.get_discretized_state(self.env, index)
            agent.idx = agent_states[agent_type][0]        
       
        return agent_states
        
    def run(self):

        for ep in range(self.num_episodes):
            
            # Save snapshot of previous Q-tables (only if not the first episode)
            if ep > 0:
                for agent in self.agents:
                    agent_key = type(agent).__name__
                    self.prev_q_tables[agent_key] = copy.deepcopy(agent.q_table)
            else:
                for agent in self.agents:
                    agent_key = type(agent).__name__
                    self.prev_q_tables[agent_key] = {state: {a:0.0 for a in agent.actions} for state in agent.q_table}
            
            # Initialization of the evaluation by episode
            self.evolution = []

            for agent in self.agents:
                # Stops the loop upon finding the battery agent
                if isinstance(agent, BatteryAgent):
                    agent.soc = SOC_INITIAL
                    break  

            for i in range(self.max_steps-1):
                
                # For each episode the power values ​​are initialized
                self.env.renewable_power = 0.0
                self.env.total_power = 0
                bat_power = 0.0
                grid_power = 0.0
                solar_power = 0.0
                wind_power = 0.0
                loadc_power = 0.0
                renewable_power_real  = 0.0
                renewable_power_real_idx = 0

                # We reset the environment and agents at the start of each episode.
                state = self.step(i)                

                self.instant["time"] = self.env.time                

                # Select the action
                for agent in self.agents:
                    if isinstance(agent, SolarAgent):
                        agent.choose_action(state['SolarAgent'], self.epsilon)

                        agent.solar_state = agent.action
                        solar_power = agent.current_power*agent.action

                        self.instant["solar"] = agent.current_power
                        self.instant["solar_state"] = agent.solar_state
                        self.instant["solar_discrete"] = agent.idx
                    elif isinstance(agent, WindAgent):
                        agent.choose_action(state['WindAgent'], self.epsilon)

                        agent.wind_state = agent.action
                        wind_power = agent.current_power*agent.action

                        self.instant["wind"] = agent.current_power
                        self.instant["wind_state"] = agent.wind_state
                        self.instant["wind_discrete"] = agent.idx
                    elif isinstance(agent, BatteryAgent):
                        agent.choose_action(state['BatteryAgent'], self.epsilon)
                        agent.battery_state = agent.action

                        # "charging" 
                        if agent.action == 1:
                            agent.battery_power = -abs(self.env.demand_power - (solar_power + wind_power))
                        # "discharging"
                        elif agent.action == 2:
                            agent.battery_power = abs(self.env.demand_power - (solar_power + wind_power))
                        # "idle"
                        else:                        
                            agent.battery_power = 0.0

                        bat_power = agent.battery_power

                        self.instant["bat"] = agent.battery_power
                        self.instant["bat_soc"] = agent.soc
                        self.instant["bat_soc_discrete"] = agent.battery_soc_idx
                        self.instant["bat_state"] = agent.battery_state
                        
                        agent.update_soc(power_w=agent.battery_power)
                    elif isinstance(agent, GridAgent):
                        agent.choose_action(state['GridAgent'], self.epsilon)
                        if agent.action == 1: # "sell"
                            agent.grid_state = 1 # "selling" 
                            agent.grid_power = abs(self.env.demand_power - (solar_power + wind_power) - bat_power)
                        else: 
                            agent.grid_state = 0 
                            agent.grid_power = 0
                        
                        grid_power = agent.grid_power

                        self.instant["grid"] = agent.grid_power
                        self.instant["grid_state"] = agent.grid_state
                        self.instant["grid_discrete"] = agent.idx
                    else:
                        agent.choose_action(state['LoadAgent'], self.epsilon)
                        agent.load_state = agent.action

                        # Turn ON
                        if agent.action == 1: 
                            agent.controllable_demand = 0
                        # Turn OFF
                        else:     
                            agent.controllable_demand = -15
                        
                        self.instant["load_state"] = agent.load_state
                        loadc_power = agent.controllable_demand

                # Calcula el balance de energia
                renewable_power_real = wind_power + solar_power
                self.env.total_power = renewable_power_real + bat_power + grid_power
                renewable_power_real_idx = self.env.digitize_clip(renewable_power_real, self.env.renewable_bins)
                self.dif_power = self.env.total_power - (self.env.demand_power + loadc_power)
                self.renewable_power_idx = self.env.digitize_clip(self.env.renewable_power, self.env.renewable_bins)
                self.total_power_idx = self.env.digitize_clip(self.env.total_power, self.env.renewable_bins)
                self.env.demand_power_idx = self.env.digitize_clip(self.env.demand_power + loadc_power, self.env.demand_bins)

                self.instant["renewable"] = self.env.renewable_power
                self.instant["renewable_discrete"] = self.renewable_power_idx
                self.instant["total"] = self.env.total_power
                self.instant["demand"] = self.env.demand_power + loadc_power
                self.instant["dif"] = self.dif_power
                self.instant["total_discrete"] = self.total_power_idx
                self.instant["demand_discrete"] = self.env.demand_power_idx
                self.instant["price"] = self.env.price

                # Ahora calculamos la recompensa individual por agente y actualizamos la Q-table                
                next_state = self.step(i + 1)  # Avanzamos el entorno un índice
                
                # Extrae el agente bateria
                battery_agent = None
                for agent in self.agents:
                    if isinstance(agent, BatteryAgent):
                        battery_agent = agent
                        break  # Detiene el bucle al encontrar el primer BatteryAgent

                for agent in self.agents:
                    agent_type = type(agent).__name__  # Obtiene el nombre de la clase del agente
                    #action = agent.choose_action(state[agent_type], self.epsilon)
                    
                    # Calculamos la recompensa según el tipo de agente
                    if isinstance(agent, SolarAgent):
                        reward = agent.calculate_reward(
                            P_H=agent.get_discretized_state(self.env, i)[0],
                            P_L=self.env.demand_power_idx,
                            S_PV=agent.action
                        )
                        self.instant["reward_solar"] = reward
                    elif isinstance(agent, WindAgent):
                        reward = agent.calculate_reward(
                            P_H=agent.get_discretized_state(self.env, i)[0],
                            P_L=self.env.demand_power_idx,
                            S_WD=agent.action
                        )
                        self.instant["reward_wind"] = reward
                    elif isinstance(agent, BatteryAgent):
                        
                        reward = agent.calculate_reward(
                                P_T=self.renewable_power_idx, 
                                P_L=self.env.demand_power_idx)

                        self.instant["bat_soc"] = agent.soc
                        self.instant["reward_bat"] = reward
                        battery_agent = agent
                    elif isinstance(agent, GridAgent):
                        reward = agent.calculate_reward(
                            P_H=renewable_power_real_idx,
                            P_L=self.env.demand_power_idx, 
                            SOC=battery_agent.battery_soc_idx,
                            C_mercado=self.env.price)
                        self.instant["reward_grid"] = reward
                        
                    elif isinstance(agent, LoadAgent):
                        reward = agent.calculate_reward(
                            action  =agent.action,
                            P_T=renewable_power_real_idx,
                            P_L=self.env.demand_power_idx,
                            SOC=battery_agent.battery_soc_idx,
                            C_mercado=self.env.price)
                        
                        self.instant["reward_demand"] = reward
                    else:
                        reward = 0.0
                    
                    # Actualizamos Q-table                    
                    agent.update_q_table(state[agent_type], agent.action, reward, next_state[agent_type])

                # Actualizamos el estado actual
                state = next_state
                self.evolution.append(copy.deepcopy(self.instant))
              
            # Cambia la relación de aprendizaje con cada episodio  
            if self.num_episodes > 1:
                self.epsilon = max(0.05, 1 - (ep / (self.num_episodes - 1)))
            else:
                self.epsilon = self.epsilon                
            
            self.df = pd.DataFrame(self.evolution)
            self.df.to_csv(f"results/evolution/learning_{ep}.csv", index=False)
            
            self.update_episode_metrics(ep, self.df)      
                        
            # Guarda Q-table actual en Excel
            for agent in self.agents:
                df_q = pd.DataFrame([
                    {
                        "State": str(state),
                        "Action": action,
                        "Q_value": q_value
                    }
                    for state, actions in agent.q_table.items()
                    for action, q_value in actions.items()
                ])

                # Nombre de archivo
                filename_q = f"results/q_tables/qtable_{agent.name}_ep{ep}.xlsx"

                df_q.to_excel(filename_q, index=False, engine="openpyxl")

            # Calcular métricas del episodio
            iae = self.calculate_iae()
            var_dif = self.df['dif'].var()

            # Calcular normas de diferencia Q
            q_norms = {
                type(agent).__name__: compute_q_diff_norm(agent.q_table, self.prev_q_tables[type(agent).__name__])
                for agent in self.agents
            }

            # Calcular recompensas promedio
            mean_rewards = {}
            for agent in self.agents:
                col_name = f"reward_{agent.name.lower()}"
                if col_name in self.df.columns:
                    mean_rewards[agent.name] = self.df[col_name].mean()
                else:
                    mean_rewards[agent.name] = 0.0

            # Registrar en DataFrame
            row = {
                "Episode": ep,
                "IAE": iae,
                "Var_dif": var_dif,
            }
            row.update({f"Q_Norm_{k}": v for k, v in q_norms.items()})
            row.update({f"Mean_Reward_{k}": v for k, v in mean_rewards.items()})

            self.df_episode_metrics = pd.concat(
                [self.df_episode_metrics, pd.DataFrame([row])],
                ignore_index=True
            )

            # Guardar a Excel UTF-8
            self.df_episode_metrics.to_excel(
                "results/metrics_episode.xlsx",
                index=False,
                engine="openpyxl"
            )
            
            print(f"Fin episodio {ep+1}/{self.num_episodes} con epsilon {self.epsilon}")

        # Graficas interactiva de potencia
        self.plot_data_interactive(
            df=self.df,
            columns_to_plot=["solar", "demand", "bat_soc", "grid", "wind", "price"],
            title="Environment variables",
            save_static_plot=True,
            static_format="svg",  # o "png", "pdf"
            static_filename="results/plots/env_plot"
        )              
        
        # Graficas interactiva de acciones y df
        self.plot_data_interactive(
            df=self.df,
            columns_to_plot=["dif"],
            title="Energy balance",
            save_static_plot=True,
            static_format="svg",  # o "png", "pdf"
            static_filename="results/plots/actions_plot"
        ) 

        self.plot_metric('Average Reward')  # Puedes usar 'Total Reward' u otra métrica
        
        # Graficar IAE
        plot_metric(
            self.df_episode_metrics,
            field="IAE",
            ylabel="Integral Absolute Error",
            filename_svg="results/plots/IAE_over_episodes.svg"
        )

        # Graficar Varianza de dif
        plot_metric(
            self.df_episode_metrics,
            field="Var_dif",
            ylabel="Variance of dif",
            filename_svg="results/plots/Var_dif_over_episodes.svg"
        )

        # Graficar normas Q por agente
        for agent in self.agents:
            agent_key = type(agent).__name__
            field_q = f"Q_Norm_{agent_key}"
            plot_metric(
                self.df_episode_metrics,
                field=field_q,
                ylabel=f"Q Norm Difference ({agent_key})",
                filename_svg=f"results/plots/Q_Norm_{agent_key}.svg"
            )
        
        
        # Calcular umbral IAE como mediana de primeros 50 episodios ±10%
        iae_median = self.df_episode_metrics[self.df_episode_metrics['Episode'] < 50]['IAE'].median()
        iae_threshold = iae_median * 1.10  # +10%

        # Verificar estabilidad
        stability = check_stability(self.df_episode_metrics, iae_threshold=iae_threshold)

        print("\n=== Stability Check ===")
        print(f"IAE Threshold: {iae_threshold:.3f}")
        print(f"Mean IAE (last 200 eps): {stability['IAE_mean']:.3f} -> {'OK' if stability['IAE_stable'] else 'NOT STABLE'}")
        print(f"Mean Var (last 200 eps): {stability['Var_mean']:.3f} -> {'OK' if stability['Var_stable'] else 'NOT STABLE'}")

        if stability['IAE_stable'] and stability['Var_stable']:
            print("SYSTEM DECLARED STABLE ✅")
        else:
            print("SYSTEM NOT STABLE ⚠️")        
       
        return self.agents

    def calculate_ise(self) -> float:
        """
        Calcula el ISE (Integral Square Error) sobre la columna 'dif'.

        :return: Valor de ISE.
        """
        ise = (self.df['dif'] ** 2).sum()
        return ise

    def calculate_mean(self) -> float:
        """
        Calcula el ISE (Integral Square Error) sobre la columna 'dif'.

        :return: Valor de ISE.
        """
        mean= self.df['dif'].mean()
        return mean

    def calculate_iae(self) -> float:
        """
        Calcula el IAE (Integral Absolute Error) sobre la columna 'dif'.

        :return: Valor de IAE.
        """
        iae = self.df['dif'].abs().sum()
        return iae

    def calculate_rep(self) -> float:
        """
        Calcula el REP (Renewable Energy Penetration), porcentaje de energía renovable sobre la total.

        :return: Valor de REP como porcentaje.
        """
        total_renewable_energy = self.df['solar_state'].sum()
        total_energy = self.df.shape[0]
        if total_energy == 0:
            return 0.0  # evitar división por cero
        rep = (total_renewable_energy / total_energy) * 100
        return rep

    def calculate_grid(self) -> float:
        """
        Calcula el REP (Renewable Energy Penetration), porcentaje de energía renovable sobre la total.

        :return: Valor de REP como porcentaje.
        """
        total_grid_energy = self.df['grid_state'].sum()
        total_energy = self.df.shape[0]
        if total_energy == 0:
            return 0.0  # evitar división por cero
        rep = (total_grid_energy / total_energy) * 100
        return rep

    def show_performance_metrics(self):
        """
        Muestra una tabla con las métricas de rendimiento calculadas: ISE, IAE y REP.
        """
        results = [
            ["MEAN (Mean)", f"{self.calculate_mean():.3f}"],
            ["ISE (Integral Square Error)", f"{self.calculate_ise():.3f}"],
            ["IAE (Integral Absolute Error)", f"{self.calculate_iae():.3f}"],
            ["REP (Renewable Energy Penetration)", f"{self.calculate_rep():.2f}%"],
            ["GEP (Grid Energy Penetration)", f"{self.calculate_grid():.2f}%"]
        ]
        print(tabulate(results, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    
    def compute_reward_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = []

        for agent in ['solar', 'bat', 'grid', 'wind', 'demand']:
            rewards = df[f'reward_{agent}']
            metrics = {
                'Agent': agent,
                'Rewards (count)': (rewards > 0).sum(),
                'Penalties (count)': (rewards < 0).sum(),
                'Total Reward': rewards.sum(),
                'Average Reward': rewards.mean(),
                'Max Reward': rewards.max(),
                'Min Reward': rewards.min(),
                'Variance': rewards.var(),
                'Interpretation': ''
            }

            if metrics['Average Reward'] > 0.1:
                metrics['Interpretation'] = 'Consistently positive learning'
            elif metrics['Average Reward'] > 0:
                metrics['Interpretation'] = 'Acceptable learning'
            else:
                metrics['Interpretation'] = 'High level of penalties, review needed'

            summary.append(metrics)

        return pd.DataFrame(summary)

    def update_episode_metrics(self, episode: int, df_episode: pd.DataFrame):
        df_metrics = self.compute_reward_metrics(df_episode)
        df_metrics['Episode'] = episode
        self.df_episode_metrics = pd.concat([self.df_episode_metrics, df_metrics], ignore_index=True)

    def plot_metric(self, metric_field='Average Reward', output_format='svg', filename='results/plots/metric_plot'):
        """
        Genera una gráfica de métricas por agente y la guarda como archivo vectorial o de alta resolución.
        
        Parámetros:
            metric_field (str): Nombre del campo de métrica a graficar.
            output_format (str): 'svg' para vectorial, 'png' para imagen de alta resolución.
            filename (str): Nombre base del archivo sin extensión.
        """
        plt.figure(figsize=(10, 6))
        for agent in self.df_episode_metrics['Agent'].unique():
            agent_df = self.df_episode_metrics[self.df_episode_metrics['Agent'] == agent]
            plt.plot(agent_df['Episode'], agent_df[metric_field], label=f'{agent} - {metric_field}')

        plt.xlabel("Episode")
        plt.ylabel(metric_field)
        plt.title(f"{metric_field} per Episode by Agent")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Guardar con alta calidad
        if output_format == 'svg':
            plt.savefig(f"{filename}.svg", format='svg')
        elif output_format == 'png':
            plt.savefig(f"{filename}.png", format='png', dpi=300)  # Alta resolución
        else:
            raise ValueError("output_format debe ser 'svg' o 'png'")

        plt.close()  # Cierra la figura para liberar memoria

        return self.df_episode_metrics     

    def plot_data_interactive(
            self,
            df: pd.DataFrame,                     # ahora entra un DataFrame
            columns_to_plot: list[str] | None = None,
            title: str = "Environment variables",
            save_static_plot: bool = False,
            static_format: str = "svg",
            static_filename: str = "interactive_plot_export",
            soc_keyword: str = "soc",             # patrón que detecta columnas SOC
            soc_scale: float = 100.0              # 0-1  → 0-100 %
        ):
        # ---------- 1. Validaciones -------------------------------
        if df.empty:
            print("No hay datos para graficar (DataFrame vacío).")
            return

        if columns_to_plot is None:
            columns_to_plot = df.columns.tolist()

        valid_cols = [c for c in columns_to_plot if c in df.columns]
        if not valid_cols:
            print("Las columnas indicadas no existen en el DataFrame.")
            return

        # ---------- 2. Clasificación ------------------------------
        soc_cols  = [c for c in valid_cols if soc_keyword.lower() in c.lower()]
        prim_cols = [c for c in valid_cols if c not in soc_cols]

        # ---------- 3. Figura y ejes ------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx() if soc_cols else None
        if ax2:
            ax2.patch.set_visible(False)          # fondo transparente

        base_colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle1 = cycle(base_colors)
        color_cycle2 = cycle(base_colors[1:]+base_colors[:1])

        lines, labels = [], []

        # ---------- 4. Potencias (eje primario) -------------------
        for col in prim_cols:
            l, = ax1.plot(df[col], label=col,
                        color=next(color_cycle1), zorder=3)
            lines.append(l); labels.append(col)

        # ---------- 5. SOC (eje secundario) ----------------------
        if ax2:
            for col in soc_cols:
                data  = df[col] * soc_scale
                label = f"{col} [%]"
                l, = ax2.plot(data, '--', lw=2,
                            color=next(color_cycle2), label=label, zorder=4)
                lines.append(l); labels.append(label)
            ax2.set_ylabel("State of Charge [%]")
            ax2.set_ylim(0, 100)

        # ---------- 6. Estilo ------------------------------------
        ax1.set_title(title)
        ax1.set_xlabel("Hours")
        ax1.set_ylabel("Power [kW]")
        ax1.grid(True, zorder=0)
        ax1.legend(lines, labels, loc="upper right")

        # ---------- 7. Check-boxes -------------------------------
        rax = fig.add_axes([0.05, 0.4, 0.17, 0.2])
        checks = CheckButtons(rax, labels, [True]*len(labels))

        def toggle(label: str):
            idx = labels.index(label)
            lines[idx].set_visible(not lines[idx].get_visible())
            plt.draw()

        checks.on_clicked(toggle)

        # ---------- 8. Exportación estática (sin check-boxes) ----
        if save_static_plot:
            valid_formats = {"svg", "png", "pdf"}
            if static_format.lower() not in valid_formats:
                raise ValueError(f"Formato '{static_format}' no soportado ({valid_formats})")

            # Ocultamos temporalmente el eje de los check-boxes
            rax.set_visible(False)
            dpi = 300 if static_format.lower() == "png" else None
            fig.savefig(f"{static_filename}.{static_format}",
                        format=static_format, dpi=dpi)
            print(f"Gráfico guardado como {static_filename}.{static_format}")
            # Volvemos a mostrarlo para la vista interactiva
            rax.set_visible(True)

        plt.show()

# -----------------------------------------------------
# Punto de entrada principal
# -----------------------------------------------------
if __name__ == "__main__":
    
    # Limpia los archivos contenidos en los directorios temporales
    clear_results_directories()

    sim1 = Simulation(num_episodes=1000, epsilon=1, learning=True, filename="Case1.csv")
    sim1.run()
    sim1.show_performance_metrics()

    # Graficas con los resultados de la interaccion cuando los agentes hayan completado el aprendizaje
    df_raw = load_latest_evolution_csv()
    df_clean = process_evolution_data(df_raw)
    plot_coordination(df_clean)