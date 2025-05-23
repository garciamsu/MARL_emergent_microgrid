import os
import sys
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import copy
from tabulate import tabulate


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

    def __init__(self, csv_filename="Case1_energy_data_with_pv_power.csv", num_demand_bins=7, num_renewable_bins=7):
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
        
        # Cargamos el DataFrame con offsets
        offsets_dict = {"demand": -9000, "price": 0, "solar_power": 0, "wind_power": 0}
        self.dataset = self._load_data(csv_filename, offsets_dict)
        
        # Graficas interactiva
        self.plot_data_interactive(csv_filename,
                                  offsets_dict,
                                  columns_to_plot=["demand", "price", "solar_power", "wind_power"],
                                  title="Gráfico imteractivo de variables de entorno")

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
        df = pd.read_csv(file_path, sep=';', engine='python')

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

    def plot_data_interactive(self, filename: str, offsets_dict, columns_to_plot=None, title: str = "Gráfico Interactivo"):
        """
        Carga los datos desde un archivo CSV y los grafica en una ventana interactiva
        que permite mostrar/ocultar cada traza usando checkboxes.

        Parámetros
        ----------
        filename : str
            Nombre del archivo CSV a cargar (se buscará en la ruta definida en _load_data).
        columns_to_plot : list, opcional
            Lista de columnas a graficar. Si es None, se graficarán todas las columnas del DataFrame.
        title : str, opcional
            Título a mostrar en el gráfico principal.
        """
        df = self._load_data(filename, offsets_dict)

        if df.empty:
            print("No hay datos para graficar (DataFrame vacío).")
            return

        # Determina qué columnas se van a graficar
        if not columns_to_plot:
            columns_to_plot = df.columns.tolist()

        # Verificamos que las columnas existan realmente en el DataFrame
        valid_columns = [col for col in columns_to_plot if col in df.columns]
        invalid_columns = set(columns_to_plot) - set(valid_columns)

        if invalid_columns:
            print("Advertencia: Las siguientes columnas no existen en el DataFrame:")
            for col in invalid_columns:
                print(f"  - {col}")

        if not valid_columns:
            print("No se encontraron columnas válidas para graficar.")
            return

        # Preparar la figura principal y el área para los checkboxes
        fig = plt.figure(figsize=(8, 6))
        # Área para el gráfico principal (izquierda y parte central de la ventana)
        ax = fig.add_axes([0.25, 0.1, 0.7, 0.8])

        # Graficar cada columna y guardar la referencia a la línea
        lines = []
        for col in valid_columns:
            line, = ax.plot(df[col], label=col)  # la coma tras 'line,' es necesaria para desempaquetar la tupla
            lines.append(line)

        ax.set_title(title)
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valores")
        ax.legend()

        # Área para el panel de checkboxes (en la parte izquierda de la ventana)
        rax = fig.add_axes([0.05, 0.4, 0.15, 0.15])  # [left, bottom, width, height] en fracción de la figura
        labels = valid_columns
        visibility = [True] * len(labels)

        # Crear los checkboxes a partir de las etiquetas de las columnas
        check = CheckButtons(rax, labels, visibility)

        # Función para mostrar/ocultar la línea correspondiente cuando se activa el checkbox
        def toggle_line(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()  # Se actualiza el gráfico

        # Asignar la función de callback al evento "on_clicked"
        check.on_clicked(toggle_line)

        # Mostrar la ventana con el gráfico interactivo
        plt.show()

    def _get_discretized_state(self, index):
        """
        Toma valores reales (irradancia, viento, demanda, precio, etc.) y los discretiza en bins,
        devolviendo una tupla como (idx_solar, idx_wind, idx_battery, idx_demand, idx_price).
        """
        row = self.dataset.iloc[index]
        
        self.demand_power = row["demand"]
        self.price = row["price"]
        self.time = row["Datetime"]

        # Discretizamos
        self.demand_power_idx = np.digitize([row["demand"]], self.demand_bins)[0] - 1
        self.renewable_power_idx = np.digitize([self.renewable_power], self.renewable_bins)[0] - 1
        
        # Retornamos la tupla de estado discretizado
        return (self.demand_power_idx, self.renewable_power_idx)

# -----------------------------------------------------
# Definimos la clase base de Agente con Q-Table
# -----------------------------------------------------
class BaseAgent:
    """
    Clase base para agentes con Q-table.
    """
    def __init__(self, name, actions, alpha=0.1, gamma=0.9, kappa=0.1, sigma=0.1, mu=0.1, nu=0.1, beta=0.1, isPower=True):
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

        #print("------------- state")
        #print(state)
        #print("------------- action")
        #print(action)
        #print("------------- reward")
        #print(reward) 
        #print("------------- next_state")
        #print(next_state)        
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

# -----------------------------------------------------
# Agentes Especializados (Solar, Wind, Battery, Grid, Load)
#    Heredan de BaseAgent y añaden sus recompensas
# -----------------------------------------------------
class SolarAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, num_solar_bins=7):
        super().__init__("solar", [0, 1], alpha=0.1, gamma=0.9, isPower=True)

        # ["idle", "produce"] -> [0, 1]

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        # Ajusta los rangos según tu dataset real
        self.solar_power_bins = np.linspace(0, env.max_value, num_solar_bins)
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

    def show_heatmap(self, b_fixed=0, c_fixed=0, action=1, max_a=6, max_d=6):
        """
        Muestra un heatmap de Q-values para una acción dada, fijando b y c.
        """
        matriz = np.zeros((max_a + 1, max_d + 1))
        for (a, b, c, d), acciones in self.q_table.items():
            if b == b_fixed and c == c_fixed:
                matriz[a][d] = acciones.get(action, 0)
        plt.figure(figsize=(8, 6))
        plt.imshow(matriz, cmap='viridis', origin='lower')
        plt.colorbar(label=f'Q[{action}]')
        plt.xlabel('Demanda (d)')
        plt.ylabel('Potencia solar (a)')
        plt.title(f'Heatmap Q[{action}] para b={b_fixed}, c={c_fixed}')
        plt.xticks(range(max_d + 1))
        plt.yticks(range(max_a + 1))
        plt.grid(False)
        plt.show()

    def show_policy_map(self, b_fixed=0, c_fixed=0, max_a=6, max_d=6):
        """
        Muestra un mapa de política greedy (mejor acción por estado).
        """
        matriz = np.full((max_a + 1, max_d + 1), -1)
        for (a, b, c, d), acciones in self.q_table.items():
            if b == b_fixed and c == c_fixed:
                mejor_accion = max(acciones, key=acciones.get)
                matriz[a][d] = mejor_accion
        plt.figure(figsize=(8, 6))
        plt.imshow(matriz, cmap='coolwarm', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='Mejor acción (0: no produce, 1: produce)')
        plt.xlabel('Demanda (d)')
        plt.ylabel('Potencia solar (a)')
        plt.title(f'Mapa de política greedy para b={b_fixed}, c={c_fixed}')
        plt.xticks(range(max_d + 1))
        plt.yticks(range(max_a + 1))
        plt.grid(False)
        plt.show()

    def show_q_histogram(self, action=1):
        """
        Muestra un histograma de todos los Q-values para una acción específica.
        """
        valores_q = [acciones.get(action, 0) for acciones in self.q_table.values()]
        plt.figure(figsize=(8, 5))
        plt.hist(valores_q, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de Q[{action}]')
        plt.xlabel('Valor Q')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.show()
  
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
        
        #print(self.q_table)
        #print(self.actions)
        #print(states)       

    def calculate_power(self, row):

        if self.isPower:
            return row["solar_power"]
        else:
            return ETA * SOLAR_AREA * row["irradiance"] * (1 - 0.005*(T_AMBIENT + 25))

        # Ejemplo muy simplificado

    def calculate_reward(self, P_H, P_L, S_PV):
        """
        P_H: potencia generada por el panel local
        P_L: demanda local (o parte de la demanda)
        SOC: estado de la batería
        """
        
        if P_H <= P_L and S_PV == 1:
            return - self.kappa * (P_L - P_H)
        elif P_H > P_L and S_PV == 1:
            return self.sigma * (P_H - P_L)
        elif P_H > P_L and S_PV == 0:
            return - self.sigma * (P_H - P_L)
        return 0.0

class WindAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, num_wind_bins=7):
        super().__init__("wind", [0, 1], alpha=0.1, gamma=0.9, isPower=True)
        
        # ["idle", "produce"] -> [0, 1]

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
        
        # Retornamos la tupla de estado discretizado
        return (wind_power_idx, wind_state_idx, env.renewable_power_idx, env.demand_power_idx)

    def calculate_power(self, row):

        if self.isPower:
            return row["wind_power"]
        else:
            return 0.5 * RHO * BLADE_AREA * C_P * (row["wind speed"]**3)

    def calculate_reward(self, P_H, P_L, S_WD):
        """
        P_H: potencia generada por el panel local
        P_L: demanda local (o parte de la demanda)
        SOC: estado de la batería
        """
        
        if P_H <= P_L and S_PV == 1:
            return - self.kappa * (P_L - P_H)
        elif P_H > P_L and S_PV == 1:
            return self.sigma * (P_H - P_L)
        elif P_H > P_L and S_PV == 0:
            return - self.sigma * (P_H - P_L)
        return 0.0

class BatteryAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, capacity_ah= 50000000, num_battery_soc_bins=4):
        super().__init__("battery", [0, 1, 2], alpha=0.1, gamma=0.9)
        
        # ["idle", "charge", "discharge"] -> [0, 1, 2]
        
        """
        Inicializa la batería con una capacidad fija en Ah y un SOC inicial del 50%.
        :param capacity_ah: Capacidad de la batería en Amperios-hora (Ah).
        """
        self.capacity_ah = capacity_ah  # Capacidad fija en Ah
        self.soc = 0.5  # Estado de carga inicial en %
        print(self.soc)
        self.battery_power = 0.0  # Potencia en W
        self.battery_state = 0  # Estado inicial de operación
        self.battery_power_idx = 0 # Estado SOC discretizado

        # Discretizacion por cuantizacion uniforme
        # Definimos los "bins" para discretizar cada variable de interés
        self.battery_soc_bins = np.linspace(0, 1, num_battery_soc_bins)

    def update_soc(self, power_w: float):
        """
        Actualiza el SOC basado en la potencia suministrada o extraída en una unidad de tiempo de 1 hora.
        
        :param power_w: Potencia en Watts (positiva si está cargando, negativa si está descargando).
        """
        # Convertimos potencia en energía transferida en Wh
        energy_wh = power_w  # Asumimos 1 hora de tiempo fijo
        
        # Convertimos la energía en Wh a Ah usando la capacidad de la batería
        delta_soc = (energy_wh / (self.capacity_ah))   # Expresado de 0 a 1
        
        # Actualizamos el SOC asegurándonos de que se mantenga en los límites de 0 a 1
        self.soc = max(0.0, min(1.0, self.soc - delta_soc))
        self.battery_power_idx = np.digitize([self.soc], self.battery_soc_bins)[0] - 1
    
    def get_soc(self) -> float:
        """
        Retorna el estado de carga actual en porcentaje.
        
        :return: SOC en porcentaje (0 - 1).
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
        self.battery_power_idx = np.digitize([self.soc], self.battery_soc_bins)[0] - 1
        
        # Retornamos la tupla de estado discretizado
        return (self.battery_power_idx, self.battery_state, env.renewable_power_idx, env.demand_power_idx)

    def calculate_reward(self, P_T, P_L):
        if self.battery_power_idx > 0  and P_T < P_L and self.battery_state == 2:
            return self.kappa * self.battery_power_idx * (P_L - P_T)
        elif self.battery_power_idx == 0 and P_T > P_L and self.battery_state == 2:
            return -self.sigma * (2 - self.battery_power_idx)  * (P_T - P_L)
        elif self.battery_power_idx == 0 and P_T < P_L and self.battery_state == 2:
            return -self.mu * (2 - self.battery_power_idx) * (P_L - P_T)
        if self.battery_power_idx <= 2 and P_T > P_L and self.battery_state == 1:
            return self.nu * (2 - self.battery_power_idx) * (P_T - P_L)
        elif self.battery_power_idx < 2 and P_T <= P_L and self.battery_state == 1:
            return -self.beta * self.battery_power_idx * (P_L - P_T)
        return 0.0  

class GridAgent(BaseAgent):
    def __init__(self, env: MultiAgentEnv, ess: BatteryAgent):
        super().__init__("grid", [0, 1], alpha=0.1, gamma=0.9)

        # ["idle", "sell"] -> [0, 1]
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
        
        # Retornamos la tupla de estado discretizado
        return (self.grid_state, battery_power_idx, self.ess.battery_state, env.renewable_power_idx, env.demand_power_idx)

    def calculate_reward(self, P_H, P_L, SOC, C_mercado):
        
        if SOC == 0 and P_H < P_L and self.grid_state == 1:
            return self.kappa / C_mercado
        elif SOC > 0 and P_H < P_L and self.grid_state == 1:
            return -self.mu * C_mercado
        elif (SOC > 0 or P_H > P_L) and self.grid_state == 1:
            return -self.sigma * C_mercado
        elif SOC == 0 and P_H <= P_L and self.grid_state == 0:
            return -self.nu * C_mercado
        else:
            return 0.0

class LoadAgent(BaseAgent):
    def __init__(self):
        super().__init__("load", [0, 1], alpha=0.1, gamma=0.9)

        # ["idle", "consume"] -> [0, 1]

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
        elif action == 0:
            if P_T > P_L or SOC >= 0.5:
                return -1.0 * P_T * C_mercado
        return 0.0

# -----------------------------------------------------
# Simulación de entrenamiento
# -----------------------------------------------------
class Simulation:
    def __init__(self, num_episodes=10, max_steps=5, epsilon=1, learning=True):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.instant = {} 
        self.evolution = []
        self.df = pd.DataFrame()
        
        # Creamos el entorno que carga el CSV y discretiza
        self.env = MultiAgentEnv(csv_filename="Case1_energy_data_with_pv_power.csv", num_demand_bins=7)
        
        bat_ag = BatteryAgent(self.env)

        # Definimos un conjunto de agentes
        self.agents = [
            SolarAgent(self.env),
            #WindAgent(self.env),
            bat_ag,
            GridAgent(self.env, bat_ag)
        ]
        
        # Parámetros de entrenamiento
        self.epsilon = epsilon  # Exploración \epsilon (0=explotación, 1=exploración)
        
        # Inicializamos Q-tables
        if learning:
            for agent in self.agents:
                agent.initialize_q_table(self.env)

    def step(self, index):
        self.env._get_discretized_state(index)
        agent_states = {}
        
        for agent in self.agents:
            agent_type = type(agent).__name__  # Obtiene el nombre de la clase del agente
            agent_states[agent_type] = agent._get_discretized_state(self.env, index)
        
        return agent_states
        
    def run(self):

        for ep in range(self.num_episodes):

            for i in range(self.max_steps-1):
                
                # Para cada episodio se inicializa los valores de potencia
                self.env.renewable_power = 0.0
                self.env.total_power = 0
                bat_power = 0.0
                grid_power = 0.0

                # Reseteamos entorno y los agentes al inicio de cada episodio
                state = self.step(i)
                
                
                self.instant["demand"] = self.env.demand_power
                self.instant["demand_discrete"] = self.env.demand_power_idx
                self.instant["price"] = self.env.price
                self.instant["time"] = self.env.time

                # Selecciona la acción 
                for agent in self.agents:
                    if isinstance(agent, SolarAgent):
                        # Escoger acción
                        agent.choose_action(state['SolarAgent'], self.epsilon)

                        if agent.action == 1: #"produce"
                            agent.solar_state = 1
                            self.env.renewable_power += agent.solar_state*agent.current_power
                        else:
                            agent.solar_state = 0
                            self.env.renewable_power += 0.0

                        self.instant["solar"] = agent.current_power
                        self.instant["solar_state"] = agent.solar_state
                    elif isinstance(agent, WindAgent):
                        agent.choose_action(state['WindAgent'], self.epsilon)
                        if agent.action == 1: #"produce"
                            agent.wind_state = 1
                            self.env.renewable_power += agent.wind_state*agent.current_power
                        else:
                            agent.wind_state = 0
                            self.env.renewable_power += 0.0

                        self.instant["wind"] = agent.current_power
                        self.instant["wind_state"] = agent.wind_state
                    elif isinstance(agent, BatteryAgent):
                        agent.choose_action(state['BatteryAgent'], self.epsilon)

                        if agent.action == 1:
                            agent.battery_state = 1 # "charging" 
                            agent.battery_power = self.env.demand_power - self.env.renewable_power
                            #agent.battery_power = -99999
                        elif agent.action == 2:
                            agent.battery_state = 2  # "discharging" 
                            agent.battery_power = self.env.demand_power - self.env.renewable_power
                            #agent.battery_power = 99999
                        else:
                            agent.battery_state = 0 
                            agent.battery_power = 0.0

                        bat_power = agent.battery_power

                        self.instant["battery"] = agent.battery_power
                        self.instant["soc"] = agent.get_soc()
                        self.instant["soc_discrete"] = agent.battery_power_idx
                        self.instant["bat_state"] = agent.battery_state
                    elif isinstance(agent, GridAgent):
                        agent.choose_action(state['GridAgent'], self.epsilon)
                        if agent.action == 1: # "sell"
                            agent.grid_state = 1 # "selling" 
                            agent.grid_power = self.env.demand_power - self.env.renewable_power - bat_power
                            #agent.grid_power = 999999
                        else: 
                            agent.grid_state = 0 
                            agent.grid_power = 0
                        
                        grid_power = agent.grid_power

                        self.instant["grid"] = agent.grid_power
                        self.instant["grid_state"] = agent.grid_state
                    else:
                        # LoadAgent no generan en este ejemplo
                        _ = agent.choose_action(state["LoadAgent"], self.epsilon)               

                self.env.total_power = self.env.renewable_power + bat_power + grid_power
                self.dif_power = self.env.total_power - self.env.demand_power
                self.renewable_power_idx = np.digitize([self.env.renewable_power], self.env.renewable_bins)[0] - 1
                self.total_power_idx = np.digitize([self.env.total_power], self.env.renewable_bins)[0] - 1

                self.instant["renewable"] = self.env.renewable_power
                self.instant["renewable_discrete"] = self.renewable_power_idx
                self.instant["total"] = self.env.total_power
                self.instant["total_discrete"] = self.total_power_idx
                self.instant["dif"] = self.dif_power

                # Ahora calculamos la recompensa individual por agente
                # y actualizamos la Q-table                
                next_state = self.step(i + 1)  # Avanzamos el entorno un índice
                
                # Extrae el agente bateria
                battery_agent = None
                for agent in self.agents:
                    if isinstance(agent, BatteryAgent):
                        battery_agent = agent
                        break  # Detiene el bucle al encontrar el primer BatteryAgent

                for agent in self.agents:
                    # Recuperamos la acción que tomó este agente en este step
                    # Para un enfoque riguroso, cada agente debería almacenar su "acción" actual,
                    # aquí simplificamos volviendo a elegir la misma acción con choose_action(...) 
                    # o se podría guardar en un diccionario {agent:action} en el bucle anterior
                    # para no volver a generarla.
                    
                    # -- EJEMPLO: asumiremos la misma acción que generamos antes (guardándola):
                    # Para hacerlo rápido, repetimos la llamada (no es lo ideal).
                    
                    agent_type = type(agent).__name__  # Obtiene el nombre de la clase del agente
                    #action = agent.choose_action(state[agent_type], self.epsilon)
                    
                    # Calculamos la recompensa según el tipo de agente
                    if isinstance(agent, SolarAgent):
                        reward = agent.calculate_reward(
                            P_H=self.env.renewable_power_idx,
                            P_L=self.env.demand_power_idx,
                            S_PV=agent.action
                        )
                        self.instant["reward_solar"] = reward
                    elif isinstance(agent, WindAgent):
                        reward = agent.calculate_reward(
                            P_H=self.env.renewable_power_idx, 
                            P_L=self.env.demand_power_idx,
                            S_WD=agent.action
                        )
                        self.instant["reward_wind"] = reward
                    elif isinstance(agent, BatteryAgent):

                        reward = agent.calculate_reward(
                                P_T=self.env.total_power_idx, 
                                P_L=self.env.demand_power_idx)

                        agent.update_soc(agent.battery_power)
                        #battery_agent = agent
                        self.instant["soc"] = agent.get_soc()
                        self.instant["reward_bat"] = reward
                    elif isinstance(agent, GridAgent):
                        reward = agent.calculate_reward(
                            P_H=self.env.renewable_power_idx,
                            P_L=self.env.demand_power_idx, 
                            SOC=battery_agent.battery_power_idx,
                            C_mercado=self.env.price)
                        self.instant["reward_grid"] = reward
                        
                    elif isinstance(agent, LoadAgent):
                        reward = agent.calculate_reward(
                            agent.action,
                            P_H=self.env.renewable_power,
                            P_L=self.env.demand_power,
                            SOC=battery_agent.get_soc(),
                            C_mercado=self.env.price)
                    else:
                        reward = 0.0
                    
                    # Actualizamos Q-table                    
                    agent.update_q_table(state[agent_type], agent.action, reward, next_state[agent_type])

                #print(self.instant)
                #print(i)
                # Actualizamos el estado actual
                state = next_state
                self.evolution.append(copy.deepcopy(self.instant))
            
            # Visualizamos las Q-tables
            for agent in self.agents:
                if isinstance(agent, SolarAgent):
                    # Mostrar heatmap para acción "producir" (1)
                    agent.show_heatmap(b_fixed=0, c_fixed=0, action=1)
                    # Mostrar mapa de política greedy
                    agent.show_policy_map(b_fixed=0, c_fixed=0)
                    # Mostrar histograma de Q[1] (producir)
                    agent.show_q_histogram(action=1)
            
            print(f"Fin episodio {ep+1}/{self.num_episodes}")

        self.df = pd.DataFrame(self.evolution)
        self.df.to_csv("evolution_learning.csv", index=False)
        
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
        total_renewable_energy = self.df['renewable'].sum()
        total_energy = self.df['total'].sum()
        if total_energy == 0:
            return 0.0  # evitar división por cero
        rep = (total_renewable_energy / total_energy) * 100
        return rep

    def calculate_grid(self) -> float:
        """
        Calcula el REP (Renewable Energy Penetration), porcentaje de energía renovable sobre la total.

        :return: Valor de REP como porcentaje.
        """
        total_grid_energy = self.df['grid'].sum()
        total_energy = self.df['total'].sum()
        if total_energy == 0:
            return 0.0  # evitar división por cero
        rep = (total_grid_energy / total_energy) * 100
        return rep

    def calculate_bat(self) -> float:
        """
        Calcula el REP (Renewable Energy Penetration), porcentaje de energía renovable sobre la total.

        :return: Valor de REP como porcentaje.
        """
        total_battery_energy = self.df['battery'].sum()
        total_energy = self.df['total'].sum()
        if total_energy == 0:
            return 0.0  # evitar división por cero
        rep = (total_battery_energy / total_energy) * 100
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
            ["GEP (Grid Energy Penetration)", f"{self.calculate_grid():.2f}%"],
            ["BEP (Battery Energy Penetration)", f"{self.calculate_bat():.2f}%"]
        ]
        print(tabulate(results, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
# -----------------------------------------------------
# Punto de entrada principal
# -----------------------------------------------------
if __name__ == "__main__":
    
    sim1 = Simulation(num_episodes=1, max_steps=8762, epsilon=1, learning=True)
    sim1.run()

    sim2 = Simulation(num_episodes=1, max_steps=8762, epsilon=0, learning=False)
    sim2.agents = sim1.agents
    for agent in sim2.agents:
        if isinstance(agent, BatteryAgent):
            agent.soc = 0.5
    sim2.run()
    
    sim1.show_performance_metrics()
    sim2.show_performance_metrics()
