import numpy as np
import random

# Parámetros físicos y constantes
ETA = 0.15  # Eficiencia de conversión solar
SOLAR_AREA = 10  # Área de los paneles solares en m^2
T_AMBIENT = 25  # Temperatura ambiente en °C
PHI = 1000  # Irradiancia solar en W/m^2
RHO = 1.225  # Densidad del aire en kg/m^3
BLADE_AREA = 5  # Área de los álabes de la turbina en m^2
C_P = 0.4  # Coeficiente de potencia
C_CONFORT = 0.5  # Umbral de confort para el costo del mercado

class Environment:
    def __init__(self):
        self.solar_bins = np.linspace(0, 1, 10)
        self.wind_bins = np.linspace(0, 1, 10)
        self.battery_bins = np.linspace(0, 1, 10)
        self.demand_bins = np.linspace(0, 1, 10)
        self.price_bins = np.linspace(0, 1, 10)

    def discretize_state(self, state):
        return tuple(np.digitize(s, b) for s, b in zip(state, [
            self.solar_bins, self.wind_bins, self.battery_bins, self.demand_bins, self.price_bins
        ]))

class Agent:
    def __init__(self, name, actions, alpha=1.0, beta=1.0, gamma=1.0):
        self.name = name
        self.actions = actions
        self.q_table = {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def initialize_q_table(self, environment):
        states = [
            (a, b, c, d, e) for a in range(len(environment.solar_bins))
            for b in range(len(environment.wind_bins))
            for c in range(len(environment.battery_bins))
            for d in range(len(environment.demand_bins))
            for e in range(len(environment.price_bins))
        ]
        self.q_table = {state: {action: 0 for action in self.actions} for state in states}

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, {a: 0 for a in self.actions})
            return max(q_values, key=q_values.get)

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        q_values = self.q_table.get(state, {a: 0 for a in self.actions})
        next_q_values = self.q_table.get(next_state, {a: 0 for a in self.actions})
        q_values[action] += alpha * (
            reward + gamma * max(next_q_values.values()) - q_values[action]
        )
        self.q_table[state] = q_values

class SolarAgent(Agent):
    def __init__(self):
        super().__init__("solar", ["produce", "idle"], alpha=1.0, beta=1.0, gamma=1.0)

    def calculate_power(self, irradiance):
        return ETA * SOLAR_AREA * irradiance * (1 - 0.005 * (T_AMBIENT - 25))

    def calculate_reward(self, P_H, P_L, SOC, active_agents):
        total_renewables = sum(agent.calculate_power(random.uniform(0, 1)) for agent in active_agents if isinstance(agent, (SolarAgent, WindAgent)))
        if P_H <= P_L:
            return self.alpha * (P_L / (P_H + total_renewables + 1e-6))
        elif P_H > P_L and SOC == 1:
            return self.beta * (P_L - (P_H + total_renewables))
        elif P_H > P_L and SOC < 1:
            return -self.gamma * ((P_H + total_renewables - P_L))
        return 0

class WindAgent(Agent):
    def __init__(self):
        super().__init__("wind", ["produce", "idle"], alpha=1.0, beta=1.0, gamma=1.0)

    def calculate_power(self, wind_speed):
        return 0.5 * RHO * BLADE_AREA * C_P * wind_speed ** 3

    def calculate_reward(self, P_H, P_L, SOC, active_agents):
        if P_H <= P_L:
            return self.alpha * (P_L / (P_H + 1e-6))
        elif P_H > P_L and SOC == 1:
            return self.beta * (P_L - (P_H / len(active_agents)))
        elif P_H > P_L and SOC < 1:
            return -self.gamma * ((P_H - P_L) / len(active_agents))
        return 0

class BatteryAgent(Agent):
    def __init__(self):
        super().__init__("battery", ["charge", "discharge", "idle"], alpha=1.0, beta=1.0, gamma=1.0)
        self.soc = 0.5

    def update_soc(self, action, charge_rate=0.1):
        if action == "charge":
            self.soc = min(self.soc + charge_rate, 1.0)
        elif action == "discharge":
            self.soc = max(self.soc - charge_rate, 0.0)

    def calculate_reward_charge(self, P_T, P_L):
        if self.soc < 1 and P_T > P_L:
            return self.alpha * (1 - self.soc) * (P_T - P_L)
        elif self.soc <= 1 and P_T <= P_L:
            return -self.gamma * self.soc * (P_L - P_T)
        return 0

    def calculate_reward_discharge(self, P_T, P_L):
        if self.soc >= 0.5 and P_T < P_L:
            return self.alpha * self.soc * (P_L - P_T)
        elif self.soc < 0.5 and P_T > P_L:
            return -self.beta * (1 - self.soc) * (P_T - P_L)
        elif self.soc < 0.5 and P_T < P_L:
            return -self.gamma * (1 - self.soc) * (P_L - P_T)
        return 0

class GridAgent(Agent):
    def __init__(self):
        super().__init__("grid", ["sell", "idle"], alpha=1.0, beta=1.0, gamma=1.0)

    def calculate_reward(self, state, action, P_T, P_L, SOC, C_mercado, S_UT):
        if action == "sell":
            if SOC < 0.5 and P_T < P_L and S_UT == 0:
                return self.alpha / (P_T * C_mercado)
            elif SOC >= 0.5 and P_T >= P_L and S_UT == 0:
                return -self.beta * P_T * C_mercado
            elif (SOC >= 0.5 or P_T >= P_L) and S_UT == 1:
                return -self.gamma * P_T * C_mercado
        return 0

class LoadAgent(Agent):
    def __init__(self):
        super().__init__("load", ["consume", "idle"], alpha=1.0, beta=1.0, gamma=1.0)

    def calculate_reward(self, state, action, P_T, P_L, SOC, C_mercado):
        if action == "consume":
            if P_T > P_L or SOC >= 0.5:
                return self.alpha * (self.beta / (P_T + C_mercado))
            elif C_mercado > C_CONFORT:
                return -self.gamma * P_T * C_mercado
        elif action == "idle":
            if P_T > P_L or SOC >= 0.5:
                return -self.beta * P_T * C_mercado
        return 0

class Simulation:
    def __init__(self, num_episodes, max_steps):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.environment = Environment()
        self.agents = [
            SolarAgent(),
            WindAgent(),
            BatteryAgent(),
            GridAgent(),
            LoadAgent()
        ]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def run(self):
        for agent in self.agents:
            agent.initialize_q_table(self.environment)

        for episode in range(self.num_episodes):
            current_state = self.environment.discretize_state(
                [random.uniform(0, 1) for _ in range(5)]
            )

            for step in range(self.max_steps):
                active_agents = [agent for agent in self.agents if isinstance(agent, (SolarAgent, WindAgent))]

                for agent in self.agents:
                    action = agent.choose_action(current_state, self.epsilon)

                    if isinstance(agent, SolarAgent):
                        irradiance = random.uniform(0, 1)
                        power = agent.calculate_power(irradiance) if action == "produce" else 0
                        reward = agent.calculate_reward(
                            P_H=power, P_L=random.uniform(0, 1), SOC=0.5, active_agents=active_agents
                        )

                    elif isinstance(agent, WindAgent):
                        wind_speed = random.uniform(0, 1)
                        power = agent.calculate_power(wind_speed) if action == "produce" else 0
                        reward = agent.calculate_reward(
                            P_H=power, P_L=random.uniform(0, 1), SOC=0.5, active_agents=active_agents
                        )

                    elif isinstance(agent, BatteryAgent):
                        if action == "charge":
                            reward = agent.calculate_reward_charge(P_T=random.uniform(0, 1), P_L=random.uniform(0, 1))
                        elif action == "discharge":
                            reward = agent.calculate_reward_discharge(P_T=random.uniform(0, 1), P_L=random.uniform(0, 1))
                        else:
                            reward = 0
                        agent.update_soc(action)

                    elif isinstance(agent, GridAgent):
                        reward = agent.calculate_reward(
                            current_state, action, P_T=random.uniform(0, 1), P_L=random.uniform(0, 1),
                            SOC=0.5, C_mercado=random.uniform(0.1, 1), S_UT=random.choice([0, 1])
                        )
                    elif isinstance(agent, LoadAgent):
                        reward = agent.calculate_reward(
                            current_state, action, P_T=random.uniform(0, 1), P_L=random.uniform(0, 1),
                            SOC=0.5, C_mercado=random.uniform(0.1, 1)
                        )
                    else:
                        reward = random.uniform(-1, 1)

                    next_state = self.environment.discretize_state(
                        [random.uniform(0, 1) for _ in range(5)]
                    )
                    agent.update_q_table(current_state, action, reward, next_state, self.alpha, self.gamma)
                    current_state = next_state

if __name__ == "__main__":
    simulation = Simulation(num_episodes=1000, max_steps=100)
    simulation.run()
