import random
from utils.discretization import digitize_clip

class BaseAgent:
    def __init__(self, env, name, actions, state_space=None, alpha=0.1, gamma=0.9, **kwargs):
        """Agente base para todos los tipos de recursos.

        Parameters
        ----------
        env : object
            Referencia al entorno que contiene dataset y discretizaciones.
        name : str
            Identificador del agente.
        actions : list
            Lista de acciones discretas disponibles.
        state_space : list | None
            Definición del espacio de estados (lista de especificaciones) opcional.
        alpha : float
            Tasa de aprendizaje para Q-Learning.
        gamma : float
            Factor de descuento.
        """
        self.name = name
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.q_table = {}
        self.action = 0
        self.idx = 0
        self.power = 0
        self.potential = 0
        # Guarda la definición del espacio de estado si se provee, evita variable no definida
        self.state_space = state_space or []
 
    def get_dataset(self, field: str, index: int) -> None:
        """
        Update agent attributes with values and discretized states
        from the dataset row at the given index.
        """

        # Extract values from dataset row
        row = self.env.dataset.iloc[index]
        self.potential = row[field]  # Update potential

        # Compute discretized states
        return digitize_clip(row[field], self.env.power_bins)

    def get_discretized_state(self, env, index):
        """
        Build the discretized state tuple for this agent.
        Iterates through self.state_space and applies logic depending on source.
        """

        state_values = []

        for state in self.state_space:
            if state["source"] == "local":
                # Example: var_wind_0 (if agent name is wind#0 and var = "var")
                var_name = f"{state['var']}_{self.name.split('#')[1]}"
                value = self.get_dataset(var_name, index)
            elif state["source"] == "global":
                # Use global index/state from environment
                value = env.get_value(state["var"])
            elif state["source"] == "self":
                # Use agent's own index/state
                value = self.idx
            elif state["source"] == "external":
                # Use external value from environment
                value = getattr(env, state["var"])
            else:
                # Use environment value
                value = env.get_dataset(state["var"], index)

            state_values.append(value)

        return tuple(state_values)

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            self.action = random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, {a: 0.0 for a in self.actions})
            self.action = max(q_values, key=q_values.get)
        return self.action

    def update_q_table(self, state, action, reward, next_state):
        q_values = self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        current_q = q_values[action]
        next_q_values = self.q_table.get(next_state, {a: 0.0 for a in self.actions})
        max_next_q = max(next_q_values.values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
