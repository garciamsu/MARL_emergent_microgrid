import random
from utils.discretization import digitize_clip
from itertools import product

class BaseAgent:
    """Base class for all resource agents.

    Provides:
        - Generic epsilon-greedy action selection (via ``choose_action``).
        - Tabular Q-value storage using nested dicts.
        - Generic discrete state construction based on a declarative
          ``state_space`` specification list.

    Expected keys in each state descriptor (state_space list):
        - var (str): Variable name.
        - source (str): One of {``local``, ``global``, ``self``, ``external``, ``env``}.
        - bins (Any): Currently unused at runtime (placeholder for future
          per-variable bin customization / validation).
    """

    def __init__(self, env, name, actions, state_space=None, alpha=0.1, gamma=0.9, **kwargs):
        """Initialize agent.

        Args:
            env: Environment reference.
            name (str): Unique agent identifier (e.g. ``solar#0``).
            actions (list[int]): Discrete action set.
            state_space (list[dict] | None): Declarative state space definition.
            alpha (float): Learning rate for Q-learning updates.
            gamma (float): Discount factor.
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
        self.power_norm = 0
        self.power_idx = 0
        self.potential = 0
        self.potential_norm = 0
        self.potential_idx = 0
        # Recompensa inyectada desde configuración (si existe)
        self.reward_fn = kwargs.get("reward_fn", None)
        # Guarda la definición del espacio de estado si se provee, evita variable no definida
        self.state_space = state_space or []
 
    def get_dataset(self, field: str, index: int) -> None:
        """Return discretized value for a dataset field at a given index.

        Also updates the agent's ``potential`` attribute with raw (continuous)
        value which several agents later reuse in ``update_power``.
        """

        # Extract values from dataset row
        row = self.env.dataset.iloc[index]
        self.potential = row[field]  # Update potential
        self.potential_norm = self.potential / self.env.max_value
        self.potential_idx = digitize_clip(self.potential, self.env.power_bins)

        # Compute discretized states
        return self.potential_idx

    def get_discretized_state(self, env, index):
        """Construct the discretized state tuple for this agent.

        Iterates over the declarative ``state_space`` configuration and pulls
        values from either local dataset columns, environment global attributes
        or internal indices.
        """

        state_values = []

        """
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
        """
        # Use episode_data (random window) if available, otherwise fall back to full dataset
        data_source = env.episode_data if env.episode_data is not None else env.dataset
        row = data_source.iloc[index]

        env.solar_potential = row["solar_potential"]
        env.solar_potential_norm = env.solar_potential / env.max_value
        env.solar_potential_idx = 1 if env.solar_potential_norm  > 0.0 else 0
        env.wind_potential = row["wind_potential"]
        env.wind_potential_norm = env.wind_potential / env.max_value
        env.wind_potential_idx = 1 if env.wind_potential_norm  > 0.0 else 0
        env.price = row["price"]
        env.demand_power = row["demand"]

        env.renewable_potential = env.solar_potential + env.wind_potential
        env.demand_power = env.demand_power

        for state in self.state_space:
            
            var = state.get("var")

            if var == "delta_ph":
               env.delta_ph =  env.renewable_power - env.demand_power
               env.delta_ph_norm = env.delta_ph / env.max_value
               env.delta_ph_idx = digitize_clip(env.delta_ph_norm, env.delta_bins)
               value = env.delta_ph_idx
            elif var == "solar_potential":
               value = env.solar_potential_idx
            elif var == "wind_potential":
               value = env.wind_potential_idx
            elif var == "soc":
                value = self.idx
            elif var == "pu_power":
                grid_power_norm = env.grid_power / env.max_value
                grid_power_idx = 1 if grid_power_norm  > 0.00 else 0
                value = grid_power_idx
            elif var == "cm":
                # Use comfort_threshold from price_bins (loaded from YAML)
                # price_bins = [0, comfort_threshold, max_price] creates binary discretization
                env.price_norm  = env.price / env.max_price
                env.price_idx = digitize_clip(env.price, env.price_bins)
                value = env.price_idx
            else:  # Default to 'env'
                value = -999 # Valor por defecto si no se reconoce la variable

            state_values.append(value)

        return tuple(state_values)

    def choose_action(self, state, epsilon=0.1):
        # No crear estados bajo demanda: la Q-table debe haberse inicializado previamente
        q_values = self.q_table.get(state)
        if q_values is None:
            raise KeyError(f"Estado {state} no está en la Q-table de {self.name}. Verifique initialize_q_table vs state_space.")
        if random.random() < epsilon:
            self.action = random.choice(self.actions)
        else:
            self.action = max(q_values, key=q_values.get)
        return self.action

    def update_q_table(self, state, action, reward, next_state):
        """One-step tabular Q-learning update."""
        q_values = self.q_table.get(state)
        if q_values is None:
            raise KeyError(f"Estado {state} no inicializado en Q-table de {self.name}.")
        current_q = q_values[action]
        next_q_values = self.q_table.get(next_state)
        if next_q_values is None:
            raise KeyError(f"Estado siguiente {next_state} no inicializado en Q-table de {self.name}.")
        max_next_q = max(next_q_values.values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def initialize_q_table(self, env):
        """Inicializa la Q-table a partir de la definición declarativa de state_space.

        Para cada dimensión en state_space, determina la cardinalidad:
        - bins es lista/tupla/ndarray -> len(bins)
        - bins == 'auto' o ausente -> len(env.power_bins)
        - Caso especial SOC: si var in {'soc','soc_idx'} y existe self.battery_soc_bins -> len(self.battery_soc_bins)
        - Caso especial price: si var == 'price' -> len(env.price_bins)
        Construye el producto cartesiano de los rangos y crea entradas con 0.0.
        """
        dims = []
        for desc in (self.state_space or []):
            bins_decl = desc.get("bins", "auto")
            var = desc.get("var")
            source = desc.get("source")

            # Caso especial SOC: usar bins de la batería si existen, de lo contrario
            # los bins globales de SOC del entorno para mantener la cardinalidad
            # consistente entre agentes que leen soc_idx desde env.
            if var in {"soc", "soc_idx"}:
                if hasattr(self, "battery_soc_bins"):
                    cardinality = len(self.battery_soc_bins)
                else:
                    cardinality = getattr(env, "num_soc_bins", 1) or 1
            # Caso especial price
            elif var == "price":
                cardinality = len(getattr(env, "price_bins", [])) or 1
            else:
                if isinstance(bins_decl, (list, tuple)):
                    cardinality = len(bins_decl)
                else:
                    # auto u otro valor -> usar bins de potencia del entorno
                    cardinality = len(getattr(env, "power_bins", [])) or 1

            dims.append(range(cardinality))

        # Si no hay state_space, no hay estados discretos definidos
        if not dims:
            self.q_table = {}
            return

        self.q_table = {state: {a: 0.0 for a in self.actions} for state in product(*dims)}
        agent_type = self.name.split("#")[0] if "#" in self.name else self.name
        print(
            f"Agente {agent_type}: tamaño actual (número de estados visitados): "
            f"{len(self.q_table)}"
        )
