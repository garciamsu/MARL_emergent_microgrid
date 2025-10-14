import random
from utils.discretization import digitize_clip

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
        self.potential = 0
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

        # Compute discretized states
        return digitize_clip(row[field], self.env.power_bins)

    def get_discretized_state(self, env, index):
        """Construct the discretized state tuple for this agent.

        Iterates over the declarative ``state_space`` configuration and pulls
        values from either local dataset columns, environment global attributes
        or internal indices.
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
        """One-step tabular Q-learning update."""
        q_values = self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        current_q = q_values[action]
        next_q_values = self.q_table.get(next_state, {a: 0.0 for a in self.actions})
        max_next_q = max(next_q_values.values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
