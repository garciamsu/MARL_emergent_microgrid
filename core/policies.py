import numpy as np
from collections import defaultdict
from core.registry import register_policy


class Policy:
    """Abstract policy interface.

    Implementations should provide epsilon-greedy compatible ``select_action``
    and a value update method (e.g. Q-learning, SARSA, etc.).
    """

    def select_action(self, state, epsilon):  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, state, action, reward, next_state):  # pragma: no cover
        raise NotImplementedError


@register_policy("tabular_ql")
class TabularQL(Policy):
    """Simple tabular Q-learning policy.

    Args:
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        n_actions (int): Number of discrete actions.
    """

    def __init__(self, alpha=0.1, gamma=0.9, n_actions=2):
        self.alpha = alpha
        self.gamma = gamma
        self.q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state, epsilon):
        """Return an action using epsilon-greedy exploration."""
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.q[state]))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state):
        """Perform the Q-learning update for one transition."""
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
