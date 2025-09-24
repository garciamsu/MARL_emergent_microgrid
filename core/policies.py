import numpy as np
from collections import defaultdict
from core.registry import register_policy


class Policy:
    def select_action(self, state, epsilon):
        raise NotImplementedError

    def update(self, state, action, reward, next_state):
        raise NotImplementedError


@register_policy("tabular_ql")
class TabularQL(Policy):
    def __init__(self, alpha=0.1, gamma=0.9, n_actions=2):
        self.alpha = alpha
        self.gamma = gamma
        self.q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.q[state]))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
