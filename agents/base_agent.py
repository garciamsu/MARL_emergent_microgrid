import random


class BaseAgent:
    def __init__(self, name, actions, alpha=0.1, gamma=0.9, **kwargs):
        self.name = name
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.action = 0
        self.idx = 0

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
