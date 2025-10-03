import random


class BaseAgent:
    def __init__(self, env, name, actions, alpha=0.1, gamma=0.9, **kwargs):
        self.name = name
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.q_table = {}
        self.action = 0
        self.idx = 0

    def get_value(self, field: str, index: int) -> None:
        """
        Update agent attributes with values and discretized states 
        from the dataset row at the given index.
        """
        
        # Extract values from dataset row
        row = self.env.dataset.iloc[index]

        # Compute discretized states
        # self.demand_power_idx = digitize_clip(self.demand_power, self.demand_bins)
        return row[field]

    def get_discretized_state(self, env, index):
        """
        Build the discretized state tuple for this agent.
        Iterates through self.state_space and applies logic depending on source.
        """
        
        state_values = []
       
        for state in self.state_space:
            if state["source"] == "local":
                # Example: var_solar_0 (if agent name is solar#0 and var = "var")
                var_name = f"{state['var']}_{self.name.split('#')[1]}"
                value = self.get_value(var_name, index)
            else:
                # Use environment value
                value = env.get_value(state["var"], index)

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
