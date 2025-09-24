from core.registry import register_policy

@register_policy("epsilon_greedy")
def epsilon_greedy(epsilon=0.1):
    return {"type": "epsilon_greedy", "epsilon": epsilon}