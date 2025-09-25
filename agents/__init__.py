import os
import importlib

# Obtiene el directorio actual
current_dir = os.path.dirname(__file__)

# Importa todos los módulos que terminan en '_agent.py'
for filename in os.listdir(current_dir):
    if filename.endswith('_agent.py'):
        module_name = f"agents.{filename[:-3]}"
        importlib.import_module(module_name)
        
from core.registry import create_agent, create_policy, create_reward

def instantiate_agents(config, env):
    agents = {}
    for agent_type, spec in config["agents"].items():
        for idx in range(spec["count"]):
            name = f"{agent_type}#{idx}"
            policy = create_policy(spec["policy"])
            reward_fn = create_reward(spec["reward"])
            spec_clean = {k: v for k, v in spec.items() if k not in ["policy", "reward"]}
            agent = create_agent(
                agent_type, env=env, policy=policy, reward_fn=reward_fn, **spec_clean
            )
            agents[name] = agent
    return agents