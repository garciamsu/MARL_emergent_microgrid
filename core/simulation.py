import pandas as pd
from core.environment import MultiAgentEnv
from core.registry import create_agent, create_policy, create_reward
import core.policies
import core.rewards

EPSILON_MIN = 0

def instantiate_agents(config, env):
    agents = {}
    for agent_type, spec in config["agents"].items():
        for idx in range(spec["count"]):
            name = f"{agent_type}#{idx}"
            policy = create_policy(spec["policy"])
            reward_fn = create_reward(spec["reward"])
            # Elimina las claves duplicadas
            spec_clean = {k: v for k, v in spec.items() if k not in ["policy", "reward"]}
            agent = create_agent(
                agent_type, env=env, policy=policy, reward_fn=reward_fn, **spec_clean
            )
            agents[name] = agent
    return agents


def run_training(config):

    env = MultiAgentEnv(config)
    agents = instantiate_agents(config, env)

    num_episodes = config["simulation"]["episodes"]
    epsilon = config["simulation"].get("epsilon", {}).get("start", 1.0)

    for ep in range(num_episodes):
        env.reset()

        for index in range(env.max_steps):
            env.get_value(index)


            
