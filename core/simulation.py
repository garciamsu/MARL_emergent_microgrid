import copy
import random
import pandas as pd
from utils.discretization import digitize_clip
from core.environment import MultiAgentEnv
from core.registry import create_agent, create_policy, create_reward
from agents import instantiate_agents
import core.policies
import core.rewards

EPSILON_MIN = 0

def run_training(config):
    """
    Run a full multi-agent Q-learning training loop.

    This function orchestrates the training process across multiple agents
    (Solar, Wind, Battery, Grid, Load), updating the environment and Q-tables
    at each step and saving per-episode results.

    Args:
        config (dict): Full configuration loaded from YAML/JSON.

    Returns:
        agents (dict): Dictionary of trained agents.
        results (list[pd.DataFrame]): List of per-episode DataFrames with logs.
    """
    env = MultiAgentEnv(config)
    agents = instantiate_agents(config, env)

    num_episodes = config["simulation"]["episodes"]
    epsilon_cfg = config["simulation"].get("epsilon", {})
    epsilon = epsilon_cfg.get("start", 1.0)
    epsilon_min = epsilon_cfg.get("min", 0.05)
    decay = epsilon_cfg.get("decay", "linear")

    results = []

    for episode in range(num_episodes):
        env.reset()
        evolution = []

        for index in range(env.max_steps - 1):
            # 1. Discretized state per agent
            state = {
                name: agent.get_discretized_state(env, index)
                for name, agent in agents.items()
            }

            # Step log: environment global variables and per-agent fields
            step_record = {
                "episode": episode,
                "step": index,
            }

            # 2. Choose action per agent
            for agent in agents.values():
                agent.choose_action(state[agent.name], epsilon)

            # 3. Environment update based on agent actions

            # Update environment with current demand
            env.get_dataset("demand", index)

            # Initialize accumulators
            total_renewable = 0.0
            total_generation = 0.0
            total_consumption = 0.0

            # Update power for each agent
            for agent in agents.values():
                agent.update_power(env)

                # Accumulate renewable generation (solar or wind)
                if "solar" in agent.name.lower() or "wind" in agent.name.lower():
                    total_renewable += agent.power

                # Classify power as generation or consumption
                if agent.power >= 0:
                    total_generation += agent.power
                else:
                    total_consumption += abs(agent.power)

            # Step log: Per-agent variables (safe defaults if attribute is missing)
            for name, agent in agents.items():
                step_record[f"potential_{name}"] = getattr(agent, "potential", None)
                step_record[f"action_{name}"] = getattr(agent, "action", None)
                step_record[f"power_{name}"] = getattr(agent, "power", 0.0)

            # Update environment global variables
            env.total_power = total_generation
            env.demand_power = max(env.demand_power + total_consumption, 0)
            env.energy_balance = total_generation - total_consumption
            env.delta_power_idx = "surplus" if env.energy_balance >= 0 else "deficit"

            # Step log: Append environment globals at the end (preserve insertion order)
            step_record.update({
                "env_total_generation": total_generation,
                "env_total_consumption": total_consumption,
                "env_total_renewable": total_renewable,
                "env_total_power": env.total_power,
                "env_demand_power": env.demand_power,
                "env_energy_balance": env.energy_balance,
                "env_delta_power_idx": env.delta_power_idx,
            })

            # 4. Next state
            next_state = {
                name: agent.get_discretized_state(env, index + 1)
                for name, agent in agents.items()
            }

            # 5. Reward calculation and Q-table update
            for name, agent in agents.items():
                print(state[agent.name])
                reward = agent.calculate_reward(state[agent.name])
                print(reward)
            #    agent.update_q_table(state[agent.name], agent.action,
            #                      reward, next_state[state[agent.name]])
            #    step_record[f"reward_{name}"] = reward
            #    step_record[f"action_{name}"] = agent.action
            #    step_record[f"power_{name}"] = getattr(agent, "power", 0.0)

            evolution.append(step_record)

        # 6. Epsilon update
        if decay == "linear":
            epsilon = max(epsilon_min, epsilon - (1.0 - epsilon_min) / num_episodes)
        elif decay == "exponential":
            epsilon = max(epsilon_min, epsilon * 0.99)

        # 7. Save episode data
        df = pd.DataFrame(evolution)
        df.to_csv(f"results/evolution/episode_{episode}.csv", index=False)
        results.append(df)

        print(f"Episode {episode+1}/{num_episodes} completed, epsilon={epsilon:.3f}")

    return agents, results