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

    print(agents)

    results = []
    
    for ep in range(num_episodes):
        env.reset()
        evolution = []

        for index in range(env.max_steps - 1):
            # 1. Discretized state per agent
            state = {
                name: ag.get_discretized_state(env, index)
                for name, ag in agents.items()
            }
            
        
            # 2. Choose action per agent
            for ag in agents.values():
                ag.choose_action(state[type(ag).__name__], epsilon)
                print(ag.action)
                                      
            # 3. Environment update based on agent actions
            solar_power, wind_power, bat_power, grid_power, load_power = 0, 0, 0, 0, 0
            battery_agent = None

            for ag in agents.values():
                ag.update_power(env)
    
            # -------------------------------------------------------------
            # Environment update (supports multiple agents per type)
            # -------------------------------------------------------------
            solar_agents = [ag for ag in agents.values() if "solar" in ag.name]
            wind_agents = [ag for ag in agents.values() if "wind" in ag.name]
            battery_agents = [ag for ag in agents.values() if "battery" in ag.name]
            grid_agents = [ag for ag in agents.values() if "grid" in ag.name]
            load_agents = [ag for ag in agents.values() if "load" in ag.name]

            # Reset dynamic records
            env.energy_balance = {}
            total_generation = 0.0
            total_consumption = 0.0

            # --- SOLAR ---
            for ag in solar_agents:
                ag.update_power(env)
                env.energy_balance[ag.name] = ag.power
                total_generation += ag.power

            # --- WIND ---
            for ag in wind_agents:
                ag.update_power(env)
                env.energy_balance[ag.name] = ag.power
                total_generation += ag.power

            # --- BATTERIES ---
            for ag in battery_agents:
                ag.update_power(env)
                env.energy_balance[ag.name] = ag.power
                if ag.power >= 0:
                    total_generation += ag.power
                else:
                    total_consumption += abs(ag.power)

            # --- GRID ---
            for ag in grid_agents:
                ag.update_power(env)
                env.energy_balance[ag.name] = ag.power
                if ag.power >= 0:
                    total_generation += ag.power
                else:
                    total_consumption += abs(ag.power)

            # --- LOADS ---
            for ag in load_agents:
                ag.update_power(env)
                env.energy_balance[ag.name] = ag.power
                if ag.power >= 0:
                    total_generation += ag.power
                else:
                    total_consumption += abs(ag.power)

            # --- AGGREGATION ---
            env.renewable_power = sum(
                env.energy_balance[n] for n in env.energy_balance if "solar" in n or "wind" in n
            )
            env.total_power = total_generation - total_consumption
            env.demand_power = max(env.demand_power + total_consumption, 0)

            # --- BALANCE ---
            env.delta_power = env.total_power - env.demand_power
            env.delta_power_idx = "surplus" if env.delta_power >= 0 else "deficit"

            # 4. Next state
            next_state = {
                name: ag.get_discretized_state(env, index + 1)
                for name, ag in agents.items()
            }

            # 5. Reward calculation and Q-table update
            step_record = {"episode": ep, "step": index}
            for name, ag in agents.items():
                reward = ag.calculate_reward(*state[type(ag).__name__])
                ag.update_q_table(state[type(ag).__name__], ag.action,
                                  reward, next_state[type(ag).__name__])
                step_record[f"reward_{name}"] = reward
                step_record[f"action_{name}"] = ag.action
                step_record[f"power_{name}"] = getattr(ag, "power", 0.0)
            evolution.append(step_record)

        # 6. Epsilon update
        if decay == "linear":
            epsilon = max(epsilon_min, epsilon - (1.0 - epsilon_min) / num_episodes)
        elif decay == "exponential":
            epsilon = max(epsilon_min, epsilon * 0.99)

        # 7. Save episode data
        df = pd.DataFrame(evolution)
        df.to_csv(f"results/evolution/episode_{ep}.csv", index=False)
        results.append(df)

        print(f"Episode {ep+1}/{num_episodes} completed, epsilon={epsilon:.3f}")

    return agents, results


            
