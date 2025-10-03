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
            
            print(state)
            #for ag in agents.values():
            #    print(ag.state_space)            
            
        '''
            # 2. Choose action per agent
            for ag in agents.values():
                ag.choose_action(state[type(ag).__name__], epsilon)

            # 3. Environment update based on agent actions
            solar_power, wind_power, bat_power, grid_power, load_power = 0, 0, 0, 0, 0
            battery_agent = None

            for ag in agents.values():
                if ag.name.startswith("solar"):
                    # Solar: produce if action=1
                    solar_power = ag.potential * ag.action
                    ag.power = solar_power

                elif ag.name.startswith("wind"):
                    # Wind: produce if action=1
                    wind_power = ag.potential * ag.action
                    ag.power = wind_power

                elif ag.name.startswith("battery"):
                    # Battery: idle=0, charge=1, discharge=2
                    if ag.action == 1:   # charge
                        bat_power = -abs(env.demand_power - (solar_power + wind_power))
                    elif ag.action == 2: # discharge
                        bat_power = abs(env.demand_power - (solar_power + wind_power))
                    else:
                        bat_power = 0
                    ag.power = bat_power
                    ag.update_soc(power_w=bat_power)
                    battery_agent = ag

                elif ag.name.startswith("grid"):
                    # Grid: supply if action=1
                    if ag.action == 1:
                        grid_power = abs(env.demand_power - (solar_power + wind_power) - bat_power)
                    else:
                        grid_power = 0
                    ag.power = grid_power

                elif ag.name.startswith("load"):
                    # Load: ON=1 (extra demand), OFF=0 (demand reduction)
                    if ag.action == 1:
                        load_power = 0  # default extra demand
                    else:
                        load_power = -15  # controllable reduction
                    ag.power = load_power

            # Update environment state
            env.renewable_power = solar_power + wind_power
            env.renewable_power_idx = digitize_clip(env.renewable_power, env.renewable_bins)

            env.total_power = env.renewable_power + bat_power + grid_power + load_power
            env.total_power_idx = digitize_clip(env.total_power, env.renewable_bins)

            env.demand_power = env.demand_power + load_power
            env.demand_power_idx = digitize_clip(env.demand_power, env.demand_bins)

            env.delta_power = env.total_power - env.demand_power
            env.delta_power_idx = 1 if env.delta_power >= 0 else 0

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

        '''
    return agents, results


            
