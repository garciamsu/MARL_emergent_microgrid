"""High-level training loop for the multi-agent microgrid.

The current implementation performs an episodic tabular Q-learning procedure
over a fixed historical dataset (no stochastic environment transitions beyond
what the dataset provides). Each agent:

1. Extracts a discretized local/global state tuple.
2. Chooses an action via epsilon-greedy policy.
3. Updates its internal power contribution through ``update_power``.
4. Receives a scalar reward from its domain-specific ``calculate_reward``.
5. Updates its tabular Q-values.

Limitations / Future work:
- Provide a proper environment ``step`` returning joint observations & rewards.
- Support parallel environments or experience replay buffers.
- Add termination conditions distinct from dataset exhaustion.
- Decouple logging export from the loop with an event/callback system.
"""

import pandas as pd
from core.environment import MultiAgentEnv
from agents import instantiate_agents
from core.utils import set_global_seed, build_logger

EPSILON_MIN = 0

def run_training(config):
    """Execute multi-agent tabular Q-learning.

    Args:
        config (dict): Full configuration structure loaded from YAML.

    Returns:
        tuple:
            agents (dict[str, BaseAgent]): Mapping agent name -> trained agent.
            results (list[pd.DataFrame]): Per-episode step-wise log DataFrames.

    Side Effects:
        Writes CSV evolution logs into ``results/evolution``.
    """
    # Seeding & logger
    seed = config.get("simulation", {}).get("seed", 42)
    set_global_seed(seed)
    logger = build_logger()

    env = MultiAgentEnv(config)
    agents = instantiate_agents(config, env)

    num_episodes = config["simulation"]["episodes"]
    epsilon_cfg = config["simulation"].get("epsilon", {})
    epsilon = epsilon_cfg.get("start", 1.0)
    epsilon_min = epsilon_cfg.get("min", 0.05)
    decay = epsilon_cfg.get("decay", "linear")

    results = []
    # Simulation time step in hours (used for SOC integration)
    dt_h = config.get("simulation", {}).get("dt_h", 1.0)

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

            # Update environment
            env.get_dataset("demand", index)
            env.get_dataset("price", index)
            env.total_power = 0.0
            env.renewable_power = 0.0

            # Update power for each agent
            for agent in agents.values():
                agent.update_power(env)

                # Accumulate renewable generation (solar or wind)
                if "solar" in agent.name.lower() or "wind" in agent.name.lower():
                    env.renewable_power += agent.power

                # Classify power as generation or consumption
                if agent.power >= 0:
                    env.total_power += agent.power
                else:
                    env.demand_power += abs(agent.power)

                # Update SOC for battery agents and publish discrete SOC to env
                if agent.name.startswith("battery"):
                    try:
                        v_nom = getattr(agent, "v_nom", 48.0)
                        agent.update_soc(agent.power, dt_h=dt_h, nominal_voltage=v_nom)
                        env.soc_idx = agent.idx
                    except AttributeError:
                        # Agent may not implement update_soc yet
                        pass

            # Step log: Per-agent variables (safe defaults if attribute is missing)
            for name, agent in agents.items():
                step_record[f"potential_{name}"] = getattr(agent, "potential", None)
                step_record[f"action_{name}"] = getattr(agent, "action", None)
                step_record[f"power_{name}"] = getattr(agent, "power", 0.0)
                if name.startswith("battery"):
                    step_record[f"soc_{name}"] = getattr(agent, "soc", None)
                    step_record[f"soc_idx_{name}"] = getattr(agent, "idx", None)

            # Update environment global variables
            env.energy_balance = env.total_power - env.demand_power
            env.delta_power_idx = "surplus" if env.energy_balance >= 0 else "deficit"

            # Step log: Append environment globals at the end (preserve insertion order)
            step_record.update({
                "env_total_generation": env.total_power,
                "env_total_consumption": env.demand_power,
                "env_total_renewable": env.renewable_power,
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
                state_tuple = state[name]
                next_state_tuple = next_state[name]

                # Call calculate_reward with unpacked state when applicable
                try:
                    reward = agent.calculate_reward(*state_tuple)
                except TypeError:
                    reward = agent.calculate_reward(state_tuple)

                # Q-learning update
                agent.update_q_table(state_tuple, agent.action, reward, next_state_tuple)

                # Log per-agent values
                step_record[f"reward_{name}"] = reward

            evolution.append(step_record)

        # 6. Epsilon update
        if decay == "linear":
            epsilon = max(epsilon_min, epsilon - (1.0 - epsilon_min) / num_episodes)
        elif decay == "exponential":
            epsilon = max(epsilon_min, epsilon * 0.99)

        # 7. Save episode data
        episode_df = pd.DataFrame(evolution)
        episode_df.to_csv(f"results/evolution/episode_{episode}.csv", index=False)
        results.append(episode_df)

    logger.info("Episode %d/%d completed | epsilon=%.3f", episode + 1, num_episodes, epsilon)

    return agents, results
