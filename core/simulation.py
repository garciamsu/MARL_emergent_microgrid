import copy
import random
import pandas as pd
from utils.discretization import digitize_clip
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
    epsilon_cfg = config["simulation"].get("epsilon", {})
    epsilon = epsilon_cfg.get("start", 1.0)
    epsilon_min = epsilon_cfg.get("min", 0.05)
    decay = epsilon_cfg.get("decay", "linear")

    results = []

    for ep in range(num_episodes):
        env.reset()
        evolution = []

        for index in range(env.max_steps - 1):
            # 1. Estado actual para cada agente
            state = {
                name: ag.get_discretized_state(env, index)
                for name, ag in agents.items()
            }

            # 2. Elegir acción para cada agente
            for ag in agents.values():
                ag.choose_action(state[type(ag).__name__], epsilon)

            # 3. Actualizar variables del entorno según las acciones
            # (ejemplo simple: sumar potencias de agentes productores)
            total_power = 0
            for ag in agents.values():
                total_power += getattr(ag, "power", 0)
            env.total_power = total_power
            env.total_power_idx = digitize_clip(env.total_power, env.renewable_bins)

            # 4. Avanzar un paso y calcular siguiente estado
            next_state = {
                name: ag.get_discretized_state(env, index + 1)
                for name, ag in agents.items()
            }

            # 5. Calcular recompensas y actualizar Q-tables
            step_record = {"episode": ep, "step": index}
            for name, ag in agents.items():
                reward = ag.calculate_reward(*state[type(ag).__name__])
                ag.update_q_table(state[type(ag).__name__], ag.action,
                                  reward, next_state[type(ag).__name__])
                step_record[f"reward_{name}"] = reward
                step_record[f"action_{name}"] = ag.action
            evolution.append(step_record)

        # 6. Actualizar política de exploración (ε)
        if decay == "linear":
            epsilon = max(epsilon_min, epsilon - (1.0 - epsilon_min) / num_episodes)
        elif decay == "exponential":
            epsilon = max(epsilon_min, epsilon * 0.99)

        # 7. Guardar evolución por episodio
        df = pd.DataFrame(evolution)
        df.to_csv(f"results/evolution/episode_{ep}.csv", index=False)
        results.append(df)

        print(f"Episode {ep+1}/{num_episodes} completed, epsilon={epsilon:.3f}")

    return agents, results


            
