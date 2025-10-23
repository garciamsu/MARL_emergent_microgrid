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
from utils.discretization import digitize_clip


def make_epsilon_scheduler(cfg: dict, episodes: int):
    """Construye un scheduler de epsilon a partir de la config.

    Admite:
    - schedule: linear | exponential | constant | custom
    - start: valor inicial
    - end: valor objetivo (linear y opcional en exponential)
    - decay: factor (exponential). Si falta y hay end/start, se deriva para alcanzar end.
    - min: clip inferior (por defecto end si existe, o 0.01)
    - values: lista de tamaño >= episodes (custom). Si es más corta, se rellena con el último valor.
    """
    schedule = cfg.get("schedule", "linear").lower()
    start = float(cfg.get("start", 1.0))
    end_raw = cfg.get("end", None)
    end = float(end_raw) if end_raw is not None else None
    decay_raw = cfg.get("decay", None)
    decay = float(decay_raw) if isinstance(decay_raw, (int, float, str)) and str(decay_raw).replace('.', '', 1).isdigit() else None
    values = cfg.get("values", []) or []
    min_eps = float(cfg.get("min", end if end is not None else 0.01))

    # Normalización/validación simples
    start = max(0.0, min(1.0, start))
    if end is not None:
        end = max(0.0, min(1.0, end))
    min_eps = max(0.0, min(1.0, min_eps))

    def clip(x: float) -> float:
        return max(min_eps, min(1.0, float(x)))

    if schedule == "linear":
        # Si no hay end, usamos min_eps como objetivo
        target = end if end is not None else min_eps

        def f(t: int, _prev: float) -> float:
            frac = (t + 1) / max(1, episodes)
            return clip(start - (start - target) * frac)

        return f

    if schedule == "exponential":
        if decay is None:
            if end is not None and start > 0 and end > 0:
                # Derivar decay para alcanzar end en 'episodes' pasos
                decay_eff = (end / start) ** (1.0 / max(1, episodes))
            else:
                decay_eff = 0.99  # fallback sensato
        else:
            decay_eff = decay

        def f(_t: int, prev: float) -> float:
            base = prev if prev is not None else start
            return clip(base * decay_eff)

        return f

    if schedule == "constant":
        def f(_t: int, _prev: float) -> float:
            return clip(start)

        return f

    if schedule == "custom":
        series = [clip(v) for v in values]
        if not series:
            # Si no hay values, caemos a constante en start
            series = [clip(start)] * episodes
        if len(series) < episodes:
            series = series + [series[-1]] * (episodes - len(series))

        def f(t: int, _prev: float) -> float:
            return series[min(t, len(series) - 1)]

        return f

    # Por defecto, lineal
    target = end if end is not None else min_eps

    def f(t: int, _prev: float) -> float:
        frac = (t + 1) / max(1, episodes)
        return clip(start - (start - target) * frac)

    return f

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
    scheduler = make_epsilon_scheduler(epsilon_cfg, num_episodes)
    epsilon = float(epsilon_cfg.get("start", 1.0))

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
            # Sequential update order: Renewables → Load → Battery → Grid

            # Load base demand and price from dataset
            base_demand_from_dataset = env.dataset.iloc[index]["demand"] * env.scale_demand
            env.get_dataset("demand", index)
            env.get_dataset("price", index)
            
            # Store base demand for load agent to use
            env.base_demand = base_demand_from_dataset
            
            # Reset power accumulators
            env.total_power = 0.0
            env.renewable_power = 0.0
            # env.demand_power = 0.0  # MUST reset to accumulate correctly

            # PHASE 1: Update renewable agents (solar, wind)
            for agent in agents.values():
                if "solar" in agent.name.lower() or "wind" in agent.name.lower():
                    agent.update_power(env)
                    env.renewable_power += agent.power
                    env.total_power += agent.power

            # PHASE 2: Update load agent (can reduce demand)
            for agent in agents.values():
                if "load" in agent.name.lower():
                    agent.update_power(env)
                    # Load power is negative (consumption)
                    env.demand_power += abs(agent.power)

            # PHASE 3: Update battery agent (reacts to balance)
            for agent in agents.values():
                if "battery" in agent.name.lower():
                    agent.update_power(env)
                    # Battery can charge (negative) or discharge (positive)
                    if agent.power >= 0:
                        env.total_power += agent.power
                    else:
                        env.demand_power += abs(agent.power)

            # PHASE 4: Update grid agent (last resort)
            for agent in agents.values():
                if "grid" in agent.name.lower():
                    agent.update_power(env)
                    # Grid only imports (positive power)
                    if agent.power > 0:
                        env.total_power += agent.power

            # Step log: Per-agent variables (safe defaults if attribute is missing)
            for name, agent in agents.items():
                step_record[f"potential_{name}"] = getattr(agent, "potential", None)
                step_record[f"action_{name}"] = getattr(agent, "action", None)
                step_record[f"power_{name}"] = getattr(agent, "power", 0.0)
                step_record[f"idx_{name}"] = getattr(agent, "idx", 0)
                if name.startswith("battery"):
                    step_record[f"soc_{name}"] = getattr(agent, "soc", None)
                    step_record[f"soc_idx_{name}"] = getattr(agent, "idx", None)

            # Update environment global variables
            env.energy_balance = env.total_power - env.demand_power
            env.delta_power_idx = "surplus" if env.energy_balance >= 0 else "deficit"

            # Step log: Append environment globals at the end (preserve insertion order)
            step_record.update({
                "env_total_renewable": env.renewable_power,
                "env_total_power": env.total_power,
                "env_demand_power": env.demand_power,
                "env_demand_power_idx": env.demand_power_idx,
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

                # Requiere reward_fn definido por YAML
                if getattr(agent, "reward_fn", None) is None:
                    raise RuntimeError(
                        f"El agente {name} no tiene reward_fn configurado. Define 'agents.{agent.name.split('#')[0]}.reward' en el YAML."
                    )
                reward = agent.reward_fn.compute(agent, env, state_tuple)

                # Q-learning update
                agent.update_q_table(state_tuple, agent.action, reward, next_state_tuple)

                # Log per-agent values
                step_record[f"reward_{name}"] = reward

            evolution.append(step_record)

        # 6. Epsilon update (según scheduler configurado)
        epsilon = scheduler(episode, epsilon)

        # 7. Save episode data
        episode_df = pd.DataFrame(evolution)
        episode_df.to_csv(f"results/evolution/episode_{episode}.csv", index=False)
        results.append(episode_df)

    logger.info("Episode %d/%d completed | epsilon=%.3f", episode + 1, num_episodes, epsilon)

    return agents, results
