"""High-level training loop for the multi-agent microgrid.








































































    main()if __name__ == "__main__":    print("\n" + "="*80)            sys.exit(1)        traceback.print_exc()        import traceback        print(f"\n❌ ERROR: {e}")    except Exception as e:                        print(f"\n❌ Metadata file NOT found")        else:                                print(f"   Windows used more than once: {len(repeated_windows)}")                repeated_windows = non_summary[non_summary['frequency'] > 1]                                print(f"   Average frequency: {non_summary['frequency'].mean():.2f}")                print(f"   Min frequency: {non_summary['frequency'].min()}")                print(f"   Max frequency: {non_summary['frequency'].max()}")                print(f"   Unique windows: {len(non_summary)}")                print(f"\n\n📊 Statistics:")            if len(non_summary) > 0:            non_summary = window_frequency[window_frequency['window_start_index'] != 'SUMMARY']            # Statistics                        print(window_frequency.head(15).to_string(index=False))            print(f"   Showing windows sorted by frequency (most used first):")            print(f"\n\n📈 Window Frequency Analysis Sheet:")                        print(episode_details.head(10).to_string(index=False))            print(f"   Total episodes: {len(episode_details)}")            print(f"\n📊 Episode Details Sheet:")                        window_frequency = pd.read_excel(metadata_path, sheet_name='Window Frequency', engine='openpyxl')            episode_details = pd.read_excel(metadata_path, sheet_name='Episode Details', engine='openpyxl')            # Read both sheets                        print(f"\n✅ Metadata file created: {metadata_path}")        if metadata_path.exists():        metadata_path = Path("results/logs/episode_metadata.xlsx")                print("\n✅ Training completed!")                agents, results = run_training(config)    try:        print(f"   Dataset: {config['simulation']['dataset']}\n")    print(f"\n📋 Running {config['simulation']['episodes']} episodes")        config['simulation']['episodes'] = 20  # Test with 20 episodes    original_episodes = config['simulation']['episodes']    config = load_config()        print("="*80)    print("🧪 Testing Window Frequency Analysis")    print("="*80)def main():from core.simulation import run_trainingfrom configs.loader import load_configsys.path.insert(0, str(Path(__file__).parent))import pandas as pdfrom pathlib import Pathimport sysThe current implementation performs an episodic tabular Q-learning procedure
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
import numpy as np
import os
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

    # --- LINEAR SCHEDULE ---
    if schedule == "linear":
        target = end if end is not None else min_eps

        # pendiente ajustada para terminar EXACTAMENTE en 'end'
        slope = (target - start) / max(1, (episodes - 1))

        def f(t: int, _prev: float) -> float:
            return clip(start + slope * t)

        return f

    # --- EXPONENTIAL SCHEDULE ---
    if schedule == "exponential":
        if decay is None:
            if end is not None and start > 0 and end > 0:
                decay_eff = (end / start) ** (1.0 / max(1, episodes - 1))
            else:
                decay_eff = 0.99
        else:
            decay_eff = decay

        def f(_t: int, prev: float) -> float:
            base = prev if prev is not None else start
            return clip(base * decay_eff)

        return f

    # --- CONSTANT SCHEDULE ---
    if schedule == "constant":
        def f(_t: int, _prev: float) -> float:
            return clip(start)
        return f

    # --- CUSTOM SCHEDULE ---
    if schedule == "custom":
        series = [clip(v) for v in values]
        if not series:
            series = [clip(start)] * episodes
        if len(series) < episodes:
            series += [series[-1]] * (episodes - len(series))

        def f(t: int, _prev: float) -> float:
            return series[min(t, len(series) - 1)]

        return f

    # fallback linear
    target = end if end is not None else min_eps
    slope = (target - start) / max(1, (episodes - 1))

    def f(t: int, _prev: float) -> float:
        return clip(start + slope * t)

    return f


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
    
    # Track episode metadata (window and initial SOC)
    episode_metadata = []

    for episode in range(num_episodes):
        # Check if this is the final episode (last one)
        is_final_episode = (episode == num_episodes - 1)
        
        # ==============================================
        # 1. Select 24-hour contiguous window
        # ==============================================
        full_dataset_length = len(env.full_dataset)
        
        if is_final_episode:
            # FINAL EPISODE: Use predefined window from config
            final_cfg = config.get("final_episode", {})
            start = int(final_cfg.get("start_index", 0))
            end = int(final_cfg.get("end_index", 24))
            
            # Validate indices are within dataset bounds
            if start < 0:
                logger.warning("Final episode start_index < 0. Setting to 0.")
                start = 0
            
            if end > full_dataset_length:
                logger.warning("Final episode end_index > dataset length. Setting to %d.", full_dataset_length)
                end = full_dataset_length
            
            if start >= end:
                logger.warning("Final episode start_index >= end_index. Using default [0:24].")
                start = 0
                end = min(24, full_dataset_length)
            
            episode_data = env.full_dataset.iloc[start:end].copy()
            window_size = end - start
            logger.info("FINAL EPISODE: Using predefined window [%d:%d] (%d hours)", start, end, window_size)
        else:
            # TRAINING EPISODES: Random 24-hour window
            if full_dataset_length >= 24:
                start = np.random.randint(0, full_dataset_length - 24)
                episode_data = env.full_dataset.iloc[start:start + 24].copy()
            else:
                # Fallback: if dataset is shorter than 24 hours, use full dataset
                episode_data = env.full_dataset.copy()
                logger.warning("Dataset shorter than 24 hours. Using full dataset for episode %d.", episode)
        
        # ==============================================
        # 2. Generate initial SOC for battery
        # ==============================================
        battery_cfg = config.get("agents", {}).get("battery", {}) or {}
        limits_cfg = battery_cfg.get("limits", {}) or {}
        
        if is_final_episode:
            # FINAL EPISODE: Use fixed initial SOC from config
            final_cfg = config.get("final_episode", {})
            initial_soc = float(final_cfg.get("initial_soc", 0.5))
            logger.info("FINAL EPISODE: Using fixed initial SOC = %.3f", initial_soc)
        else:
            # TRAINING EPISODES: Random initial SOC
            init_soc_min = float(limits_cfg.get("initial_soc_min", 0.1))
            init_soc_max = float(limits_cfg.get("initial_soc_max", 0.9))
            
            # Ensure valid range
            if init_soc_max < init_soc_min:
                init_soc_min, init_soc_max = init_soc_max, init_soc_min
            
            # Generate random initial SOC
            initial_soc = np.random.uniform(init_soc_min, init_soc_max)
        
        # ==============================================
        # 3. Reset environment with episode data and initial SOC
        # ==============================================
        env.reset(episode_data, initial_soc)
        
        # Record episode metadata
        if is_final_episode:
            # For final episode, use actual end index
            recorded_start = start
            recorded_end = end
        elif full_dataset_length >= 24:
            # For training episodes with valid window
            recorded_start = start
            recorded_end = start + 24
        else:
            # Fallback
            recorded_start = 0
            recorded_end = full_dataset_length
        
        episode_metadata.append({
            "episode": episode,
            "window_start_index": recorded_start,
            "window_end_index": recorded_end,
            "window_size": recorded_end - recorded_start,
            "initial_soc": initial_soc,
            "is_final_episode": is_final_episode
        })
        
        # ==============================================
        # 4. Set initial SOC for all battery agents
        # ==============================================
        soc_min_cfg = float(limits_cfg.get("soc_min", 0.0))
        soc_max_cfg = float(limits_cfg.get("soc_max", 1.0))
        
        for a_name, a in agents.items():
            if "battery" in a_name.lower():
                # Clip to configured battery limits
                a.soc = max(soc_min_cfg, min(soc_max_cfg, initial_soc))
                # Update discrete index according to battery bins
                if hasattr(a, "battery_soc_bins"):
                    a.idx = digitize_clip(a.soc, a.battery_soc_bins)
                
                # Verification log
                if is_final_episode:
                    logger.info(
                        "✅ FINAL EPISODE - Battery SOC initialized: %.3f (idx=%d)",
                        a.soc, a.idx
                    )
                    if abs(a.soc - initial_soc) > 0.001:
                        logger.warning(
                            "⚠️  SOC mismatch! Expected %.3f, got %.3f",
                            initial_soc, a.soc
                        )
                else:
                    logger.debug(
                        "Episode %d - Battery SOC initialized: %.3f (idx=%d)",
                        episode, a.soc, a.idx
                    )

        evolution = []

        # 0. Record initial state (step -1) to capture initial SOC before any actions
        initial_state_record = {
            "episode": episode,
            "step": -1,
            "epsilon": 0.0
        }
        
        # Add initial agent states (especially battery SOC)
        for name, agent in agents.items():
            initial_state_record[f"action_{name}"] = None
            initial_state_record[f"power_{name}"] = 0.0
            initial_state_record[f"potential_{name}"] = None
            initial_state_record[f"idx_{name}"] = getattr(agent, "idx", 0)
            if name.startswith("battery"):
                initial_state_record[f"soc_{name}"] = agent.soc
                initial_state_record[f"soc_idx_{name}"] = agent.idx
        
        # Add initial environment states
        initial_state_record.update({
            "env_price": 0.0,
            "env_renewable_potential": 0.0,
            "env_renewable_potential_idx": 0,
            "env_total_renewable": 0.0,
            "env_total_renewable_idx": 0,
            "env_total_power": 0.0,
            "env_total_power_idx": 0,
            "env_demand_power": 0.0,
            "env_demand_power_idx": 0,
            "env_energy_balance": 0.0,
            "env_energy_balance_idx": 0,
            "env_delta_power_idx": "surplus",
        })
        
        # Add rewards (all zero at initial state)
        for name in agents.keys():
            initial_state_record[f"reward_{name}"] = 0.0
        
        evolution.append(initial_state_record)

        # 1. Epsilon update
        if is_final_episode:
            # FINAL EPISODE: Pure exploitation (epsilon = 0)
            epsilon = 0.0
            logger.info("FINAL EPISODE: Setting epsilon = 0.0 (pure exploitation)")
        else:
            # TRAINING EPISODES: Use scheduler
            epsilon = scheduler(episode, epsilon)
        print(30*"*")
        # Use actual episode data length, stop before last index for next_state calculation
        episode_steps = len(episode_data) - 1 if is_final_episode else env.max_steps - 1
        for index in range(episode_steps):

            # 1. Discretized state per agent
            state = {
                name: agent.get_discretized_state(env, index)
                for name, agent in agents.items()
            }

            # Step log: environment global variables and per-agent fields
            step_record = {
                "episode": episode,
                "step": index,
                "epsilon": epsilon
            }

            # 2. Choose action per agent
            for agent in agents.values():
                agent.choose_action(state[agent.name], epsilon)

            # 3. Environment update based on agent actions
            # Sequential update order: Renewables → Load → Battery → Grid

            # Load base demand and price from episode data (24h window)
            data_source = env.episode_data if env.episode_data is not None else env.dataset
            base_demand_from_dataset = data_source.iloc[index]["demand"]
            env.get_dataset("demand", index)
            env.get_dataset("price", index)
            
            # Store base demand for load agent to use
            env.base_demand = base_demand_from_dataset
            
            # Reset power accumulators
            env.total_power = 0.0
            env.renewable_power = 0.0
            env.renewable_potential = 0.0
            
            # Update discretized indices    
            env.renewable_potential_idx = digitize_clip(env.renewable_potential, env.power_bins)
            env.renewable_power_idx = digitize_clip(env.renewable_power, env.power_bins)
            env.total_power_idx = digitize_clip(env.total_power, env.power_bins)

            # PHASE 1: Update renewable agents (solar, wind)
            for agent in agents.values():
                if "solar" in agent.name.lower() or "wind" in agent.name.lower():
                    agent.update_power(env)
                    env.renewable_potential += agent.potential
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

            # Update discretized indices    
            env.renewable_potential_idx = digitize_clip(env.renewable_potential, env.power_bins)
            env.renewable_power_idx = digitize_clip(env.renewable_power, env.power_bins)
            env.demand_power_idx = digitize_clip(env.demand_power, env.power_bins)
            env.total_power_idx = digitize_clip(env.total_power, env.power_bins)
            env.energy_balance_idx = digitize_clip(env.energy_balance, env.power_bins)

            # Step log: Append environment globals at the end (preserve insertion order)
            step_record.update({
                "env_price": env.price,
                "env_renewable_potential": env.renewable_potential,
                "env_renewable_potential_idx": env.renewable_potential_idx,
                "env_total_renewable": env.renewable_power,
                "env_total_renewable_idx": env.renewable_power_idx,
                "env_total_power": env.total_power,
                "env_total_power_idx": env.total_power_idx,
                "env_demand_power": env.demand_power,
                "env_demand_power_idx": env.demand_power_idx,
                "env_energy_balance": env.energy_balance,
                "env_energy_balance_idx": env.energy_balance_idx,
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

                # Q-learning update (skip in final episode for pure evaluation)
                if not is_final_episode:
                    agent.update_q_table(state_tuple, agent.action, reward, next_state_tuple)

                # Log per-agent values
                step_record[f"reward_{name}"] = reward

            evolution.append(step_record)



        # 7. Save episode data
        episode_df = pd.DataFrame(evolution)
        episode_df.to_csv(f"results/evolution/episode_{episode}.csv", index=False)
        results.append(episode_df)
        
        logger.info("Episode %d/%d completed | epsilon=%.3f", episode + 1, num_episodes, epsilon)

    # Save episode metadata to Excel with frequency analysis
    metadata_df = pd.DataFrame(episode_metadata)
    os.makedirs("results/logs", exist_ok=True)
    metadata_excel_path = "results/logs/episode_metadata.xlsx"
    
    # Create frequency analysis of windows
    window_frequency = metadata_df.groupby(['window_start_index', 'window_end_index']).agg(
        frequency=('episode', 'count'),
        episodes=('episode', lambda x: ', '.join(map(str, sorted(x))))
    ).reset_index()
    window_frequency = window_frequency.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    # Calculate statistics
    total_episodes = len(metadata_df)
    unique_windows = len(window_frequency)
    max_frequency = window_frequency['frequency'].max()
    min_frequency = window_frequency['frequency'].min()
    avg_frequency = window_frequency['frequency'].mean()
    
    # Add summary row
    summary_data = {
        'window_start_index': ['SUMMARY'],
        'window_end_index': [''],
        'frequency': [f'Total Episodes: {total_episodes}'],
        'episodes': [f'Unique Windows: {unique_windows}, Max Freq: {max_frequency}, Min Freq: {min_frequency}, Avg Freq: {avg_frequency:.2f}']
    }
    summary_df = pd.DataFrame(summary_data)
    window_frequency_with_summary = pd.concat([window_frequency, summary_df], ignore_index=True)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(metadata_excel_path, engine='openpyxl') as writer:
        metadata_df.to_excel(writer, sheet_name='Episode Details', index=False)
        window_frequency_with_summary.to_excel(writer, sheet_name='Window Frequency', index=False)
    
    logger.info("Episode metadata saved to %s", metadata_excel_path)
    logger.info("Window frequency analysis: %d unique windows used across %d episodes", unique_windows, total_episodes)

    return agents, results
