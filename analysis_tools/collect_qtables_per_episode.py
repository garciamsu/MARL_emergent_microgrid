"""Data collection script for stability analysis.

This script runs a training session and collects Q-table snapshots after
each episode, enabling post-hoc stability analysis without modifying the
core training loop or agent implementations.

The script creates a deep copy of each agent's Q-table at the end of each
episode, storing them for later analysis.

Usage:
    python analysis_tools/collect_qtables_per_episode.py

Output:
    - Q-table snapshots saved to results/stability/qtables_per_episode.npz
    - Training proceeds normally with all standard outputs
"""

import sys
from pathlib import Path
import numpy as np
import copy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_config
from core.environment import MultiAgentEnv
from agents import instantiate_agents
from core.utils import set_global_seed, build_logger, ensure_dir
from utils.discretization import digitize_clip


def collect_qtables_snapshot(agents):
    """Create deep copy of all agent Q-tables.
    
    Args:
        agents: Dictionary of agent instances.
    
    Returns:
        Dictionary mapping agent names to Q-table copies.
    """
    snapshot = {}
    for name, agent in agents.items():
        q_table = getattr(agent, "q_table", {})
        if q_table:
            # Deep copy to preserve state at this episode
            snapshot[name] = {
                state: dict(actions)
                for state, actions in q_table.items()
            }
    return snapshot


def save_qtables_history(qtables_per_episode, output_path):
    """Save Q-table history to compressed numpy file.
    
    Args:
        qtables_per_episode: List of Q-table snapshots per episode.
        output_path: Path for output .npz file.
    """
    # Convert to serializable format
    # Structure: episode -> agent -> flattened (state, action, value) arrays
    
    data_to_save = {}
    
    for episode_idx, episode_qtables in enumerate(qtables_per_episode):
        for agent_name, qtable in episode_qtables.items():
            # Flatten Q-table for this agent at this episode
            states = []
            actions = []
            values = []
            
            for state, action_dict in qtable.items():
                for action, value in action_dict.items():
                    states.append(state)
                    actions.append(action)
                    values.append(value)
            
            if states:
                # Use keys like "episode_0_agent_solar#0_states"
                prefix = f"episode_{episode_idx}_agent_{agent_name}"
                data_to_save[f"{prefix}_states"] = np.array(states, dtype=object)
                data_to_save[f"{prefix}_actions"] = np.array(actions, dtype=int)
                data_to_save[f"{prefix}_values"] = np.array(values, dtype=float)
    
    # Save metadata
    data_to_save["num_episodes"] = len(qtables_per_episode)
    data_to_save["agent_names"] = np.array(
        list(qtables_per_episode[0].keys()) if qtables_per_episode else [],
        dtype=object
    )
    
    np.savez_compressed(output_path, **data_to_save)
    print(f"[OK] Q-table history saved to {output_path}")


def load_qtables_history(input_path):
    """Load Q-table history from compressed numpy file.
    
    Args:
        input_path: Path to .npz file.
    
    Returns:
        List of Q-table snapshots per episode.
    """
    data = np.load(input_path, allow_pickle=True)
    
    num_episodes = int(data["num_episodes"])
    agent_names = data["agent_names"].tolist()
    
    qtables_per_episode = []
    
    for episode_idx in range(num_episodes):
        episode_qtables = {}
        
        for agent_name in agent_names:
            prefix = f"episode_{episode_idx}_agent_{agent_name}"
            
            # Check if this agent has data for this episode
            states_key = f"{prefix}_states"
            if states_key not in data:
                continue
            
            states = data[states_key]
            actions = data[f"{prefix}_actions"]
            values = data[f"{prefix}_values"]
            
            # Reconstruct Q-table
            qtable = {}
            for state, action, value in zip(states, actions, values):
                state_tuple = tuple(state) if not isinstance(state, tuple) else state
                if state_tuple not in qtable:
                    qtable[state_tuple] = {}
                qtable[state_tuple][int(action)] = float(value)
            
            episode_qtables[agent_name] = qtable
        
        qtables_per_episode.append(episode_qtables)
    
    print(f"[OK] Loaded Q-table history: {num_episodes} episodes")
    
    return qtables_per_episode


def run_training_with_qtable_collection(config):
    """Run training and collect Q-table snapshots per episode.
    
    This is a simplified version of core.simulation.run_training that
    focuses on Q-table collection without modifying the training logic.
    
    Args:
        config: Configuration dictionary from load_config().
    
    Returns:
        List of Q-table snapshots per episode.
    """
    # Setup
    set_global_seed(config["simulation"]["seed"])
    logger = build_logger("qtable_collection")
    
    num_episodes = config["simulation"]["episodes"]
    logger.info(f"Starting training with Q-table collection: {num_episodes} episodes")
    
    # Initialize environment and agents
    env = MultiAgentEnv(config)
    agents = instantiate_agents(config, env)
    
    # Epsilon scheduler (simplified linear decay)
    epsilon_config = config["simulation"].get("epsilon", {})
    epsilon_start = epsilon_config.get("start", 1.0)
    epsilon_end = epsilon_config.get("end", 0.01)
    epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
    
    # Storage for Q-table snapshots
    qtables_per_episode = []
    
    # Get episode window configuration
    episode_window_hours = config.get("simulation", {}).get("episode_window_hours", 48)
    
    # Training loop
    for episode in range(num_episodes):
        # Compute current epsilon
        epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        
        # Select random contiguous window for episode
        full_dataset_length = len(env.full_dataset)
        
        if full_dataset_length >= episode_window_hours:
            start = np.random.randint(0, full_dataset_length - episode_window_hours)
            episode_data = env.full_dataset.iloc[start:start + episode_window_hours].copy()
        else:
            episode_data = env.full_dataset.copy()
            logger.warning(
                "Dataset shorter than configured window (%d hours). Using full dataset for episode %d.",
                episode_window_hours, episode
            )
        
        # Generate random initial SOC for battery
        battery_cfg = config.get("agents", {}).get("battery", {}) or {}
        limits_cfg = battery_cfg.get("limits", {}) or {}
        init_soc_min = float(limits_cfg.get("initial_soc_min", 0.1))
        init_soc_max = float(limits_cfg.get("initial_soc_max", 0.9))
        if init_soc_max < init_soc_min:
            init_soc_min, init_soc_max = init_soc_max, init_soc_min
        initial_soc = np.random.uniform(init_soc_min, init_soc_max)
        
        # Reset environment with episode data and initial SOC
        env.reset(episode_data, initial_soc)
        
        # Set initial SOC for battery agents
        soc_min_cfg = float(limits_cfg.get("soc_min", 0.0))
        soc_max_cfg = float(limits_cfg.get("soc_max", 1.0))
        
        for a_name, a in agents.items():
            if "battery" in a_name.lower():
                a.soc = max(soc_min_cfg, min(soc_max_cfg, initial_soc))
                if hasattr(a, "battery_soc_bins"):
                    a.idx = digitize_clip(a.soc, a.battery_soc_bins)
        
        # Reset agent state
        for agent in agents.values():
            agent.action = 0
            agent.power = 0
        
        # Run episode
        episode_length = len(env.episode_data) - 1
        
        for step in range(episode_length):
            index = step
            
            # Get states
            state = {
                name: agent.get_discretized_state(env, index)
                for name, agent in agents.items()
            }
            
            # Choose actions
            for name, agent in agents.items():
                state_tuple = state[name]
                agent.choose_action(state_tuple, epsilon)
            
            # Update agent powers (simplified - just call update_power)
            for name, agent in agents.items():
                agent.update_power(env)
            
            # Get next states
            if index + 1 < episode_length:
                next_state = {
                    name: agent.get_discretized_state(env, index + 1)
                    for name, agent in agents.items()
                }
                
                # Update Q-tables
                for name, agent in agents.items():
                    state_tuple = state[name]
                    next_state_tuple = next_state[name]
                    
                    if hasattr(agent, "reward_fn") and agent.reward_fn:
                        reward = agent.reward_fn.compute(agent, env, state_tuple)
                        agent.update_q_table(state_tuple, agent.action, reward, next_state_tuple)
        
        # Collect Q-table snapshot after episode
        snapshot = collect_qtables_snapshot(agents)
        qtables_per_episode.append(snapshot)
        
        if (episode + 1) % 10 == 0 or episode == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes} completed (ε={epsilon:.3f})")
    
    logger.info("Training completed with Q-table collection")
    
    return qtables_per_episode


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("Q-TABLE COLLECTION FOR STABILITY ANALYSIS")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    print(f"\nConfiguration:")
    print(f"   Episodes: {config['simulation']['episodes']}")
    print(f"   Dataset: {config['simulation']['dataset']}")
    print(f"   Seed: {config['simulation']['seed']}")
    
    # Run training with Q-table collection
    print(f"\nStarting training with Q-table collection...")
    qtables_per_episode = run_training_with_qtable_collection(config)
    
    # Save Q-table history
    output_dir = Path("results/stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qtables_per_episode.npz"
    
    save_qtables_history(qtables_per_episode, output_path)
    
    print(f"\n[OK] Collection complete!")
    print(f"   Episodes collected: {len(qtables_per_episode)}")
    print(f"   Output file: {output_path}")
    print(f"\nNext step: Run stability analysis")
    print(f"   python analysis_tools/run_stability_analysis.py")
    

if __name__ == "__main__":
    main()
