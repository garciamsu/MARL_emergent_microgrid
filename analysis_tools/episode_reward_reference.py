"""
CORRECT EPISODE REWARD CALCULATION - REFERENCE IMPLEMENTATION

This module demonstrates the ONLY acceptable way to compute episode rewards
in a Reinforcement Learning environment.

KEY PRINCIPLES:
===============
1. episode_reward = sum of timestep_rewards WITHIN current episode ONLY
2. Reset episode_reward to 0.0 at the start of EACH episode
3. Never accumulate rewards across multiple episodes
4. Store ONE value per episode in episode_rewards list

INCORRECT PATTERNS TO AVOID:
============================
❌ Global cumulative sum across all episodes
❌ Appending timestep rewards to the plot
❌ Mixing episodes without reset
❌ Cumulative plots that never reset

CORRECT PATTERN:
================
✅ episode_rewards = []  # One value per episode
✅ episode_reward = 0.0  # Reset at start of each episode
✅ episode_reward += timestep_reward  # Accumulate within episode
✅ episode_rewards.append(episode_reward)  # Store at end of episode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


def compute_episode_rewards_correct(
    num_episodes: int,
    max_steps: int,
    reward_function
) -> Dict[str, List[float]]:
    """
    CORRECT implementation of episode reward calculation.
    
    This function demonstrates the ONLY acceptable pattern for computing
    episode rewards in reinforcement learning.
    
    Args:
        num_episodes: Total number of training episodes
        max_steps: Maximum timesteps per episode
        reward_function: Function that returns reward for a single timestep
        
    Returns:
        Dictionary mapping agent names to list of episode rewards.
        Each list has length = num_episodes.
        Each value = sum of timestep rewards for that episode ONLY.
    """
    # Initialize episode rewards storage
    # This will store ONE value per episode for each agent
    episode_rewards = {
        'agent_1': [],
        'agent_2': []
    }
    
    # Training loop
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # =====================================================
        # CRITICAL: Reset episode reward to 0.0 at start
        # =====================================================
        episode_reward_agent1 = 0.0
        episode_reward_agent2 = 0.0
        
        # Episode loop (timesteps within this episode)
        for step in range(max_steps):
            # Get timestep reward (this is the reward at time t)
            timestep_reward_agent1 = reward_function()
            timestep_reward_agent2 = reward_function()
            
            # =====================================================
            # CRITICAL: Accumulate timestep reward into episode reward
            # =====================================================
            episode_reward_agent1 += timestep_reward_agent1
            episode_reward_agent2 += timestep_reward_agent2
            
            print(f"  Step {step}: r1={timestep_reward_agent1:.3f}, "
                  f"r2={timestep_reward_agent2:.3f} | "
                  f"Episode totals: {episode_reward_agent1:.3f}, "
                  f"{episode_reward_agent2:.3f}")
        
        # =====================================================
        # CRITICAL: Store episode reward (sum of all timesteps)
        # =====================================================
        episode_rewards['agent_1'].append(episode_reward_agent1)
        episode_rewards['agent_2'].append(episode_reward_agent2)
        
        print(f"\n✅ Episode {episode} complete:")
        print(f"   Agent 1 episode reward: {episode_reward_agent1:.3f}")
        print(f"   Agent 2 episode reward: {episode_reward_agent2:.3f}")
        
        # Note: episode_reward is NOT carried over to next episode
        # It will be reset to 0.0 in the next iteration
    
    return episode_rewards


def plot_episode_rewards_correct(episode_rewards: Dict[str, List[float]]):
    """
    CORRECT visualization of episode rewards.
    
    This plots episode rewards (NOT cumulative across episodes).
    Each point represents the total reward for ONE episode.
    
    Args:
        episode_rewards: Dictionary with agent names as keys,
                        list of episode rewards as values
    """
    num_episodes = len(next(iter(episode_rewards.values())))
    episodes = list(range(num_episodes))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for agent_name, rewards in episode_rewards.items():
        ax.plot(episodes, rewards, marker='o', label=agent_name, linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Episode Rewards (Per Episode - NOT Cumulative)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correct_episode_rewards.png', dpi=300)
    print("\n✅ Plot saved as: correct_episode_rewards.png")


def example_incorrect_pattern():
    """
    ❌ INCORRECT PATTERN - DO NOT USE
    
    This demonstrates common mistakes to avoid.
    """
    print("\n" + "="*60)
    print("❌ INCORRECT PATTERN (DO NOT USE)")
    print("="*60)
    
    # WRONG: Global cumulative sum
    global_cumulative_reward = 0.0  # ❌ This is WRONG
    
    for episode in range(3):
        for step in range(5):
            timestep_reward = np.random.randn()
            
            # ❌ WRONG: Accumulating across ALL episodes
            global_cumulative_reward += timestep_reward
        
        print(f"Episode {episode}: cumulative = {global_cumulative_reward:.3f}")
        # ❌ WRONG: Never resets, keeps growing forever
    
    print("\n❌ This pattern is INCORRECT because:")
    print("   - Rewards accumulate across episodes")
    print("   - No way to see individual episode performance")
    print("   - Misleading learning curves")


def demo_correct_implementation():
    """
    Run a demonstration of the CORRECT episode reward calculation.
    """
    print("\n" + "="*60)
    print("✅ CORRECT PATTERN DEMONSTRATION")
    print("="*60)
    
    # Simple reward function for demo
    def random_reward():
        return np.random.randn()
    
    # Run correct implementation
    episode_rewards = compute_episode_rewards_correct(
        num_episodes=5,
        max_steps=10,
        reward_function=random_reward
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    df = pd.DataFrame(episode_rewards)
    df.insert(0, 'episode', range(len(df)))
    print("\n" + df.to_string(index=False))
    
    print("\n✅ Notice that each row is ONE episode")
    print("✅ Each value is the TOTAL reward for that episode ONLY")
    print("✅ Values can go up or down between episodes")
    print("✅ This shows the TRUE learning curve")
    
    # Plot results
    plot_episode_rewards_correct(episode_rewards)
    
    # Compare with incorrect pattern
    example_incorrect_pattern()


# ============================================================
# FORMULA REFERENCE
# ============================================================

CORRECT_FORMULA = """
CORRECT FORMULA:
===============

For each episode e:
    episode_reward[e] = Σ(timestep_reward[t]) for t in [0, T-1]
    
where T is the number of steps in episode e

Python implementation:
----------------------
episode_rewards = []

for episode in range(num_episodes):
    episode_reward = 0.0  # Reset at start
    
    for t in range(max_steps):
        timestep_reward = compute_reward(...)
        episode_reward += timestep_reward  # Accumulate within episode
    
    episode_rewards.append(episode_reward)  # Store episode total
    
# Result: episode_rewards[i] = total reward for episode i
"""


if __name__ == "__main__":
    print(CORRECT_FORMULA)
    print("\n" + "="*60)
    
    # Run demonstration
    demo_correct_implementation()
    
    print("\n" + "="*60)
    print("✅ REFERENCE IMPLEMENTATION COMPLETE")
    print("="*60)
    print("\nKey takeaways:")
    print("1. Reset episode_reward = 0.0 at START of each episode")
    print("2. Accumulate timestep rewards WITHIN episode only")
    print("3. Store ONE value per episode in episode_rewards list")
    print("4. NEVER accumulate across multiple episodes")
    print("="*60)
