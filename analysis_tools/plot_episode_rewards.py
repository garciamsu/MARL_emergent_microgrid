"""
Plot episode rewards correctly.

This script visualizes episode rewards (NOT cumulative across episodes).
Each episode's reward is the sum of timestep rewards WITHIN that episode only.

Expected data format:
    - episode_rewards.csv with columns: episode, agent1, agent2, ...
    - Each row represents ONE episode
    - Each value is the TOTAL reward for that agent in that episode

Output:
    - Line plot showing episode reward progression
    - Statistics summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path


def plot_episode_rewards(rewards_csv_path: str, output_dir: str = None):
    """
    Plot episode rewards from the episode_rewards.csv file.
    
    Args:
        rewards_csv_path: Path to episode_rewards.csv
        output_dir: Directory to save plots (default: same as CSV)
    """
    # Load episode rewards
    if not os.path.exists(rewards_csv_path):
        print(f"❌ File not found: {rewards_csv_path}")
        return
    
    df = pd.read_csv(rewards_csv_path)
    print(f"✅ Loaded {len(df)} episodes from {rewards_csv_path}")
    
    # Validate data format
    if 'episode' not in df.columns:
        print("❌ Missing 'episode' column in CSV")
        return
    
    # Get agent columns (all columns except 'episode')
    agent_columns = [col for col in df.columns if col != 'episode']
    
    if not agent_columns:
        print("❌ No agent columns found in CSV")
        return
    
    print(f"📊 Found {len(agent_columns)} agents: {agent_columns}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(rewards_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure matplotlib
    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-paper')
    
    # ========================================
    # Plot 1: Episode Rewards (NOT cumulative)
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colormap = plt.get_cmap('tab10')
    colors = [colormap(i) for i in range(len(agent_columns))]
    
    for idx, agent in enumerate(agent_columns):
        ax.plot(df['episode'], df[agent], 
                marker='o', markersize=3, linewidth=1.5,
                label=agent, color=colors[idx], alpha=0.8)
    
    ax.set_title('Episode Rewards (Per Episode)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Episode Reward', fontsize=13)
    ax.legend(title='Agent', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save plot
    plot_path = os.path.join(output_dir, 'episode_rewards.svg')
    plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_path}")
    plt.close()
    
    # ========================================
    # Plot 2: Moving Average (smoothed)
    # ========================================
    window_size = min(10, len(df) // 4)  # Adaptive window size
    if window_size >= 2:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for idx, agent in enumerate(agent_columns):
            moving_avg = df[agent].rolling(window=window_size, min_periods=1).mean()
            ax.plot(df['episode'], moving_avg,
                    linewidth=2.5, label=agent, color=colors[idx], alpha=0.9)
        
        ax.set_title(f'Episode Rewards (Moving Average, window={window_size})', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=13)
        ax.set_ylabel('Moving Average Reward', fontsize=13)
        ax.legend(title='Agent', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        
        # Save plot
        plot_path = os.path.join(output_dir, 'episode_rewards_moving_avg.svg')
        plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {plot_path}")
        plt.close()
    
    # ========================================
    # Print Statistics
    # ========================================
    print("\n" + "="*70)
    print("📊 EPISODE REWARD STATISTICS")
    print("="*70)
    
    stats = df[agent_columns].describe()
    print(stats.to_string())
    
    print("\n" + "="*70)
    print("📈 LEARNING PROGRESS")
    print("="*70)
    
    # Compare first 10% vs last 10% of episodes
    n_episodes = len(df)
    n_compare = max(1, n_episodes // 10)
    
    first_episodes = df.head(n_compare)[agent_columns].mean()
    last_episodes = df.tail(n_compare)[agent_columns].mean()
    improvement = last_episodes - first_episodes
    improvement_pct = (improvement / first_episodes.abs()) * 100
    
    comparison = pd.DataFrame({
        'First Episodes (avg)': first_episodes,
        'Last Episodes (avg)': last_episodes,
        'Improvement': improvement,
        'Improvement %': improvement_pct
    })
    
    print(f"\nComparing first {n_compare} vs last {n_compare} episodes:")
    print(comparison.to_string())
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    # Default path
    rewards_csv = "results/logs/episode_rewards.csv"
    
    # Check if file exists
    if not os.path.exists(rewards_csv):
        print(f"❌ File not found: {rewards_csv}")
        print("Please run training first to generate episode_rewards.csv")
    else:
        plot_episode_rewards(rewards_csv, output_dir="results/plots")
