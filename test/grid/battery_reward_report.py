import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def generate_reports(input_file: str, output_dir: Path):
    """
    Generates analytical reports and visualizations based on the GridAgent's reward output.
    
    Parameters:
    - input_file: path to CSV containing reward evaluations
    - output_dir: path where all reports and visualizations will be saved
    """
    # Load input data
    df = pd.read_csv(input_file, delimiter=',', engine='python')

    # Plot: boxplot of rewards per action
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='action', y='reward')
    plt.title("Reward Boxplot by Action (GridAgent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_reward_by_action_grid.png")
    plt.close()

    # Plot: histogram of rewards per action
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='reward', hue='action', multiple='stack', bins=50)
    plt.title("Reward Histogram by Action (GridAgent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_rewards_action_grid.png")
    plt.close()

    # Plot: heatmap of average rewards by total_power and demand_power for each SoC
    for soc in sorted(df['battery_soc_idx'].unique()):
        subset = df[df['battery_soc_idx'] == soc]
        heatmap_data = subset.pivot_table(
            index='total_power_idx', columns='demand_power_idx', values='reward', aggfunc='mean'
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Reward'})
        plt.title(f"Heatmap of Average Reward (SoC={soc})")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_reward_soc_{soc}.png")
        plt.close()

    # Table: summary of reward statistics per action
    summary = df.groupby('action')['reward'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    summary.to_csv(output_dir / "reward_summary_by_action_grid.csv", index=False, encoding='utf-8')

    # Plot: global reward distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['reward'], bins=60, kde=True)
    plt.title("Global Reward Distribution (GridAgent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "reward_distribution_grid.png")
    plt.close()

    # Plot: rolling mean of reward
    df_sorted = df.sort_values(by=['battery_soc_idx', 'total_power_idx', 'demand_power_idx'])
    rolling_mean = df_sorted['reward'].rolling(window=100, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_mean)
    plt.title("Rolling Mean of Reward (GridAgent, Window=100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_mean_reward_grid.png")
    plt.close()

    # Plot: frequency of action usage
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='action')
    plt.title("Action Frequency (GridAgent)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "action_frequency_grid.png")
    plt.close()

    print(f"✅ Reports successfully generated in: {output_dir}")

if __name__ == "__main__":
    input_file = Path(__file__).parent / 'reports' / 'reward_grid.csv'
    output_dir = Path(__file__).parent / 'reports'
    generate_reports(input_file, output_dir)
