import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def classify_behavior(row):
    """
    Classifies the agent's behavior based on the system's context and action taken.

    Returns a behavior label (Optimal, Sub-optimal, Risky, Invalid) and a justification string.
    """
    soc = row['battery_soc_idx']
    action = row['action']
    r = row['renewable_potential_idx']
    t = row['total_power_idx']
    d = row['demand_power_idx']
    power_gap = t - d

    if action == 2:  # discharge
        if soc == 0:
            return 'Invalid', 'Discharging with empty battery is not allowed.'
        elif power_gap < 0:
            return 'Optimal', 'Discharging to help during system deficit.'
        elif power_gap > 0:
            return 'Risky', 'Discharging during power surplus may destabilize the system.'
        else:
            return 'Sub-optimal', 'Discharging in equilibrium is allowed but not ideal.'
    elif action == 1:  # charge
        if r > d:
            return 'Optimal', 'Charging with renewable surplus is desired.'
        else:
            return 'Sub-optimal', 'Charging without renewable surplus may increase grid load.'
    elif action == 0:  # idle
        return 'Invalid', 'Idle action is not valid in any situation.'

def generate_reports(input_file: str, output_dir: Path):
    """
    Generates analytical reports and visualizations based on the battery agent's reward output.

    Parameters:
    - input_file: path to CSV containing reward evaluations
    - output_dir: path where all reports and visualizations will be saved
    """
    # Load input data
    df = pd.read_csv(input_file, delimiter=',', engine='python')

    # Classify expected behavior for each state-action
    df[['behavior', 'justification']] = df.apply(classify_behavior, axis=1, result_type='expand')

    # Save behavior map to Excel
    df.to_excel(output_dir / "expected_behavior_map.xlsx", index=False)

    # Plot: boxplot of rewards per behavior
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='behavior', y='reward')
    plt.title("Reward Boxplot by Expected Behavior")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_behavior.png")
    plt.close()

    # Plot: histogram of rewards per action
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='reward', hue='action', multiple='stack', bins=50)
    plt.title("Reward Histogram by Action")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_rewards_action.png")
    plt.close()

    # Plot: heatmap of average rewards by battery SoC and renewable potential
    heatmap_data = df.groupby(['battery_soc_idx', 'renewable_potential_idx'])['reward'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Reward'})
    plt.title("Heatmap of Average Reward by SoC and Renewable Potential")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_reward_soc_renewable.png")
    plt.close()

    # Table: summary of reward statistics per action
    summary = df.groupby('action')['reward'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    summary.to_csv(output_dir / "reward_summary_by_action.csv", index=False, encoding='utf-8')

    # Plot: global reward distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['reward'], bins=60, kde=True)
    plt.title("Global Reward Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "reward_distribution.png")
    plt.close()

    # Plot: rolling mean of reward
    df_sorted = df.sort_values(by=['battery_soc_idx', 'renewable_potential_idx'])
    rolling_mean = df_sorted['reward'].rolling(window=100, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_mean)
    plt.title("Rolling Mean of Reward (Window=100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_mean_reward.png")
    plt.close()

    # Plot: frequency of action usage
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='action')
    plt.title("Action Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "action_frequency.png")
    plt.close()

    print(f"✅ Reports successfully generated in: {output_dir}")

if __name__ == "__main__":
    input_file = Path(__file__).parent / 'reports' / 'reward_battery.csv'
    output_dir = Path(__file__).parent / 'reports'
    generate_reports(input_file, output_dir)

