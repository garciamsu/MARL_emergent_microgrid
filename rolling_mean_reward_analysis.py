import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing the CSV episode files
folder_path = "./results/evolution/"
output_folder = "./results/stability"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize list to store episode rewards
episode_rewards = []

# Loop over all expected episode files
for i in range(2000):
    file_name = f"learning_{i}.csv"
    file_path = os.path.join(folder_path, file_name)

    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)

        # Compute total reward per timestep
        df["reward_total"] = (
            df["reward_solar"] +
            df["reward_wind"] +
            df["reward_bat"] +
            df["reward_grid"] +
            df["reward_demand"]
        )

        # Calculate 24-step rolling mean
        rolling_mean = df["reward_total"].rolling(window=24).mean()

        # Compute mean of rolling mean for the episode
        mean_reward = rolling_mean.mean()
        episode_rewards.append(mean_reward)
    else:
        episode_rewards.append(None)

# Create results DataFrame
results_df = pd.DataFrame({
    "episode": list(range(2000)),
    "rolling_mean_reward": episode_rewards
})

# Save results to CSV
csv_output_path = os.path.join(output_folder, "rolling_mean_rewards.csv")
results_df.to_csv(csv_output_path, index=False, encoding='utf-8')

# Plot rolling mean reward
plt.figure(figsize=(14, 7))
plt.plot(results_df["episode"], results_df["rolling_mean_reward"], label="Rolling Mean Reward (24h)", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward")
plt.title("Stability of Episodic Learning (MARL)")
plt.grid(True)
plt.legend()

# Save high-resolution SVG plot
svg_output_path = os.path.join(output_folder, "rolling_mean_rewards.svg")
plt.savefig(svg_output_path, format='svg', dpi=300)
plt.close()