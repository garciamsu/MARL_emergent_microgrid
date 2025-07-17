import os
import pandas as pd
import matplotlib.pyplot as plt


def analyze_solaragent_log():
    # Input and output paths
    input_path = "results/logs/solaragent/solaragent_log.csv"
    output_dir = "results/logs/solaragent/"

    # Check if the log file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Log file not found at the specified path.")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the log file
    print("📥 Loading log file...")
    df = pd.read_csv(input_path)

    # ------------------------------
    # 1. Reward Summary by Action
    # ------------------------------
    print("📊 Generating reward summary by action...")
    reward_summary = df.groupby("action")["reward"].agg(
        count="count", mean="mean", std="std", min="min", max="max"
    ).reset_index()
    reward_summary.to_csv(os.path.join(output_dir, "rewards_summary.csv"), index=False)
    print("✅ Saved 'rewards_summary.csv'.")

    # ------------------------------
    # 2. Reward Distribution (Boxplot)
    # ------------------------------
    print("📦 Generating reward distribution boxplot...")
    plt.figure(figsize=(8, 5))
    data = [df[df["action"] == a]["reward"].values for a in sorted(df["action"].unique())]
    plt.boxplot(data, labels=[f"Action {int(a)}" for a in sorted(df["action"].unique())], showfliers=False)
    plt.title("Reward Distribution by Action")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rewards_boxplot.svg"))
    plt.close()
    print("✅ Saved 'rewards_boxplot.svg'.")

    # -----------------------------------------
    # 3. Rolling Mean of Reward (Dynamic Window)
    # -----------------------------------------
    print("📈 Calculating rolling mean of reward...")
    window = max(1, int(0.05 * len(df)))
    df["rolling_mean"] = df["reward"].rolling(window=window).mean()
    df[["step", "rolling_mean"]].to_csv(os.path.join(output_dir, "rolling_mean.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(df["step"], df["rolling_mean"])
    plt.title(f"Rolling Mean of Reward (window = {window} steps)")
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_mean.svg"))
    plt.close()
    print("✅ Saved 'rolling_mean.csv' and 'rolling_mean.svg'.")

    # ------------------------------
    # 4. Action Frequency
    # ------------------------------
    print("📊 Generating action frequency...")
    freq_abs = df["action"].value_counts().sort_index()
    freq_rel = (freq_abs / freq_abs.sum() * 100).round(2)
    freq_table = pd.DataFrame({"action": freq_abs.index, "count": freq_abs.values, "percentage": freq_rel.values})
    freq_table.to_csv(os.path.join(output_dir, "action_frequency.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(freq_abs.index.astype(str), freq_abs.values)
    plt.title("Action Frequency of Solar Agent")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_frequency.svg"))
    plt.close()
    print("✅ Saved 'action_frequency.csv' and 'action_frequency.svg'.")

    print("\n✅ Log analysis completed.")


if __name__ == "__main__":
    analyze_solaragent_log()