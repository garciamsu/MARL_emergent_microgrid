import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import glob
import os
import csv

def load_latest_evolution_csv():
    """
    Searches for the most recent learning_XXX.csv file in /results/evolution/,
    reads its content with robust handling of separator and encoding,
    and returns a ready-to-process DataFrame.
    """
    import glob

    # 1. Search for all matching files
    files = glob.glob("results/evolution/learning_*.csv")
    if not files:
        raise FileNotFoundError("No learning_XXX.csv files found in /results/evolution/")

    # 2. Extract numbers and find the highest
    numbers = []
    for f in files:
        match = re.search(r"learning_(\d+)\.csv", f)
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        raise ValueError("No episode numbers found in file names.")

    max_number = max(numbers)
    latest_file = f"results/evolution/learning_{max_number}.csv"
    print(f"Selected file: {latest_file}")

    # 3. Try robust reading
    try:
        df = pd.read_csv(latest_file, sep=",", encoding="utf-8")
    except Exception as e1:
        print("First read failed, retrying with ';' and latin-1...")
        try:
            df = pd.read_csv(latest_file, sep=";", encoding="latin-1")
        except Exception as e2:
            raise RuntimeError(
                f"Could not read the file.\nFirst error: {e1}\nSecond error: {e2}"
            )

    return df

def plot_metric(df, field, ylabel, filename_svg):
    """
    Plots the evolution of a metric over episodes.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df['Episode'], df[field], marker='o')
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename_svg, format="svg")
    plt.close()

def compute_q_diff_norm(current_q, prev_q):
    """
    Computes the L2 norm of the difference between two Q-tables.
    
    current_q: dict of dicts, current Q-table.
    prev_q: dict of dicts, previous Q-table.
    
    Returns: float
    """
    diffs = []
    for state in current_q:
        for action in current_q[state]:
            curr = current_q[state][action]
            prev = prev_q.get(state, {}).get(action, 0.0)
            diffs.append((curr - prev)**2)
    return np.sqrt(sum(diffs))

def check_stability(df, iae_threshold, var_threshold=1.0):
    """
    Checks whether IAE and variance remain stable in the last episodes.
    
    df: DataFrame with metrics.
    iae_threshold: Acceptable IAE threshold.
    var_threshold: Acceptable variance threshold.
    
    Returns: dict with results.
    """
    # Filter last 200 episodes
    df_recent = df[df['Episode'] >= df['Episode'].max() - 200]
    
    iae_mean = df_recent["IAE"].mean()
    var_mean = df_recent["Var_dif"].mean()
    
    result = {
        "IAE_mean": iae_mean,
        "Var_mean": var_mean,
        "IAE_stable": iae_mean <= iae_threshold,
        "Var_stable": var_mean <= var_threshold
    }
    return result

def process_evolution_data(df):
    """
    Applies bat_state transformation and validates required columns.
    Returns the DataFrame ready for plotting.
    """
    required_columns = [
        "solar_state",
        "wind_state",
        "bat_state",
        "bat_soc",
        "grid_state",
        "dif",
        "demand"
    ]

    # Check columns
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' is not in the DataFrame.")

    # Copy to avoid modifying the original
    df_plot = df.copy()

    # Transform bat_state according to rules
    def transform_bat_state(x):
        if x == 0:
            return 0
        elif x == 1:
            return -1
        elif x == 2:
            return 1
        else:
            return 0  # Safety default

    df_plot["bat_state_transformed"] = df_plot["bat_state"].apply(transform_bat_state)

    return df_plot

def plot_coordination(df):
    """
    Generates an SVG plot with 6 vertically aligned subplots:
    Solar, Wind, Battery (State + SOC), Grid, Demand, Dif.
    Uses bars and lines as appropriate.
    """
    # Fixed color mapping
    colors = {
        "solar_potential": "orange",
        "wind_potential": "blue",
        "bat_state": "green",
        "bat_soc": "green",
        "grid": "purple",
        "demand": "black",
        "dif": "red"
    }

    fig, axes = plt.subplots(
        6, 1,
        figsize=(12, 18),
        sharex=True,
        constrained_layout=True
    )

    # Dynamic X-axis based on number of rows
    time = list(range(1, len(df)+1))

    # Subplot (A): Solar
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(time, df["solar_state"], color=colors["solar_potential"], label="Solar State")
    ax2.plot(
        time,
        df["solar_potential"],
        linestyle="--",
        color=colors["solar_potential"],
        linewidth=2.5,
        label="Solar Power"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Power")
    ax.set_title("(A)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subplot (B): Wind
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(time, df["wind_state"], color=colors["wind_potential"], label="Wind State")
    ax2.plot(
        time,
        df["wind_potential"],
        linestyle="--",
        color=colors["wind_potential"],
        linewidth=2.5,
        label="Wind Power"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Power")
    ax.set_title("(B)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subplot (C): Battery State + SOC
    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(time, df["bat_state_transformed"], color=colors["bat_state"], label="Battery State")
    ax2.plot(
        time,
        df["bat_soc"],
        linestyle="--",
        color=colors["bat_soc"],
        linewidth=2.5,
        label="Battery SOC"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("SOC [0-1]")
    ax.set_title("(C)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subplot (D): Grid
    ax = axes[3]
    ax2 = ax.twinx()
    ax.bar(time, df["grid_state"], color=colors["grid"], label="Grid State")
    ax2.plot(
        time,
        df["price"],
        linestyle="--",
        color=colors["grid"],
        linewidth=2.5,
        label="Price"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Price")
    ax.set_title("(D)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subplot (E): Demand + Load State
    ax = axes[4]
    ax2 = ax.twinx()
    ax.bar(
        time,
        df["load_state"],
        color=colors["demand"],
        alpha=0.6,
        label="Load State"
    )
    ax2.plot(
        time,
        df["demand"],
        linestyle="--",
        color=colors["demand"],
        linewidth=2.5,
        label="Demand"
    )
    ax.set_ylabel("Load State")
    ax2.set_ylabel("Demand Power")
    ax.set_title("(E)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subplot (F): Dif (area)
    ax = axes[5]
    ax.fill_between(
        time,
        0,
        df["dif"],
        color=colors["dif"],
        alpha=0.5,
        label="Energy Balance"
    )
    ax.set_ylabel("Power")
    ax.set_xlabel("Time Steps")
    ax.set_title("(F)", loc="center", pad=10)
    ax.grid(True, which='both')
    ax.legend(loc="upper right")

    # X-axis ticks
    for ax in axes:
        ax.set_xticks(time)

    # Save SVG
    output_path = "results/plots/plot_coordination_last.svg"
    fig.savefig(output_path, format="svg")
    print(f"Plot saved at {output_path}")

    plt.show()

def clear_results_directories():
    """
    Deletes all files inside the directories:
    results/, results/evolution/, results/plots/.
    Does not delete the directories themselves, only their contents.
    """
    directories = [
        "results/",
        "results/evolution/",
        "results/plots/",
        "results/logs/batteryagent",
        "results/logs/gridagent",
        "results/logs/loadagent",
        "results/logs/solaragent",
        "results/logs/windagent",
    ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
            continue

        files = glob.glob(os.path.join(dir_path, "*"))
        if not files:
            print(f"No files in {dir_path}")
            continue

        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
            else:
                print(f"Ignored (not a file): {file_path}")

    print("Cleanup completed.")

def digitize_clip(value: float, bins: np.ndarray) -> int:
    # Robust and reusable discretization
    idx = np.digitize([value], bins)[0] - 1
    idx = np.clip(idx, 0, len(bins)-2)        # avoid -1 and last overflow
    return int(idx)

def log_q_update(
    agent_type: str,
    episode: int,
    step: int,
    state,
    action,
    reward,
    next_state,
    current_q,
    max_next_q,
    updated_q,
    epsilon: float
):
    """
    Logs Q-table update information for each agent per time step.

    Parameters:
        agent_type (str): Agent type name (e.g., 'SolarAgent').
        episode (int): Episode number.
        step (int): Step within episode.
        state (tuple): Current state.
        action (int): Action taken.
        reward (float): Reward received.
        next_state (tuple): Resulting state.
        current_q (float): Q(s,a) before update.
        max_next_q (float): max_a' Q(s', a') used in update.
        updated_q (float): Q(s,a) after update.
        epsilon (float): Current exploration rate.
    """
    log_dir = f"results/logs/{agent_type.lower()}/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{agent_type.lower()}_log.csv")
    
    header = [
        "episode", "step", "state", "action", "reward", "next_state",
        "current_q", "max_next_q", "updated_q", "epsilon"
    ]

    row = [
        episode, step, str(state), action, reward, str(next_state),
        current_q, max_next_q, updated_q, epsilon
    ]

    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
