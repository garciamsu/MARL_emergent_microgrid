"""LoadAgent reward reporting utilities.

Computes KPIs, saves summaries and creates visualizations for load agent
state-action CSVs.
"""

import csv
import math
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal configuration
AGENT_NAME = "load"
OUTPUT_ROOT = "./test"
INPUT_FILE_PRIMARY = "reward_load.csv"
INPUT_FILE_FALLBACK = (
    f"{OUTPUT_ROOT}/{AGENT_NAME}/output/reward_load.csv"
)
MAKE_PLOTS = True
WEIGHTS = {
    "mean_balance": 1.0,
    "IAE_mean": 1.0,
    "ISE_mean": 1.0,
    "variability": 1.0,
    "reward_mean": 1.0,
}
SCALE_FACTOR_REWARD = 50.0


def safe_mkdir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights so they sum to 1, or return equal weights."""
    total_weight = sum(max(0.0, float(v)) for v in weights.values())
    if total_weight <= 0:
        count = len(weights)
        return {k: 1.0 / count for k in weights.keys()}
    return {k: float(max(0.0, v)) / total_weight for k, v in weights.items()}


def resolve_input_path() -> Path:
    """Resolve the preferred input CSV path for this script."""
    here = Path(__file__).parent
    primary_path = here / INPUT_FILE_PRIMARY
    if primary_path.exists():
        return primary_path
    return Path(INPUT_FILE_FALLBACK)


def load_state_action_csv(path: Path) -> pd.DataFrame:
    """Load CSV and ensure expected columns exist; compute optional dP."""
    data_frame = pd.read_csv(path, encoding="utf-8")
    required = [
        "battery_soc_idx",
        "renewable_potential_idx",
        "comfort_idx",
        "action",
        "reward",
    ]
    missing = [c for c in required if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    # Convert numeric columns to int when possible. Some CSVs encode
    # `comfort_idx` as categorical strings (e.g. 'acceptable', 'expensive'),
    # so handle it separately: keep strings if conversion fails.
    for col in ["battery_soc_idx", "renewable_potential_idx", "action"]:
        data_frame[col] = pd.to_numeric(data_frame[col], errors="coerce").fillna(0).astype(int)

    # Handle comfort_idx robustly: try numeric conversion, otherwise keep as string
    if "comfort_idx" in data_frame.columns:
        comfort_numeric = pd.to_numeric(data_frame["comfort_idx"], errors="coerce")
        if comfort_numeric.isna().any():
            # Keep original as categorical string
            data_frame["comfort_idx"] = data_frame["comfort_idx"].astype(str)
        else:
            data_frame["comfort_idx"] = comfort_numeric.astype(int)
    data_frame["reward"] = pd.to_numeric(data_frame["reward"], errors="coerce")
    if "total_power_idx" in data_frame.columns and "demand_power_idx" in data_frame.columns:
        data_frame["dP"] = (
            data_frame["total_power_idx"] - data_frame["demand_power_idx"]
        )
    else:
        data_frame["dP"] = 0.0
    return data_frame


def compute_kpis(data_frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute KPI statistics for the provided DataFrame."""
    d_p = data_frame["dP"].to_numpy(dtype=float)
    mean_balance = float(np.mean(d_p))
    iae_mean = float(np.mean(np.abs(d_p)))
    ise_mean = float(np.mean(np.square(d_p)))
    variability = float(np.std(d_p, ddof=0))
    reward_mean = float(np.mean(data_frame["reward"].to_numpy(dtype=float)))
    return {
        "rows": int(len(data_frame)),
        "mean_balance": mean_balance,
        "IAE_mean": iae_mean,
        "ISE_mean": ise_mean,
        "variability": variability,
        "reward_mean": reward_mean,
    }


def compute_fitness(kpis: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Aggregate KPIs into a scalar fitness value."""
    weights_norm = normalize_weights(weights)
    score_balance = 1.0 / (1.0 + abs(float(kpis["mean_balance"])))
    score_iae = 1.0 / (1.0 + float(kpis["IAE_mean"]))
    score_ise = 1.0 / (1.0 + float(kpis["ISE_mean"]))
    score_var = 1.0 / (1.0 + float(kpis["variability"]))
    reward_mean_val = float(kpis["reward_mean"])
    score_reward = 0.5 * (math.tanh(reward_mean_val / float(SCALE_FACTOR_REWARD)) + 1.0)
    return float(
        weights_norm.get("mean_balance", 0.0) * score_balance
        + weights_norm.get("IAE_mean", 0.0) * score_iae
        + weights_norm.get("ISE_mean", 0.0) * score_ise
        + weights_norm.get("variability", 0.0) * score_var
        + weights_norm.get("reward_mean", 0.0) * score_reward
    )


def save_per_action_summary(data_frame: pd.DataFrame, out_csv: Path) -> None:
    """Save reward summary per action to CSV."""
    grouped = (
        data_frame.groupby("action")["reward"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    grouped.to_csv(out_csv, index=False, encoding="utf-8")


def plot_common(data_frame: pd.DataFrame, out_dir: Path) -> None:
    """Generate basic plots shared by load reports."""
    plt.figure(figsize=(9, 5))
    plt.hist(data_frame["reward"].to_numpy(dtype=float), bins=30)
    plt.title("Reward Distribution (LoadAgent)")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_distribution_load.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(data_frame["dP"].to_numpy(dtype=float), bins=30)
    plt.title("Energy Balance dP Distribution (LoadAgent)")
    plt.tight_layout()
    plt.savefig(out_dir / "balance_distribution_load.png")
    plt.close()


def append_to_kpi_log(
    agent_name: str,
    output_root: Path,
    input_file: Path,
    kpis: Dict[str, Any],
    fitness: float,
) -> Path:
    """Append KPIs and fitness to the agent KPI log CSV."""
    out_dir = Path(output_root) / agent_name / "output"
    safe_mkdir(out_dir)
    out_csv = out_dir / "kpi_eval.csv"
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "agent": agent_name,
        "input_file": str(input_file),
        "rows": int(kpis["rows"]),
        "mean_balance": kpis["mean_balance"],
        "IAE_mean": kpis["IAE_mean"],
        "ISE_mean": kpis["ISE_mean"],
        "variability": kpis["variability"],
        "reward_mean": kpis["reward_mean"],
        "fitness_score": fitness,
    }
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_csv


def main() -> None:
    """Main entrypoint: compute KPIs and create outputs for load agent."""
    input_path = resolve_input_path()
    output_root = Path(OUTPUT_ROOT)
    out_dir = output_root / AGENT_NAME / "output"
    safe_mkdir(out_dir)

    print(f"[DEBUG] Using input file: {input_path}")
    data_frame = load_state_action_csv(input_path)

    kpis = compute_kpis(data_frame)
    fitness = compute_fitness(kpis, WEIGHTS)

    out_kpi_csv = append_to_kpi_log(
        agent_name=AGENT_NAME,
        output_root=output_root,
        input_file=input_path,
        kpis=kpis,
        fitness=fitness,
    )
    out_action_summary = out_dir / "reward_summary_by_action_load.csv"
    save_per_action_summary(data_frame, out_action_summary)

    print("=== KPI SUMMARY (State-Action Grid) ===")
    for k in [
        "rows",
        "mean_balance",
        "IAE_mean",
        "ISE_mean",
        "variability",
        "reward_mean",
    ]:
        val = kpis[k]
        if isinstance(val, float):
            print(f"{k:>14}: {val:.6f}")
        else:
            print(f"{k:>14}: {val}")
    print(f"{'fitness_score':>14}: {fitness:.6f}")
    print(f"[+] Appended to: {out_kpi_csv}")
    print(f"[+] Per-action summary: {out_action_summary}")

    if MAKE_PLOTS:
        plot_common(data_frame, out_dir)
        # heatmaps by comfort
        for comfort in sorted(data_frame["comfort_idx"].unique()):
            subset = data_frame[data_frame["comfort_idx"] == comfort]
            heatmap_data = subset.pivot_table(
                index="renewable_potential_idx",
                columns="battery_soc_idx",
                values="reward",
                aggfunc="mean",
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                annot=False,
                fmt=".1f",
                cmap="YlGnBu",
                cbar_kws={"label": "Reward"},
            )
            plt.title(f"Heatmap of Avg Reward (comfort_idx = {comfort})")
            plt.tight_layout()
            plt.savefig(out_dir / f"heatmap_reward_comfort_{comfort}.png")
            plt.close()
        # rolling mean
        df_sorted = data_frame.sort_values(
            by=["battery_soc_idx", "renewable_potential_idx", "action"]
        )
        rolling_mean = df_sorted["reward"].rolling(window=100, min_periods=1).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_mean)
        plt.title("Rolling Mean of Reward (LoadAgent, Window=100)")
        plt.tight_layout()
        plt.savefig(out_dir / "rolling_mean_reward_load.png")
        plt.close()
        # action frequency
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data_frame, x="action")
        plt.title("Action Frequency (LoadAgent)")
        plt.tight_layout()
        plt.savefig(out_dir / "action_frequency_load.png")
        plt.close()
        print(f"[+] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
