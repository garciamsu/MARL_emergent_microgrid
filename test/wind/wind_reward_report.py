"""WindAgent reward report utilities.

Provides KPI computation, fitness scoring and plotting for wind agent
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
AGENT_NAME = "wind"
OUTPUT_ROOT = "./test"
INPUT_FILE_PRIMARY = "reward_wind.csv"
INPUT_FILE_FALLBACK = (
    f"{OUTPUT_ROOT}/{AGENT_NAME}/output/reward_wind.csv"
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
    """Create directory if not exists."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_input_path() -> Path:
    """Resolve input CSV path: prefer script dir, fallback to test output."""
    here = Path(__file__).parent
    primary_path = here / INPUT_FILE_PRIMARY
    if primary_path.exists():
        return primary_path
    return Path(INPUT_FILE_FALLBACK)


def load_state_action_csv(path: Path) -> pd.DataFrame:
    """Load CSV and validate required columns.

    Expected: total_power_idx, demand_power_idx, action, reward
    """
    data_frame = pd.read_csv(path, encoding="utf-8")
    required = ["total_power_idx", "demand_power_idx", "action", "reward"]
    missing = [c for c in required if c not in data_frame.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    for col in ["total_power_idx", "demand_power_idx", "action"]:
        data_frame[col] = pd.to_numeric(data_frame[col], errors="coerce").astype(int)
    data_frame["reward"] = pd.to_numeric(data_frame["reward"], errors="coerce")
    data_frame["dP"] = data_frame["total_power_idx"] - data_frame["demand_power_idx"]
    return data_frame


def compute_kpis(data_frame: pd.DataFrame) -> Dict[str, Any]:
    """Compute KPIs from the DataFrame and return a dict of metrics."""
    d_p = data_frame["dP"].to_numpy(dtype=float)
    return {
        "rows": int(len(data_frame)),
        "mean_balance": float(np.mean(d_p)),
        "IAE_mean": float(np.mean(np.abs(d_p))),
        "ISE_mean": float(np.mean(np.square(d_p))),
        "variability": float(np.std(d_p, ddof=0)),
        "reward_mean": float(np.mean(data_frame["reward"].to_numpy(dtype=float))),
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights or return equal weights if total is zero."""
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        count = len(weights)
        return {k: 1.0 / count for k in weights.keys()}
    return {k: float(max(0.0, v)) / total for k, v in weights.items()}


def compute_fitness(kpis: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Combine KPIs into a scalar fitness score using normalized weights."""
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
    """Save per-action reward summary to CSV."""
    grouped = (
        data_frame.groupby("action")["reward"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    grouped.to_csv(out_csv, index=False, encoding="utf-8")


def plot_common(data_frame: pd.DataFrame, out_dir: Path) -> None:
    """Generate common visualizations for the wind agent."""
    plt.figure(figsize=(9, 5))
    plt.hist(data_frame["reward"].to_numpy(dtype=float), bins=30)
    plt.title("Reward Distribution (WindAgent)")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_distribution_wind.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(data_frame["dP"].to_numpy(dtype=float), bins=30)
    plt.title("Energy Balance dP Distribution (WindAgent)")
    plt.tight_layout()
    plt.savefig(out_dir / "balance_distribution_wind.png")
    plt.close()


def append_to_kpi_log(
    agent_name: str,
    output_root: Path,
    input_file: Path,
    kpis: Dict[str, Any],
    fitness: float,
) -> Path:
    """Append KPIs + fitness to the agent log CSV file."""
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
    """Main: read CSV, compute KPIs and optionally generate plots."""
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
    out_action_summary = out_dir / "reward_summary_by_action_wind.csv"
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
        # action frequency
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data_frame, x="action")
        plt.title("Action Frequency (WindAgent)")
        plt.tight_layout()
        plt.savefig(out_dir / "action_frequency_wind.png")
        plt.close()
        print(f"[+] Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
