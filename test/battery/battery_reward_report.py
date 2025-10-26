#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battery_reward_report.py (state-action space evaluator, no-CLI)
================================================================

Purpose
-------
Autonomous KPI evaluator for a **full discrete state–action space** of the battery agent.
Each CSV row is treated as an independent (state, action, reward) sample — **not** a temporal step in an episode.

This script does **not** accept command-line arguments; all configuration is defined via **internal variables**
at the top of this file.

Expected Input
--------------
A CSV containing the **complete (or partial) combinatorial grid** of discretized states and actions, with columns:
    total_power_idx, demand_power_idx, battery_soc_idx, action, reward

Key Definitions
---------------
- dP := total_power_idx - demand_power_idx  (dimensionless, discretized)
- Each row is a (state, action) evaluation with an associated reward.

KPIs (Global, statistical)
--------------------------
- mean_balance          : mean(dP)
- IAE_mean              : mean(|dP|)
- ISE_mean              : mean(dP^2)
- variability           : std(dP)              # population standard deviation
- reward_mean           : mean(reward)
- fitness_score         : scalar score combining the above (weights configurable)

Additional Outputs
------------------
- Per-action reward summary (mean, std, min, max, count)
- Behavior classification map (XLSX), using a rule-based heuristic
- Visualizations (PNG): reward histogram, balance distribution, heatmap of avg reward by SoC vs Action,
  scatter dP vs reward

Filesystem Outputs
------------------
- KPIs are appended to: {OUTPUT_ROOT}/{AGENT_NAME}/output/kpi_eval.csv
- Per-action summary to: {OUTPUT_ROOT}/{AGENT_NAME}/output/reward_summary_by_action.csv
- Visualization files + behavior map: saved under {OUTPUT_ROOT}/{AGENT_NAME}/output/

Usage
-----
Simply run:
    python battery_reward_report.py
Adjust internal variables if needed.

Author
------
Generated for Juan Carlos (MARL research) — Documentation in English, explanations in code comments.
"""

# =========================
# INTERNAL VARIABLES (EDIT)
# =========================
AGENT_NAME   = "battery"                # Agent name to build output path
OUTPUT_ROOT  = "./test"                 # Root where /{agent}/output/ will be created
INPUT_FILE_PRIMARY   = "reward_battery.csv"  # Default: same folder as this script
INPUT_FILE_FALLBACK  = f"{OUTPUT_ROOT}/{AGENT_NAME}/output/reward_battery.csv"  # Fallback path

MAKE_PLOTS   = True                     # Always generate plots/behavior map as requested
WEIGHTS      = {                        # Fitness weights (will be normalized internally)
    "mean_balance": 1.0,
    "IAE_mean": 1.0,
    "ISE_mean": 1.0,
    "variability": 1.0,
    "reward_mean": 1.0
}
SCALE_FACTOR_REWARD = 50.0              # Squashing for reward in fitness

# =========================
# Imports
# =========================
import csv
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Utilities
# =========================
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights so sum=1 (if total>0); otherwise return equal weights."""
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights.keys()}
    return {k: float(max(0.0, v)) / total for k, v in weights.items()}


def resolve_input_path() -> Path:
    """Try to read from script directory first, fallback to OUTPUT_ROOT/{agent}/output path."""
    here = Path(__file__).parent
    p1 = here / INPUT_FILE_PRIMARY
    if p1.exists():
        return p1
    p2 = Path(INPUT_FILE_FALLBACK)
    return p2


def load_state_action_csv(path: Path) -> pd.DataFrame:
    """Read CSV and ensure required columns exist."""
    print(f"[DEBUG] Reading CSV from: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    required = ["total_power_idx", "demand_power_idx", "battery_soc_idx", "action", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    # ensure dtypes are sensible
    for col in ["total_power_idx", "demand_power_idx", "battery_soc_idx", "action"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    # compute dP
    df["dP"] = df["total_power_idx"] - df["demand_power_idx"]
    return df


# =========================
# Behavior classification
# =========================
def classify_behavior_row(row: pd.Series) -> Tuple[str, str]:
    """Classify the agent's behavior based on context and action (no renewable index required)."""
    soc = int(row["battery_soc_idx"])
    action = int(row["action"])
    power_gap = int(row["dP"])  # surplus(+) / deficit(-)

    if action == 2:  # discharge
        if soc == 0:
            return "Invalid", "Discharging with empty battery is not allowed."
        elif power_gap < 0:
            return "Optimal", "Discharging to help during system deficit."
        elif power_gap > 0:
            return "Risky", "Discharging during power surplus may destabilize the system."
        else:
            return "Sub-optimal", "Discharging in equilibrium is allowed but not ideal."
    elif action == 1:  # charge
        if power_gap > 0 and soc < 4:  # without renewable index, assume surplus-based heuristic
            return "Optimal", "Charging with system surplus is desired."
        elif power_gap <= 0:
            return "Sub-optimal", "Charging without surplus may increase grid stress."
        else:
            return "Risky", "Charging at (near) full SOC may be unnecessary."
    elif action == 0:  # idle
        return "Invalid", "Idle action does not actively contribute to balancing."
    else:
        return "Invalid", "Unknown action code."


# =========================
# KPI computation (global statistics)
# =========================
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    dP = df["dP"].to_numpy(dtype=float)
    mean_balance = float(np.mean(dP))
    IAE_mean = float(np.mean(np.abs(dP)))
    ISE_mean = float(np.mean(np.square(dP)))
    variability = float(np.std(dP, ddof=0))  # population std
    reward_mean = float(np.mean(df["reward"].to_numpy(dtype=float)))
    return {
        "rows": int(len(df)),
        "mean_balance": mean_balance,
        "IAE_mean": IAE_mean,
        "ISE_mean": ISE_mean,
        "variability": variability,
        "reward_mean": reward_mean
    }


def compute_fitness(kpis: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Scalar fitness using inverse-like transforms for error terms + squashed reward mean."""
    w = normalize_weights(weights)
    sb   = 1.0 / (1.0 + abs(float(kpis["mean_balance"])))
    sIAE = 1.0 / (1.0 + float(kpis["IAE_mean"]))
    sISE = 1.0 / (1.0 + float(kpis["ISE_mean"]))
    svar = 1.0 / (1.0 + float(kpis["variability"]))
    r    = float(kpis["reward_mean"])
    srew = 0.5 * (math.tanh(r / float(SCALE_FACTOR_REWARD)) + 1.0)  # in [0,1]
    return float(
        w.get("mean_balance", 0.0) * sb
      + w.get("IAE_mean", 0.0)     * sIAE
      + w.get("ISE_mean", 0.0)     * sISE
      + w.get("variability", 0.0)  * svar
      + w.get("reward_mean", 0.0)  * srew
    )


# =========================
# Visualization (matplotlib)
# =========================
def plot_reward_histogram(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5))
    rewards = df["reward"].to_numpy(dtype=float)
    plt.hist(rewards, bins=30)
    plt.title("Reward Distribution (State-Action Grid)")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_balance_distribution(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5))
    dP = df["dP"].to_numpy(dtype=float)
    plt.hist(dP, bins=30)
    plt.title("Energy Balance dP Distribution (State-Action Grid)")
    plt.xlabel("dP = total_idx - demand_idx")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_heatmap_reward_soc_action(df: pd.DataFrame, out_png: Path) -> None:
    pivot = df.pivot_table(index="battery_soc_idx", columns="action", values="reward", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    data = pivot.to_numpy(dtype=float)
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Avg Reward")
    plt.title("Avg Reward by SoC (rows) vs Action (cols)")
    plt.xlabel("Action")
    plt.ylabel("battery_soc_idx")
    plt.xticks(ticks=range(pivot.shape[1]), labels=[str(c) for c in pivot.columns])
    plt.yticks(ticks=range(pivot.shape[0]), labels=[str(r) for r in pivot.index])
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_scatter_dp_reward(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.scatter(df["dP"].to_numpy(dtype=float), df["reward"].to_numpy(dtype=float), s=10, alpha=0.6)
    plt.title("dP vs Reward (State-Action Grid)")
    plt.xlabel("dP = total_idx - demand_idx")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_behavior_map(df: pd.DataFrame, out_xlsx: Path) -> None:
    labels, reasons = [], []
    for _, row in df.iterrows():
        lab, rea = classify_behavior_row(row)
        labels.append(lab); reasons.append(rea)
    out = df.copy()
    out["behavior"] = labels
    out["justification"] = reasons
    out.to_excel(out_xlsx, index=False)


# =========================
# Aggregation into CSV log
# =========================
def append_to_kpi_log(agent_name: str, output_root: Path, input_file: Path,
                      kpis: Dict[str, Any], fitness: float) -> Path:
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
        "fitness_score": fitness
    }
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_csv


def save_per_action_summary(df: pd.DataFrame, out_csv: Path) -> None:
    g = df.groupby("action")["reward"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    g.to_csv(out_csv, index=False, encoding="utf-8")


# =========================
# Main execution
# =========================
def main() -> None:
    here = Path(__file__).parent
    input_path = resolve_input_path()
    output_root = Path(OUTPUT_ROOT)
    out_dir = output_root / AGENT_NAME / "output"
    safe_mkdir(out_dir)

    print(f"[DEBUG] Using input file: {input_path}")
    df = load_state_action_csv(input_path)

    # KPIs + fitness
    kpis = compute_kpis(df)
    fitness = compute_fitness(kpis, WEIGHTS)

    # Append to KPI log
    out_kpi_csv = append_to_kpi_log(agent_name=AGENT_NAME,
                                    output_root=output_root,
                                    input_file=input_path,
                                    kpis=kpis, fitness=fitness)

    # Per-action summary
    out_action_summary = out_dir / "reward_summary_by_action.csv"
    save_per_action_summary(df, out_action_summary)

    # STDOUT summary
    print("=== KPI SUMMARY (State-Action Grid) ===")
    for k in ["rows", "mean_balance", "IAE_mean", "ISE_mean", "variability", "reward_mean"]:
        val = kpis[k]
        if isinstance(val, float):
            print(f"{k:>14}: {val:.6f}")
        else:
            print(f"{k:>14}: {val}")
    print(f"{'fitness_score':>14}: {fitness:.6f}")
    print(f"[+] Appended to: {out_kpi_csv}")
    print(f"[+] Per-action summary: {out_action_summary}")

    # Plots & behavior map
    if MAKE_PLOTS:
        plot_reward_histogram(df, out_dir / "reward_distribution.png")
        plot_balance_distribution(df, out_dir / "balance_distribution.png")
        plot_heatmap_reward_soc_action(df, out_dir / "heatmap_reward_soc_action.png")
        plot_scatter_dp_reward(df, out_dir / "scatter_dp_vs_reward.png")
        save_behavior_map(df, out_dir / "expected_behavior_map.xlsx")
        print(f"[+] Plots and behavior map saved in: {out_dir}")


if __name__ == "__main__":
    main()