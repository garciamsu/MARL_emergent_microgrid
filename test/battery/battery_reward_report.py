#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battery_reward_report.py (no-CLI version)
=========================================

Purpose
-------
Autonomous KPI evaluator for a single-day (24-step) MARL energy episode focused on the **battery** agent.
This version does **not** accept command-line arguments; all configuration is defined via **internal variables**
at the top of this script.

**Inputs**
- A CSV file with exactly ONE episode (24 rows by default), including these columns:
  episode, step, total_power_idx, demand_power_idx, battery_soc_idx, action, reward

**Core KPIs**
- mean_balance               : mean of dP, where dP_t = total_power_idx_t - demand_power_idx_t
- IAE (Integral Absolute Error) : sum(|dP_t|) over the 24 steps
- ISE (Integral Square Error)   : sum((dP_t)^2) over the 24 steps
- variability (Energy Balance Variability) : standard deviation of dP over the 24 steps
- total_reward               : sum of reward over the 24 steps
- fitness_score              : scalar objective combining the above (weights configurable below)

**Outputs**
- Appends one row to /test/{agent_name}/output/kpi_eval.csv (created if missing) with KPIs and metadata
- Automatically generates charts and a behavior map in /test/{agent_name}/output/ (PNG + XLSX)

Usage
-----
Simply run:
    python battery_reward_report.py

Adjust the INTERNAL VARIABLES below to point to your files and preferences.

Author
------
Generated for Juan Carlos (MARL research) — Documentation in English, explanations in code comments.
"""

# =========================
# INTERNAL VARIABLES (EDIT)
# =========================
AGENT_NAME   = "battery"                                   # Agent name to build output path
OUTPUT_ROOT  = "./test"  # Changed to a relative path for user-accessible directory
INPUT_FILE   = f"{OUTPUT_ROOT}/{AGENT_NAME}/output/reward_battery.csv"  # Updated to use dynamic path based on OUTPUT_ROOT and AGENT_NAME
EXPECTED_STEPS = 24                                           # Expected steps per episode (default: 24)
MAKE_PLOTS   = True                                           # Always generate plots/behavior map as requested
WEIGHTS      = {                                              # Fitness weights (will be normalized internally)
    "mean_balance": 1.0,
    "IAE": 1.0,
    "ISE": 1.0,
    "variability": 1.0,
    "reward": 1.0
}
SCALE_FACTOR_REWARD = 50.0                                    # Squashing factor for reward term in fitness

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
    """Return a normalized copy of weights so that sum = 1.0 (if positive total),
    otherwise fall back to equal weights."""
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights.keys()}
    return {k: float(max(0.0, v)) / total for k, v in weights.items()}


def load_episode_csv(path: Path, expected_steps: int) -> pd.DataFrame:
    """Read one-episode CSV and ensure required columns exist. Auto-generates episode and step columns if missing."""
    print(f"[DEBUG] Attempting to read CSV from: {path}")
    print(f"[DEBUG] File exists: {path.exists()}")
    if path.exists():
        print(f"[DEBUG] File size: {path.stat().st_size} bytes")
    
    df = pd.read_csv(path, encoding="utf-8")
    print(f"[DEBUG] CSV columns found: {list(df.columns)}")
    
    # Add missing episode/step columns if needed
    if 'episode' not in df.columns:
        print("[INFO] Auto-generating 'episode' column with value 0")
        df['episode'] = 0
    
    if 'step' not in df.columns:
        print("[INFO] Auto-generating 'step' column from index")
        df['step'] = range(len(df))
    
    required = ["episode", "step", "total_power_idx", "demand_power_idx", "battery_soc_idx", "action", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
        
    if len(df) != expected_steps:
        # Allow mismatch but warn, then proceed using available rows
        print(f"[WARN] Expected {expected_steps} steps, but found {len(df)}. Proceeding with available rows.")
    df = df.sort_values(by="step").reset_index(drop=True)
    return df


# =========================
# Behavior classification
# =========================
def classify_behavior_row(row: pd.Series) -> Tuple[str, str]:
    """Classify the agent's behavior based on context and action (no renewable index required)."""
    soc = int(row["battery_soc_idx"])
    action = int(row["action"])
    t = int(row["total_power_idx"])
    d = int(row["demand_power_idx"])
    power_gap = t - d  # surplus(+) / deficit(-)

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
        if power_gap > 0 and soc < 4:  # lightweight heuristic without renewable index
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
# KPI computation
# =========================
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    dP = (df["total_power_idx"] - df["demand_power_idx"]).to_numpy(dtype=float)
    mean_balance = float(np.mean(dP))
    IAE = float(np.sum(np.abs(dP)))
    ISE = float(np.sum(np.square(dP)))
    variability = float(np.std(dP, ddof=0))  # population std
    total_reward = float(np.sum(df["reward"].to_numpy(dtype=float)))
    return {
        "mean_balance": mean_balance,
        "IAE": IAE,
        "ISE": ISE,
        "variability": variability,
        "total_reward": total_reward
    }


def compute_fitness(kpis: Dict[str, Any], weights: Dict[str, float]) -> float:
    """Scalar fitness using inverse-like transforms for error terms + squashed reward."""
    w = normalize_weights(weights)
    sb   = 1.0 / (1.0 + abs(float(kpis["mean_balance"])))
    sIAE = 1.0 / (1.0 + float(kpis["IAE"]))
    sISE = 1.0 / (1.0 + float(kpis["ISE"]))
    svar = 1.0 / (1.0 + float(kpis["variability"]))
    r    = float(kpis["total_reward"])
    srew = 0.5 * (math.tanh(r / float(SCALE_FACTOR_REWARD)) + 1.0)  # in [0,1]
    return float(
        w.get("mean_balance", 0.0) * sb
      + w.get("IAE", 0.0)         * sIAE
      + w.get("ISE", 0.0)         * sISE
      + w.get("variability", 0.0) * svar
      + w.get("reward", 0.0)      * srew
    )


# =========================
# Visualization (matplotlib)
# =========================
def plot_reward_histogram(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5))
    rewards = df["reward"].to_numpy(dtype=float)
    plt.hist(rewards, bins=20)
    plt.title("Reward Distribution (Episode)")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_balance_timeseries(df: pd.DataFrame, out_png: Path) -> None:
    steps = df["step"].to_numpy(dtype=int)
    dP = (df["total_power_idx"] - df["demand_power_idx"]).to_numpy(dtype=float)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, dP, marker="o")
    plt.title("Energy Balance dP over Time (Episode)")
    plt.xlabel("Step (hour)")
    plt.ylabel("dP = total_idx - demand_idx")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_action_frequency(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["action"].value_counts().sort_index()
    labels = [str(int(a)) for a in counts.index.tolist()]
    values = counts.to_numpy(dtype=int)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Action Frequency (Episode)")
    plt.xlabel("Action (0=idle,1=charge,2=discharge)")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.3)
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
                      kpis: Dict[str, Any], fitness: float, steps: int) -> Path:
    out_dir = Path(output_root) / agent_name / "output"
    safe_mkdir(out_dir)
    out_csv = out_dir / "kpi_eval.csv"
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "agent": agent_name,
        "input_file": str(input_file),
        "steps": int(steps),
        "mean_balance": kpis["mean_balance"],
        "IAE": kpis["IAE"],
        "ISE": kpis["ISE"],
        "variability": kpis["variability"],
        "total_reward": kpis["total_reward"],
        "fitness_score": fitness
    }
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_csv


# =========================
# Main execution
# =========================
def main() -> None:
    # Debug info
    print(f"[DEBUG] Current working directory: {Path.cwd()}")
    print(f"[DEBUG] INPUT_FILE path: {INPUT_FILE}")
    print(f"[DEBUG] OUTPUT_ROOT path: {OUTPUT_ROOT}")
    
    # Resolve paths
    input_path  = Path(INPUT_FILE)
    output_root = Path(OUTPUT_ROOT)
    out_dir     = output_root / AGENT_NAME / "output"
    
    print(f"[DEBUG] Resolved input_path: {input_path}")
    print(f"[DEBUG] Resolved out_dir: {out_dir}")
    safe_mkdir(out_dir)

    # Load episode
    df = load_episode_csv(input_path, expected_steps=EXPECTED_STEPS)

    # KPIs + fitness
    kpis = compute_kpis(df)
    fitness = compute_fitness(kpis, WEIGHTS)

    # Append to KPI log
    out_csv = append_to_kpi_log(agent_name=AGENT_NAME,
                                output_root=output_root,
                                input_file=input_path,
                                kpis=kpis, fitness=fitness, steps=len(df))

    # STDOUT summary
    print("=== KPI SUMMARY (Episode) ===")
    for k, v in kpis.items():
        print(f"{k:>14}: {v:.6f}")
    print(f"{'fitness_score':>14}: {fitness:.6f}")
    print(f"[+] Appended to: {out_csv}")

    # Plots & behavior map (always on per user request)
    plot_reward_histogram(df, out_dir / "reward_distribution.png")
    plot_balance_timeseries(df, out_dir / "balance_timeseries.png")
    plot_action_frequency(df, out_dir / "action_frequency.png")
    save_behavior_map(df, out_dir / "expected_behavior_map.xlsx")
    print(f"[+] Plots and behavior map saved in: {out_dir}")
    print(f"[DEBUG] Input file path: {INPUT_FILE}")  # Debugging output to verify file path


if __name__ == "__main__":
    main()