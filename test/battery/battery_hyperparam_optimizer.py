#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battery_hyperparam_optimizer.py
================================

Purpose
-------
Hyperparameter optimizer for the battery agent's reward function using the exact
conditional logic of `DefaultBatteryReward.compute()`.
It recomputes rewards *in memory* from a discrete state–action dataset and
evaluates global KPIs + a combined fitness score. Multiple optimization
methods are supported.

Inputs
------
- /test/battery/output/reward_battery.csv

Outputs
-------
- /test/battery/output/hyperparam_optimization_log.csv
- /test/battery/output/*.svg (visualizations)
"""

# ==================================================
# Dependency check
# ==================================================
def check_dependencies():
    missing = []
    for pkg, cmd in [
        ("numpy", "pip install numpy"),
        ("pandas", "pip install pandas"),
        ("matplotlib", "pip install matplotlib"),
        ("tqdm", "pip install tqdm"),
        ("yaml", "pip install PyYAML"),
    ]:
        try:
            __import__(pkg)
        except Exception:
            missing.append((pkg, cmd))
    try:
        import skopt  # noqa: F401
    except Exception:
        pass
    if missing:
        print("[ERROR] Missing required libraries:")
        for name, hint in missing:
            print(f"  - {name}  →  {hint}")
        raise SystemExit(1)

check_dependencies()

# ==================================================
# Imports
# ==================================================
import os, time, math, random, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False


# ==================================================
# Configuration
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATASET_FILE = os.path.join(PROJECT_ROOT, "test/battery/output/reward_battery.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "test/battery/output/")
DEFAULT_YAML = os.path.join(PROJECT_ROOT, "configs/default.yaml")

OPTIMIZATION_METHOD = "random_search"  # "random_search" | "bayesian" | "evolutionary"
MAX_ITERATIONS = 100
POP_SIZE = 24
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

WEIGHTS = {
    "mean_balance": 1.0,
    "IAE_mean": 1.0,
    "ISE_mean": 1.0,
    "variability": 1.0,
    "reward_mean": 1.0,
}
SCALE_FACTOR_REWARD = 50.0
SOC_MAX = 4

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "psi":   (0.5, 4.0),
    "sigma": (0.5, 4.0),
    "nu":    (0.5, 4.0),
    "beta":  (0.5, 3.0),
    "xi":    (0.1, 2.0),
}


# ==================================================
# Utilities
# ==================================================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_seed_from_yaml(path: str, default_seed: int = 42) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        seed = int(conf.get("simulation", {}).get("seed", default_seed))
        print(f"[INFO] Using simulation seed: {seed}")
        return seed
    except Exception:
        print(f"[WARN] Could not read seed from {path}, using default={default_seed}")
        return default_seed

def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        n = len(weights)
        return {k: 1.0/n for k in weights.keys()}
    return {k: float(max(0.0, v))/s for k, v in weights.items()}

def load_grid_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    required = ["total_power_idx", "demand_power_idx", "battery_soc_idx", "action", "reward"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["dP"] = df["total_power_idx"] - df["demand_power_idx"]
    return df


# ==================================================
# Reward Function
# ==================================================
@dataclass
class DefaultBatteryReward:
    psi: float
    sigma: float
    nu: float
    beta: float
    xi: float
    soc_max: int = SOC_MAX

    def compute_row(self, total_idx: int, demand_idx: int, soc_idx: int, action: int) -> float:
        dP = total_idx - demand_idx
        soc = soc_idx
        if action == 2:
            if dP < 0 and soc > 0:
                return self.psi * (abs(dP) * soc)
            else:
                return -self.sigma
        elif action == 1:
            if dP > 0:
                return self.nu * (dP * max(self.soc_max - soc, 0))
            else:
                return -self.beta * abs(dP)
        elif action == 0:
            return -self.xi * abs(dP)
        else:
            return -10.0


# ==================================================
# KPIs and Fitness
# ==================================================
def compute_kpis_from_params(df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, float]:
    rew = DefaultBatteryReward(**params)
    rewards = [rew.compute_row(r.total_power_idx, r.demand_power_idx, r.battery_soc_idx, r.action)
               for _, r in df.iterrows()]
    rewards = np.asarray(rewards, dtype=float)
    dP = df["dP"].to_numpy(dtype=float)
    return {
        "mean_balance": float(np.mean(dP)),
        "IAE_mean": float(np.mean(np.abs(dP))),
        "ISE_mean": float(np.mean(np.square(dP))),
        "variability": float(np.std(dP, ddof=0)),
        "reward_mean": float(np.mean(rewards)),
    }

def compute_fitness(kpis: Dict[str, float]) -> float:
    w = normalize_weights(WEIGHTS)
    sb = 1/(1+abs(kpis["mean_balance"]))
    sIAE = 1/(1+kpis["IAE_mean"])
    sISE = 1/(1+kpis["ISE_mean"])
    svar = 1/(1+kpis["variability"])
    srew = 0.5*(math.tanh(kpis["reward_mean"]/SCALE_FACTOR_REWARD)+1.0)
    return w["mean_balance"]*sb + w["IAE_mean"]*sIAE + w["ISE_mean"]*sISE + w["variability"]*svar + w["reward_mean"]*srew


# ==================================================
# Logging & Visualization
# ==================================================
def append_log_row(log_path: str, params: Dict[str, float], kpis: Dict[str, float], fitness: float):
    import csv
    exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp","psi","sigma","nu","beta","xi",
                             "mean_balance","IAE_mean","ISE_mean","variability","reward_mean","fitness_score"])
        writer.writerow([pd.Timestamp.now(tz="UTC").isoformat(),
                         params["psi"],params["sigma"],params["nu"],params["beta"],params["xi"],
                         kpis["mean_balance"],kpis["IAE_mean"],kpis["ISE_mean"],kpis["variability"],kpis["reward_mean"],fitness])

def make_figures_svg(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(8,4.5))
    plt.plot(df["fitness_score"].values, marker="o")
    plt.title("Fitness Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"fitness_convergence.svg"), format="svg")
    plt.close()


# ==================================================
# Optimization Methods
# ==================================================
def sample_params_uniform() -> Dict[str, float]:
    return {k: np.random.uniform(low, high) for k,(low,high) in PARAM_BOUNDS.items()}

def random_search(df, budget, log_path):
    rows = []
    pbar = tqdm(total=budget, desc="[random_search]", ncols=100)
    for _ in range(budget):
        params = sample_params_uniform()
        kpis = compute_kpis_from_params(df, params)
        fitness = compute_fitness(kpis)
        append_log_row(log_path, params, kpis, fitness)
        rows.append({**params, **kpis, "fitness_score": fitness})
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ==================================================
# Main
# ==================================================
def main():
    ensure_dirs()
    seed = load_seed_from_yaml(DEFAULT_YAML)
    set_global_seed(seed)
    df = load_grid_dataset(DATASET_FILE)
    log_csv = os.path.join(OUTPUT_DIR, "hyperparam_optimization_log.csv")
    if os.path.exists(log_csv):
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        os.rename(log_csv, os.path.join(OUTPUT_DIR, f"hyperparam_optimization_log_{ts}.csv"))

    start = time.time()
    df_log = random_search(df, MAX_ITERATIONS, log_csv)
    make_figures_svg(df_log, OUTPUT_DIR)

    best = df_log.sort_values(by="fitness_score", ascending=False).head(5)
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = elapsed % 60

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Execution time: {mins} min {secs:.1f} s")
    for i, r in best.iterrows():
        print(f"{i+1}) psi={r.psi:.3f}, sigma={r.sigma:.3f}, nu={r.nu:.3f}, "
              f"beta={r.beta:.3f}, xi={r.xi:.3f} | fitness={r.fitness_score:.6f}")
    print(f"Results saved to: {log_csv}")
    print(f"Figures saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
