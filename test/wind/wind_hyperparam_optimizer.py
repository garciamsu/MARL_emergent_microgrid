#!/usr/bin/env python3
"""
wind_hyperparam_optimizer.py
================================

Hyperparameter optimizer for WindAgent reward function. Based on the
battery/grid implementation; paths and names adjusted for wind agent.
"""

import os, time, math, random, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATASET_FILE = os.path.join(PROJECT_ROOT, "test/wind/output/reward_wind.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test/wind/output/")
DEFAULT_YAML = os.path.join(PROJECT_ROOT, "configs/default.yaml")

OPTIMIZATION_METHOD = "random_search"
MAX_ITERATIONS = 500
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

import inspect
from core import rewards as core_rewards
from .grid_hyperparam_optimizer import (
    ensure_dirs, load_seed_from_yaml, set_global_seed, normalize_weights,
    append_log_row, make_figures_svg, sample_params_uniform, random_search,
    bayesian_search, evolutionary_search,
)


def load_wind_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=';', encoding="utf-8")
    required = ["wind_idx", "demand_idx", "renewable_potential_idx", "action"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df["dP"] = df["wind_idx"] - df["demand_idx"]
    return df


def _build_reward_instance(params: Dict[str, float]):
    cls = core_rewards.DefaultWindReward
    sig = inspect.signature(cls.__init__)
    filtered = {k: float(v) for k, v in params.items() if k in sig.parameters}
    return cls(**filtered)


def compute_kpis_from_params(df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, float]:
    rew = _build_reward_instance(params)
    rewards = []

    class _Agent:
        pass

    class _Env:
        def __init__(self):
            self.price = 0.0

    env = _Env()
    for _, row in df.iterrows():
        agent = _Agent()
        agent.action = int(row.action)
        state_tuple = (int(row.wind_idx), int(row.demand_idx), int(row.renewable_potential_idx))
        try:
            val = float(rew.compute(agent, env, state_tuple))
        except Exception:
            val = -1e6
        rewards.append(val)

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
    return compute_fitness_grid(kpis)


def main():
    ensure_dirs()
    seed = load_seed_from_yaml(DEFAULT_YAML)
    set_global_seed(seed)
    df = load_wind_dataset(DATASET_FILE)
    log_csv = os.path.join(OUTPUT_DIR, "hyperparam_optimization_log.csv")
    if os.path.exists(log_csv):
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        os.rename(log_csv, os.path.join(OUTPUT_DIR, f"hyperparam_optimization_log_{ts}.csv"))

    start = time.time()
    method = str(OPTIMIZATION_METHOD).lower()
    if method in ("random_search", "random"):
        df_log = random_search(df, MAX_ITERATIONS, log_csv)
    elif method in ("bayesian", "bayes", "gp", "gp_minimize"):
        df_log = bayesian_search(df, MAX_ITERATIONS, log_csv)
    elif method in ("evolutionary", "evolution", "ga"):
        df_log = evolutionary_search(df, MAX_ITERATIONS, log_csv)
    else:
        print(f"[WARN] Unknown OPTIMIZATION_METHOD='{OPTIMIZATION_METHOD}', falling back to random_search")
        df_log = random_search(df, MAX_ITERATIONS, log_csv)

    make_figures_svg(df_log, OUTPUT_DIR)

    best = df_log.sort_values(by="fitness_score", ascending=False).head(5)
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = elapsed % 60

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Execution time: {mins} min {secs:.1f} s")
    for idx, row in best.iterrows():
        print(f"{idx+1}) psi={row.psi:.3f}, sigma={row.sigma:.3f}, nu={row.nu:.3f}, "
              f"beta={row.beta:.3f}, xi={row.xi:.3f} | fitness={row.fitness_score:.6f}")
    print(f"Results saved to: {log_csv}")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
