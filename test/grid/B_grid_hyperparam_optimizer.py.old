#!/usr/bin/env python3
"""
grid_hyperparam_optimizer.py
================================

Hyperparameter optimizer for GridAgent reward function. Based on
battery_hyperparam_optimizer.py; adapts paths and class/name to Grid.
"""

import os, time, math, random, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
import inspect
from core import rewards as core_rewards
from typing import Dict, Tuple, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATASET_FILE = os.path.join(PROJECT_ROOT, "test/grid/output/reward_grid.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test/grid/output/")
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


# Use the canonical reward implementations from core.rewards to avoid
# duplication and ensure consistency with the rest of the app.


def _build_reward_instance(params: Dict[str, float]):
    """Instantiate the core DefaultGridReward using only the
    constructor parameters it accepts (filter params to avoid unexpected kwargs).
    """
    cls = core_rewards.DefaultGridReward
    sig = inspect.signature(cls.__init__)
    filtered = {k: float(v) for k, v in params.items() if k in sig.parameters}
    return cls(**filtered)


def compute_kpis_from_params(df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, float]:
    rew = _build_reward_instance(params)
    rewards = []
    # Create minimal dummy agent/env objects expected by core reward functions
    class _Agent:
        pass

    class _Env:
        def __init__(self):
            self.price = 0.0

    env = _Env()
    for _, row in df.iterrows():
        agent = _Agent()
        agent.action = int(row.action)
        # core.DefaultGridReward.compute expects state_tuple = (soc_idx, demand_idx, total_idx)
        state_tuple = (int(row.battery_soc_idx), int(row.demand_power_idx), int(row.total_power_idx))
        try:
            val = float(rew.compute(agent, env, state_tuple))
        except Exception:
            # In case core reward raises, record a large negative value to penalize
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
    w = normalize_weights(WEIGHTS)
    sb = 1/(1+abs(kpis["mean_balance"]))
    sIAE = 1/(1+kpis["IAE_mean"])
    sISE = 1/(1+kpis["ISE_mean"])
    svar = 1/(1+kpis["variability"])
    srew = 0.5*(math.tanh(kpis["reward_mean"]/SCALE_FACTOR_REWARD)+1.0)
    return (w["mean_balance"]*sb + w["IAE_mean"]*sIAE + w["ISE_mean"]*sISE
            + w["variability"]*svar + w["reward_mean"]*srew)


def append_log_row(log_path: str, params: Dict[str, float], kpis: Dict[str, float], fitness: float):
    import csv
    exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp", "psi", "sigma", "nu", "beta", "xi",
                "mean_balance", "IAE_mean", "ISE_mean", "variability", "reward_mean", "fitness_score",
            ])
        writer.writerow([
            pd.Timestamp.now(tz="UTC").isoformat(),
            params["psi"], params["sigma"], params["nu"], params["beta"], params["xi"],
            kpis["mean_balance"], kpis["IAE_mean"], kpis["ISE_mean"], kpis["variability"],
            kpis["reward_mean"], fitness,
        ])


def make_figures_svg(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["fitness_score"].values, marker="o")
    plt.title("Fitness Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fitness_convergence.svg"), format="svg")
    plt.close()


def sample_params_uniform() -> Dict[str, float]:
    return {k: np.random.uniform(low, high) for k, (low, high) in PARAM_BOUNDS.items()}


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


def bayesian_search(df, budget, log_path):
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception:
        print("[WARN] skopt not available; falling back to random_search for bayesian method.")
        return random_search(df, budget, log_path)

    rows: List[Dict] = []
    space = [Real(low, high, name=name) for name, (low, high) in PARAM_BOUNDS.items()]

    def objective(x):
        params = dict(zip(PARAM_BOUNDS.keys(), x))
        kpis = compute_kpis_from_params(df, params)
        fitness = compute_fitness(kpis)
        append_log_row(log_path, params, kpis, fitness)
        rows.append({**params, **kpis, "fitness_score": fitness})
        return -fitness

    print(f"[INFO] Running bayesian optimization (n_calls={budget})")
    gp_minimize(objective, space, n_calls=budget, random_state=42)
    return pd.DataFrame(rows)


def evolutionary_search(df, budget, log_path):
    rows: List[Dict] = []

    def evaluate_params(params: Dict[str, float]) -> float:
        kpis = compute_kpis_from_params(df, params)
        fitness = compute_fitness(kpis)
        append_log_row(log_path, params, kpis, fitness)
        rows.append({**params, **kpis, "fitness_score": fitness})
        return fitness

    population = [sample_params_uniform() for _ in range(POP_SIZE)]
    fitnesses = [evaluate_params(p) for p in population]
    evals = len(population)

    gen = 0
    while evals < budget:
        gen += 1
        paired = list(zip(population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)
        survivors = [p for p, _ in paired[: max(2, len(paired)//2)]]

        children: List[Dict[str, float]] = []
        while len(children) + len(survivors) < POP_SIZE and evals < budget:
            a, b = random.sample(survivors, 2)
            child = {}
            for k in PARAM_BOUNDS.keys():
                child[k] = a[k] if random.random() < 0.5 else b[k]
                if random.random() < MUTATION_RATE:
                    low, high = PARAM_BOUNDS[k]
                    child[k] = np.clip(child[k] + np.random.normal(scale=(high-low)*0.1), low, high)
            children.append(child)

        population = survivors + children
        while len(population) < POP_SIZE:
            population.append(sample_params_uniform())

        fitnesses = [evaluate_params(p) for p in population]
        evals += len(population)
        print(f"[INFO] Evolution gen={gen} evals={evals} best_fitness={max(fitnesses):.6f}")

    return pd.DataFrame(rows)


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
