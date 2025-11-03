#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B_wind_hyperparam_optimizer.py
===============================

Optimizador de hiperparámetros para la función de recompensa del agente Wind.
Utiliza el enfoque basado en márgenes: maximiza la diferencia entre la recompensa
de la acción correcta y la mejor acción incorrecta para cada estado.

Inputs
------
- /test/wind/input/Wind_Agent_Reward_Table.csv

Outputs
-------
- /test/wind/output/hyperparam_optimization_log.csv
- /test/wind/output/fitness_convergence.svg

Métodos de optimización soportados
-----------------------------------
- random_search: Búsqueda aleatoria
- bayesian: Optimización bayesiana (requiere scikit-optimize)
- evolutionary: Algoritmo evolutivo simple
"""

import os
import sys
import time
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List

# Agregar directorio raíz al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Importar desde core y test utils
from core.rewards import DefaultWindReward
from test.optimizer_utils import (
    get_correct_action_wind,
    compute_total_margin,
)

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


# ==================================================
# CONFIGURACIÓN
# ==================================================

# Rutas
DATASET_FILE = os.path.join(PROJECT_ROOT, "test/wind/input/Wind_Agent_Reward_Table.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test/wind/output/")
DEFAULT_YAML = os.path.join(PROJECT_ROOT, "configs/default.yaml")

# Método de optimización: "random_search" | "bayesian" | "evolutionary"
OPTIMIZATION_METHOD = "random_search"

# Configuración de la optimización
MAX_ITERATIONS = 500
POP_SIZE = 24
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7

# Rangos de búsqueda para hiperparámetros (ajustables)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "theta": (0.1, 5.0),
    "beta":  (0.1, 5.0),
    "eta":   (0.1, 5.0),
    "nu":    (0.1, 5.0),
    "xi":    (0.1, 5.0),
}

# Configuración específica del agente
ALL_ACTIONS = [0, 1]  # 0: no suministrar, 1: suministrar


# ==================================================
# UTILIDADES
# ==================================================

def ensure_dirs():
    """Crea el directorio de salida si no existe."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_seed_from_yaml(path: str, default_seed: int = 42) -> int:
    """Carga la semilla desde el archivo YAML de configuración."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f) or {}
        seed = int(conf.get("simulation", {}).get("seed", default_seed))
        print(f"[INFO] Usando semilla: {seed}")
        return seed
    except Exception:
        print(f"[WARN] No se pudo leer semilla de {path}, usando default={default_seed}")
        return default_seed


def set_global_seed(seed: int):
    """Establece la semilla global para reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Carga el dataset de entrada (formato CSV con delimitador ';')."""
    df = pd.read_csv(csv_path, delimiter=';', encoding="utf-8")
    
    # Validar columnas requeridas
    required = ["wind_idx", "demand_idx", "renewable_idx", "action"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")
    
    print(f"[INFO] Dataset cargado: {len(df)} filas, {len(df.groupby(['wind_idx', 'demand_idx', 'renewable_idx']))} estados únicos")
    
    return df


def state_transform_wind(state_dict: Dict, row: pd.Series) -> Dict:
    """
    Transforma el estado para incluir variables derivadas necesarias
    para determinar la acción correcta.
    """
    state_dict['wind_potential_idx'] = row['wind_idx']
    return state_dict


def compute_fitness_wind(df: pd.DataFrame, params: Dict[str, float]) -> float:
    """
    Función objetivo: calcula el margen total para un conjunto de hiperparámetros.
    
    Margen = suma(recompensa_correcta - max_recompensa_incorrecta) para todos los estados.
    """
    # Preparar atributos del entorno que la función de recompensa necesita
    env_attrs = {
        'renewable_power_idx': 0,  # Se actualizará por estado
        'demand_power_idx': 0,     # Se actualizará por estado
    }
    
    agent_attrs = {}
    
    # Columnas de estado
    state_columns = ['wind_idx', 'demand_idx', 'renewable_idx']
    
    # Calcular margen total
    total_margin = compute_total_margin(
        df=df,
        params=params,
        reward_class=DefaultWindReward,
        state_columns=state_columns,
        action_column='action',
        get_correct_action_fn=get_correct_action_wind,
        all_actions=ALL_ACTIONS,
        env_extra_attrs=env_attrs,
        agent_extra_attrs=agent_attrs,
        state_transform_fn=state_transform_wind,
    )
    
    return total_margin


def append_log_row(log_path: str, params: Dict[str, float], fitness: float):
    """Registra una fila en el log CSV."""
    import csv
    exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            # Escribir encabezado
            header = ["timestamp"] + list(PARAM_BOUNDS.keys()) + ["margin_total"]
            writer.writerow(header)
        # Escribir datos
        row = [pd.Timestamp.now(tz="UTC").isoformat()]
        row.extend([params[k] for k in PARAM_BOUNDS.keys()])
        row.append(fitness)
        writer.writerow(row)


def make_figures_svg(df: pd.DataFrame, out_dir: str):
    """Genera gráfico de convergencia en formato SVG."""
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["margin_total"].values, marker="o", alpha=0.7)
    plt.title("Convergencia del Margen Total")
    plt.xlabel("Iteración")
    plt.ylabel("Margen Total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fitness_convergence.svg"), format="svg")
    plt.close()
    print(f"[INFO] Gráfico guardado en {out_dir}/fitness_convergence.svg")


# ==================================================
# MÉTODOS DE OPTIMIZACIÓN
# ==================================================

def sample_params_uniform() -> Dict[str, float]:
    """Muestrea hiperparámetros uniformemente en el rango definido."""
    return {k: np.random.uniform(low, high) for k, (low, high) in PARAM_BOUNDS.items()}


def random_search(df: pd.DataFrame, budget: int, log_path: str) -> pd.DataFrame:
    """Búsqueda aleatoria."""
    rows = []
    pbar = tqdm(total=budget, desc="[random_search]", ncols=100)
    for _ in range(budget):
        params = sample_params_uniform()
        fitness = compute_fitness_wind(df, params)
        append_log_row(log_path, params, fitness)
        rows.append({**params, "margin_total": fitness})
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


def bayesian_search(df: pd.DataFrame, budget: int, log_path: str) -> pd.DataFrame:
    """Optimización bayesiana usando scikit-optimize (gp_minimize)."""
    if not SKOPT_AVAILABLE:
        print("[WARN] scikit-optimize no disponible; usando random_search")
        return random_search(df, budget, log_path)

    rows: List[Dict] = []
    space = [Real(low, high, name=name) for name, (low, high) in PARAM_BOUNDS.items()]

    def objective(x):
        params = dict(zip(PARAM_BOUNDS.keys(), x))
        fitness = compute_fitness_wind(df, params)
        append_log_row(log_path, params, fitness)
        rows.append({**params, "margin_total": fitness})
        # gp_minimize minimiza, queremos maximizar el margen
        return -fitness

    print(f"[INFO] Ejecutando optimización bayesiana (n_calls={budget})")
    gp_minimize(objective, space, n_calls=budget, random_state=42, verbose=False)

    return pd.DataFrame(rows)


def evolutionary_search(df: pd.DataFrame, budget: int, log_path: str) -> pd.DataFrame:
    """Algoritmo evolutivo simple (estilo genético)."""
    rows: List[Dict] = []

    def evaluate_params(params: Dict[str, float]) -> float:
        fitness = compute_fitness_wind(df, params)
        append_log_row(log_path, params, fitness)
        rows.append({**params, "margin_total": fitness})
        return fitness

    # Inicializar población
    population = [sample_params_uniform() for _ in range(POP_SIZE)]
    fitnesses = [evaluate_params(p) for p in population]
    evals = len(population)

    gen = 0
    while evals < budget:
        gen += 1
        # Selección: mantener los mejores 50%
        paired = list(zip(population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)
        survivors = [p for p, _ in paired[: max(2, len(paired)//2)]]

        # Generar hijos por cruce y mutación
        children: List[Dict[str, float]] = []
        while len(children) + len(survivors) < POP_SIZE and evals < budget:
            a, b = random.sample(survivors, 2)
            child = {}
            for k in PARAM_BOUNDS.keys():
                # Cruce uniforme
                child[k] = a[k] if random.random() < 0.5 else b[k]
                # Mutación
                if random.random() < MUTATION_RATE:
                    low, high = PARAM_BOUNDS[k]
                    child[k] = np.clip(child[k] + np.random.normal(scale=(high-low)*0.1), low, high)
            children.append(child)

        # Nueva población
        population = survivors + children
        while len(population) < POP_SIZE:
            population.append(sample_params_uniform())

        # Evaluar nueva población
        fitnesses = [evaluate_params(p) for p in population]
        evals += len(population)
        print(f"[INFO] Generación {gen}, evaluaciones={evals}, mejor_margen={max(fitnesses):.2f}")

    return pd.DataFrame(rows)


# ==================================================
# MAIN
# ==================================================

def main():
    print("=" * 60)
    print(" OPTIMIZADOR DE HIPERPARÁMETROS - WIND AGENT")
    print("=" * 60)
    
    ensure_dirs()
    seed = load_seed_from_yaml(DEFAULT_YAML)
    set_global_seed(seed)
    
    # Cargar dataset
    df = load_dataset(DATASET_FILE)
    
    # Preparar log
    log_csv = os.path.join(OUTPUT_DIR, "hyperparam_optimization_log.csv")
    if os.path.exists(log_csv):
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        backup = os.path.join(OUTPUT_DIR, f"hyperparam_optimization_log_{ts}.csv")
        os.rename(log_csv, backup)
        print(f"[INFO] Log anterior respaldado en {backup}")

    # Ejecutar optimización
    start = time.time()
    method = str(OPTIMIZATION_METHOD).lower()
    
    if method in ("random_search", "random"):
        df_log = random_search(df, MAX_ITERATIONS, log_csv)
    elif method in ("bayesian", "bayes", "gp"):
        df_log = bayesian_search(df, MAX_ITERATIONS, log_csv)
    elif method in ("evolutionary", "evolution", "ga"):
        df_log = evolutionary_search(df, MAX_ITERATIONS, log_csv)
    else:
        print(f"[WARN] Método desconocido '{OPTIMIZATION_METHOD}', usando random_search")
        df_log = random_search(df, MAX_ITERATIONS, log_csv)
    
    # Generar visualizaciones
    make_figures_svg(df_log, OUTPUT_DIR)
    
    # Mostrar mejores resultados
    best = df_log.sort_values(by="margin_total", ascending=False).head(5)
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = elapsed % 60

    print("\n" + "=" * 60)
    print(" OPTIMIZACIÓN COMPLETADA")
    print("=" * 60)
    print(f"Tiempo de ejecución: {mins} min {secs:.1f} s")
    print(f"\nTOP 5 MEJORES HIPERPARÁMETROS:")
    print("-" * 60)
    for idx, (i, row) in enumerate(best.iterrows(), 1):
        print(f"{idx}) theta={row.theta:.3f}, beta={row.beta:.3f}, eta={row.eta:.3f}, "
              f"nu={row.nu:.3f}, xi={row.xi:.3f} | margen={row.margin_total:.2f}")
    print("-" * 60)
    print(f"Resultados guardados en: {log_csv}")
    print(f"Gráficos guardados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
