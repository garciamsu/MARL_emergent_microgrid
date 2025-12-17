#!/usr/bin/env python3
"""
D_compute_metrics.py

Calcula métricas operativas y de aprendizaje cuando el entrenamiento ha
finalizado, usando el último episodio disponible en results/evolution/.

Métricas calculadas (en columnas separadas):
- MEAN/VAR/IAE/ISE para env_energy_balance y env_energy_balance_idx (VAR muestral)
- Penetración por energía (renovables y red) filtrando por acciones
- Penetración por tiempo (renovables y red) basada en acciones

Imprime una tabla resumida y guarda un CSV con los resultados.
"""

import os
import sys
from pathlib import Path
import re
import glob
import numpy as np
import pandas as pd

# Añadir raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_config
from core.csv_handler import read_result_csv, write_result_csv


def _find_latest_episode_file(pattern: str = "results/evolution/episode_*.csv") -> tuple[str, int]:
    """Devuelve la ruta y el número del último episodio (número más alto).
    Si hay empates, se usa el de timestamp más reciente.
    """
    files = glob.glob(pattern)
    if not files:
        return "", -1

    def key_fn(path_str):
        match = re.search(r"episode_(\d+)\.csv", path_str)
        num = int(match.group(1)) if match else -1
        return (num, os.path.getmtime(path_str))

    latest = max(files, key=key_fn)
    match = re.search(r"episode_(\d+)\.csv", latest)
    ep_num = int(match.group(1)) if match else -1
    return latest, ep_num


def _series_metrics(series: pd.Series) -> dict:
    """Calcula MEAN, VAR (ddof=1), IAE, ISE para una serie numérica."""
    arr = series.to_numpy(dtype=float)
    n_samples = arr.size
    mean = float(np.mean(arr)) if n_samples else 0.0
    var = float(np.var(arr, ddof=1)) if n_samples > 1 else 0.0
    iae = float(np.sum(np.abs(arr))) if n_samples else 0.0
    ise = float(np.sum(arr ** 2)) if n_samples else 0.0
    return {"MEAN": mean, "VAR": var, "IAE": iae, "ISE": ise}


def _penetrations_with_actions(frame: pd.DataFrame) -> dict:
    """Calcula penetraciones por energía y tiempo usando filtros por acción.

    - Renovables (energía): sum_t min(solar_t + wind_t, demand_t) solo si
      (action_solar != 0 o action_wind != 0) dividido por sum_t demand_t.
    - Red (energía): sum_t max(grid_t, 0) solo si (action_grid != 0)
      dividido por sum_t demand_t.
    - Renovables (tiempo): fracción de pasos con (action_solar != 0 o action_wind != 0).
    - Red (tiempo): fracción de pasos con (action_grid != 0).
    
    If individual agent columns are not present, uses aggregated columns.
    """
    demand = frame.get("env_demand_power", pd.Series(dtype=float)).astype(float)
    
    # Check if individual agent columns exist
    has_individual_agents = "power_solar#0" in frame.columns or "power_wind#0" in frame.columns
    
    if has_individual_agents:
        # Use individual agent data
        solar = frame.get("power_solar#0", pd.Series(dtype=float)).astype(float)
        wind = frame.get("power_wind#0", pd.Series(dtype=float)).astype(float)
        grid = frame.get("power_grid#0", pd.Series(dtype=float)).astype(float)

        a_solar = frame.get("action_solar#0", pd.Series(dtype=float)).fillna(0)
        a_wind = frame.get("action_wind#0", pd.Series(dtype=float)).fillna(0)
        a_grid = frame.get("action_grid#0", pd.Series(dtype=float)).fillna(0)

        use_re = (a_solar != 0) | (a_wind != 0)
        use_grid = (a_grid != 0)

        re_power = (solar + wind).clip(lower=0)
    else:
        # Use aggregated data when individual agents are not present
        re_power = frame.get("env_total_renewable", pd.Series(dtype=float)).astype(float)
        grid = frame.get("power_grid#0", pd.Series(dtype=float)).astype(float)
        
        a_grid = frame.get("action_grid#0", pd.Series(dtype=float)).fillna(0)
        
        # Assume renewables are used when total_renewable > 0
        use_re = (re_power > 0)
        use_grid = (a_grid != 0)
    
    effective_re = np.minimum(re_power, demand)

    demand_sum = float(demand.sum())
    n_steps = len(frame)

    if demand_sum <= 0:
        ren_energy_pen = 0.0
        grid_energy_pen = 0.0
    else:
        # Use .values to avoid index alignment issues
        ren_energy_pen = float(effective_re.values[use_re.values].sum() / demand_sum)
        grid_energy_pen = float(grid.clip(lower=0).values[use_grid.values].sum() / demand_sum)

    ren_time_pen = float(use_re.mean()) if n_steps > 0 else 0.0
    grid_time_pen = float(use_grid.mean()) if n_steps > 0 else 0.0

    return {
        "Renewable_Penetration_energy": ren_energy_pen,
        "Grid_Penetration_energy": grid_energy_pen,
        "Renewable_Penetration_time": ren_time_pen,
        "Grid_Penetration_time": grid_time_pen,
    }


def _format_table(metrics_row: dict) -> pd.DataFrame:
    """Convierte el diccionario de métricas a una tabla ordenada para impresión."""
    ordered = [
        ("episode", metrics_row.get("episode")),
        ("n_steps", metrics_row.get("n_steps")),
        ("balance_MEAN", metrics_row.get("balance_MEAN")),
        ("balance_VAR", metrics_row.get("balance_VAR")),
        ("balance_IAE", metrics_row.get("balance_IAE")),
        ("balance_ISE", metrics_row.get("balance_ISE")),
        ("balance_idx_MEAN", metrics_row.get("balance_idx_MEAN")),
        ("balance_idx_VAR", metrics_row.get("balance_idx_VAR")),
        ("balance_idx_IAE", metrics_row.get("balance_idx_IAE")),
        ("balance_idx_ISE", metrics_row.get("balance_idx_ISE")),
        ("Renewable_Penetration_energy", metrics_row.get("Renewable_Penetration_energy")),
        ("Grid_Penetration_energy", metrics_row.get("Grid_Penetration_energy")),
        ("Renewable_Penetration_time", metrics_row.get("Renewable_Penetration_time")),
        ("Grid_Penetration_time", metrics_row.get("Grid_Penetration_time")),
    ]
    table_df = pd.DataFrame(ordered, columns=["Metric", "Value"])
    return table_df


def main():
    """Punto de entrada principal."""
    print("=" * 80)
    print("[D] D_compute_metrics.py - Metricas del ultimo episodio")
    print("=" * 80)

    # Cargar configuración (por si se requiere en el futuro)
    try:
        _ = load_config()
    except Exception as exc:
        print(f"\n[WARN] Advertencia al cargar configuracion (no bloqueante): {exc}")

    # Localizar último episodio
    latest_path, ep_num = _find_latest_episode_file()
    if not latest_path or ep_num < 0:
        print("\n[ERROR] No se encontraron episodios en results/evolution/.")
        print("   Ejecuta primero analysis_tools/B_run_training.py o main.py")
        sys.exit(1)

    print(f"\n[INFO] Usando episodio mas reciente: episode_{ep_num}.csv")
    print(f"       Ruta: {latest_path}")

    # Cargar episodio
    try:
        episode_df = read_result_csv(latest_path)
    except Exception as exc:
        print(f"\n[ERROR] Error leyendo {latest_path}: {exc}")
        sys.exit(1)

    # Métricas de balance (continuo e índice)
    n_steps = len(episode_df)
    bal = episode_df.get("env_energy_balance", pd.Series(dtype=float))
    bal_idx = episode_df.get("env_energy_balance_idx", pd.Series(dtype=float))

    m_bal = _series_metrics(bal)
    m_idx = _series_metrics(bal_idx)

    # Penetraciones con acciones
    pens = _penetrations_with_actions(episode_df)

    metrics_row = {
        "episode": ep_num,
        "n_steps": n_steps,
        "balance_MEAN": m_bal["MEAN"],
        "balance_VAR": m_bal["VAR"],
        "balance_IAE": m_bal["IAE"],
        "balance_ISE": m_bal["ISE"],
        "balance_idx_MEAN": m_idx["MEAN"],
        "balance_idx_VAR": m_idx["VAR"],
        "balance_idx_IAE": m_idx["IAE"],
        "balance_idx_ISE": m_idx["ISE"],
        **pens,
    }

    table = _format_table(metrics_row)

    # Imprimir tabla
    print("\n" + "-" * 80)
    print("Tabla de métricas (último episodio)")
    print("-" * 80)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 40):
        print(table.to_string(index=False, header=["Métrica", "Valor"]))

    # Guardar CSV con resultados
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_episode_{ep_num}.csv"
    write_result_csv(pd.DataFrame([metrics_row]), out_path)

    print("\n" + "=" * 80)
    print("[OK] Calculo de metricas completado y guardado.")
    print(f"   Archivo: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
