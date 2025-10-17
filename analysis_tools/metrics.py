import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def compute_q_diff_norm(q_new, q_old):
    """
    Calcula la norma L1 de la diferencia entre dos Q-tables.
    
    Args:
        q_new: Q-table nueva (dict de dicts).
        q_old: Q-table anterior (dict de dicts).
    
    Returns:
        Norma L1 total.
    """
    total = 0.0
    for state in q_new:
        for a, v in q_new[state].items():
            total += abs(v - q_old.get(state, {}).get(a, 0.0))
    return total


def check_stability(df_metrics, iae_threshold):
    """
    Verifica estabilidad de métricas en los últimos 200 episodios.
    
    Args:
        df_metrics: DataFrame con métricas por episodio (debe tener columnas IAE y Var_dif).
        iae_threshold: Umbral aceptable para IAE.
    
    Returns:
        Diccionario con resultados de estabilidad.
    """
    recent = df_metrics.tail(200)
    return {
        "IAE_mean": recent["IAE"].mean(),
        "Var_mean": recent["Var_dif"].mean(),
        "IAE_stable": recent["IAE"].mean() <= iae_threshold,
        "Var_stable": recent["Var_dif"].mean() <= recent["Var_dif"].median() * 1.1,
    }


def compute_energy_balance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula métricas de balance energético a partir de un DataFrame de un episodio.
    
    Usa la columna env_energy_balance directamente (supply - demand).
    
    Args:
        df: DataFrame con columnas: env_energy_balance, env_total_renewable, 
            env_demand_power, power_grid#0.
    
    Returns:
        Diccionario con métricas: MEAN, ISE, IAE, Variability.
    """
    e_t = df["env_energy_balance"].values
    
    return {
        "MEAN": float(np.mean(e_t)),
        "ISE": float(np.sum(e_t ** 2)),
        "IAE": float(np.sum(np.abs(e_t))),
        "Variability": float(np.std(e_t)),
    }


def compute_penetration_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula penetraciones de energía renovable y de red.
    
    Args:
        df: DataFrame con columnas: env_total_renewable, env_demand_power, power_grid#0.
    
    Returns:
        Diccionario con: Renewable_Penetration, Grid_Penetration.
    """
    renewable_used = df["env_total_renewable"].sum()
    demand_total = df["env_demand_power"].sum()
    
    # Grid import (solo valores positivos)
    grid_import = df["power_grid#0"].clip(lower=0).sum()
    
    if demand_total == 0:
        return {
            "Renewable_Penetration": 0.0,
            "Grid_Penetration": 0.0,
            "note": "sin demanda",
        }
    
    return {
        "Renewable_Penetration": float(renewable_used / demand_total),
        "Grid_Penetration": float(grid_import / demand_total),
    }


def compute_cumulative_rewards(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcula recompensas acumuladas por agente y total.
    
    Args:
        df: DataFrame con columnas reward_<agent>#0.
    
    Returns:
        Diccionario con total_reward_<agent> y total_reward_all.
    """
    reward_cols = [c for c in df.columns if c.startswith("reward_")]
    
    results = {}
    total_all = 0.0
    
    for col in reward_cols:
        agent_name = col.replace("reward_", "").replace("#0", "")
        cumulative = df[col].sum()
        results[f"total_reward_{agent_name}"] = float(cumulative)
        total_all += cumulative
    
    results["total_reward_all"] = float(total_all)
    
    return results


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula todas las métricas clave de un episodio.
    
    Args:
        df: DataFrame de un episodio completo.
    
    Returns:
        Diccionario con todas las métricas operativas y de aprendizaje.
    """
    metrics = {}
    
    # Métricas de balance energético
    metrics.update(compute_energy_balance_metrics(df))
    
    # Penetraciones
    metrics.update(compute_penetration_metrics(df))
    
    # Recompensas acumuladas
    metrics.update(compute_cumulative_rewards(df))
    
    return metrics


def compute_rolling_metrics(
    series: pd.Series, 
    window: int = 50, 
    center: bool = True
) -> pd.Series:
    """
    Calcula la media móvil de una serie temporal.
    
    Args:
        series: Serie de pandas.
        window: Tamaño de la ventana.
        center: Si True, centra la ventana.
    
    Returns:
        Serie con media móvil.
    """
    return series.rolling(window=window, center=center).mean()
