#!/usr/bin/env python3
"""
D_compute_metrics.py

Calcula métricas operativas y de aprendizaje a partir de los episodios.
Imprime resumen en consola con PASS/FAIL según umbrales de configs/default.yaml.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Añadir raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_tools.metrics import (
    compute_energy_balance_metrics,
    compute_penetration_metrics,
    compute_cumulative_rewards
)
from analysis_tools.utils import load_episode_csvs
from configs.loader import load_config


def compute_metrics_all_episodes(pattern: str = "results/evolution/episode_*.csv"):
    """
    Calcula métricas para todos los episodios.
    
    Args:
        pattern: Patrón glob para buscar episodios.
    
    Returns:
        DataFrame con métricas por episodio.
    """
    import glob
    import re
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"⚠️  No se encontraron episodios en {pattern}")
        return pd.DataFrame()
    
    all_metrics = []
    
    for file_path in files:
        ep_match = re.search(r"episode_(\d+)\.csv", file_path)
        if not ep_match:
            continue
        
        ep_num = int(ep_match.group(1))
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️  Error cargando {file_path}: {e}")
            continue
        
        # Métricas del episodio
        metrics = {"episode": ep_num}
        
        # Balance energético
        metrics.update(compute_energy_balance_metrics(df))
        
        # Penetraciones
        metrics.update(compute_penetration_metrics(df))
        
        # Recompensas acumuladas
        metrics.update(compute_cumulative_rewards(df))
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)


def evaluate_thresholds(df_metrics: pd.DataFrame, thresholds: dict) -> dict:
    """
    Evalúa métricas contra umbrales y retorna PASS/FAIL.
    
    Args:
        df_metrics: DataFrame con métricas por episodio.
        thresholds: Diccionario de umbrales desde config.
    
    Returns:
        Diccionario con resultados de evaluación.
    """
    # Promedios globales
    mean_metrics = df_metrics.mean(numeric_only=True)
    
    results = {}
    
    # IAE
    iae_threshold = thresholds.get("IAE", float("inf"))
    results["IAE"] = {
        "value": mean_metrics.get("IAE", 0),
        "threshold": iae_threshold,
        "pass": mean_metrics.get("IAE", 0) <= iae_threshold
    }
    
    # ISE
    ise_threshold = thresholds.get("ISE", float("inf"))
    results["ISE"] = {
        "value": mean_metrics.get("ISE", 0),
        "threshold": ise_threshold,
        "pass": mean_metrics.get("ISE", 0) <= ise_threshold
    }
    
    # Variability
    var_threshold = thresholds.get("Variability", float("inf"))
    results["Variability"] = {
        "value": mean_metrics.get("Variability", 0),
        "threshold": var_threshold,
        "pass": mean_metrics.get("Variability", 0) <= var_threshold
    }
    
    # Renewable Penetration (mínimo)
    re_min = thresholds.get("Renewable_Penetration_min", 0)
    results["Renewable_Penetration"] = {
        "value": mean_metrics.get("Renewable_Penetration", 0),
        "threshold": f">= {re_min}",
        "pass": mean_metrics.get("Renewable_Penetration", 0) >= re_min
    }
    
    # Grid Penetration (máximo)
    grid_max = thresholds.get("Grid_Penetration_max", float("inf"))
    results["Grid_Penetration"] = {
        "value": mean_metrics.get("Grid_Penetration", 0),
        "threshold": f"<= {grid_max}",
        "pass": mean_metrics.get("Grid_Penetration", 0) <= grid_max
    }
    
    # Cumulative Reward Total (mínimo)
    reward_min = thresholds.get("Cumulative_Reward_min", -float("inf"))
    results["Cumulative_Reward_Total"] = {
        "value": mean_metrics.get("total_reward_all", 0),
        "threshold": f">= {reward_min}",
        "pass": mean_metrics.get("total_reward_all", 0) >= reward_min
    }
    
    return results


def main():
    """Punto de entrada principal."""
    print("="*80)
    print("📊 D_compute_metrics.py - Calcular Métricas")
    print("="*80)
    
    # Cargar configuración
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ ERROR cargando configuración: {e}")
        sys.exit(1)
    
    # Umbrales (placeholder: agregar a default.yaml si se desea)
    thresholds = config.get("analysis", {}).get("thresholds", {
        "IAE": 10000,
        "ISE": 50000,
        "Variability": 500,
        "Renewable_Penetration_min": 0.3,
        "Grid_Penetration_max": 0.7,
        "Cumulative_Reward_min": -5000,
    })
    
    print(f"\n📋 Umbrales configurados:")
    for key, val in thresholds.items():
        print(f"   {key}: {val}")
    
    # Calcular métricas
    print(f"\n🔄 Calculando métricas de todos los episodios...")
    
    try:
        df_metrics = compute_metrics_all_episodes()
    except Exception as e:
        print(f"\n❌ ERROR calculando métricas: {e}")
        sys.exit(1)
    
    if df_metrics.empty:
        print(f"\n⚠️  No se encontraron episodios para analizar.")
        print("   Ejecuta primero B_run_training.py.")
        sys.exit(1)
    
    print(f"✅ Métricas calculadas para {len(df_metrics)} episodios.")
    
    # Evaluar contra umbrales
    results = evaluate_thresholds(df_metrics, thresholds)
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("📈 RESUMEN DE MÉTRICAS (Promedios)")
    print("="*80)
    
    for metric_name, data in results.items():
        status = "✅ PASS" if data["pass"] else "❌ FAIL"
        print(f"\n{metric_name}:")
        print(f"   Valor: {data['value']:.4f}")
        print(f"   Umbral: {data['threshold']}")
        print(f"   Estado: {status}")
    
    # Recompensas por agente
    print("\n" + "="*80)
    print("🏆 RECOMPENSAS POR AGENTE (Promedio)")
    print("="*80)
    
    reward_cols = [c for c in df_metrics.columns if c.startswith("total_reward_") and c != "total_reward_all"]
    for col in sorted(reward_cols):
        agent_name = col.replace("total_reward_", "")
        mean_reward = df_metrics[col].mean()
        print(f"   {agent_name.capitalize()}: {mean_reward:.2f}")
    
    print("\n" + "="*80)
    print("✅ Cálculo de métricas completado.")
    print("   Siguiente paso: E_plot_metrics.py")
    print("="*80)


if __name__ == "__main__":
    main()
