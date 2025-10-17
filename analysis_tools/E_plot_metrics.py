#!/usr/bin/env python3
"""
E_plot_metrics.py

Genera visualizaciones clave a partir de los episodios:
- Serie temporal de env_energy_balance
- Cumulative Reward por episodio (total y por agente)
- Barras de métricas agregadas
- Penetraciones de renovables y red
- Histograma de balance energético
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Añadir raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_tools.plotting import (
    plot_time_series,
    plot_cumulative_rewards,
    plot_metrics_bars,
    plot_penetrations,
    plot_energy_balance_histogram
)
from analysis_tools.utils import load_episode_csvs, load_all_episodes_metrics
from analysis_tools.metrics import compute_all_metrics
from configs.loader import load_config


def main():
    """Genera visualizaciones clave."""
    print("="*80)
    print("📊 E_plot_metrics.py - Generar Gráficos")
    print("="*80)
    
    # Crear directorio de plots
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Cargar configuración
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ ERROR cargando configuración: {e}")
        sys.exit(1)
    
    # Cargar episodios consolidados
    print(f"\n🔄 Cargando métricas consolidadas...")
    
    try:
        df_consolidated = load_all_episodes_metrics()
    except Exception as e:
        print(f"\n❌ ERROR cargando episodios: {e}")
        sys.exit(1)
    
    if df_consolidated.empty:
        print(f"\n⚠️  No se encontraron episodios para graficar.")
        print("   Ejecuta primero B_run_training.py y C_collect_episodes.py.")
        sys.exit(1)
    
    print(f"✅ {len(df_consolidated)} episodios cargados.")
    
    # 1. Cumulative Rewards por agente
    print(f"\n📈 Generando gráfico de recompensas acumuladas...")
    agents = ["solar", "wind", "battery", "grid", "load"]
    plot_cumulative_rewards(
        df_consolidated,
        agents=agents,
        filename=os.path.join(plots_dir, "cumulative_rewards.svg"),
        window=min(50, len(df_consolidated) // 10 + 1)
    )
    print(f"   ✅ Guardado: {plots_dir}/cumulative_rewards.svg")
    
    # 2. Penetraciones
    print(f"\n🌍 Generando gráfico de penetraciones...")
    
    # Calcular penetraciones por episodio
    penetrations = []
    for file_path in sorted(Path("results/evolution").glob("episode_*.csv")):
        try:
            df_ep = pd.read_csv(file_path)
            from analysis_tools.metrics import compute_penetration_metrics
            import re
            ep_match = re.search(r"episode_(\d+)\.csv", str(file_path))
            ep_num = int(ep_match.group(1)) if ep_match else 0
            
            pen = compute_penetration_metrics(df_ep)
            pen["episode"] = ep_num
            penetrations.append(pen)
        except Exception as e:
            print(f"⚠️  Error procesando {file_path}: {e}")
    
    if penetrations:
        df_pen = pd.DataFrame(penetrations)
        plot_penetrations(df_pen, filename=os.path.join(plots_dir, "penetrations.svg"))
        print(f"   ✅ Guardado: {plots_dir}/penetrations.svg")
    else:
        print(f"   ⚠️  No se pudieron calcular penetraciones.")
    
    # 3. Métricas agregadas (promedio de últimos 10 episodios)
    print(f"\n📊 Generando gráfico de métricas agregadas...")
    
    recent_episodes = sorted(Path("results/evolution").glob("episode_*.csv"))[-10:]
    
    all_metrics = []
    for file_path in recent_episodes:
        try:
            df_ep = pd.read_csv(file_path)
            metrics = compute_all_metrics(df_ep)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"⚠️  Error procesando {file_path}: {e}")
    
    if all_metrics:
        df_metrics_agg = pd.DataFrame(all_metrics)
        
        # Métricas operativas
        operational_metrics = {
            "MEAN": df_metrics_agg["MEAN"].mean(),
            "IAE": df_metrics_agg["IAE"].mean(),
            "ISE": df_metrics_agg["ISE"].mean(),
            "Variability": df_metrics_agg["Variability"].mean(),
        }
        
        plot_metrics_bars(
            operational_metrics,
            title="Métricas Operativas (Promedio últimos 10 ep)",
            ylabel="Valor",
            filename=os.path.join(plots_dir, "operational_metrics.svg")
        )
        print(f"   ✅ Guardado: {plots_dir}/operational_metrics.svg")
    else:
        print(f"   ⚠️  No se pudieron calcular métricas agregadas.")
    
    # 4. Histograma de balance energético (último episodio)
    print(f"\n📊 Generando histograma de balance energético (último episodio)...")
    
    last_episode_files = sorted(Path("results/evolution").glob("episode_*.csv"))
    if last_episode_files:
        try:
            df_last = pd.read_csv(last_episode_files[-1])
            plot_energy_balance_histogram(
                df_last,
                filename=os.path.join(plots_dir, "energy_balance_histogram.svg"),
                bins=50
            )
            print(f"   ✅ Guardado: {plots_dir}/energy_balance_histogram.svg")
        except Exception as e:
            print(f"   ⚠️  Error generando histograma: {e}")
    else:
        print(f"   ⚠️  No se encontró el último episodio.")
    
    # 5. Serie temporal de balance energético (primer y último episodio)
    print(f"\n📈 Generando series temporales de balance energético...")
    
    if len(last_episode_files) >= 1:
        # Primer episodio
        try:
            df_first = pd.read_csv(last_episode_files[0])
            plot_time_series(
                df_first,
                x_col="step",
                y_col="env_energy_balance",
                title="Balance Energético - Primer Episodio",
                xlabel="Paso Temporal",
                ylabel="Balance (W)",
                filename=os.path.join(plots_dir, "balance_first_episode.svg")
            )
            print(f"   ✅ Guardado: {plots_dir}/balance_first_episode.svg")
        except Exception as e:
            print(f"   ⚠️  Error graficando primer episodio: {e}")
        
        # Último episodio
        try:
            df_last = pd.read_csv(last_episode_files[-1])
            plot_time_series(
                df_last,
                x_col="step",
                y_col="env_energy_balance",
                title="Balance Energético - Último Episodio",
                xlabel="Paso Temporal",
                ylabel="Balance (W)",
                filename=os.path.join(plots_dir, "balance_last_episode.svg")
            )
            print(f"   ✅ Guardado: {plots_dir}/balance_last_episode.svg")
        except Exception as e:
            print(f"   ⚠️  Error graficando último episodio: {e}")
    
    print("\n" + "="*80)
    print(f"✅ Gráficos generados en: {plots_dir}/")
    print("   - cumulative_rewards.svg")
    print("   - penetrations.svg")
    print("   - operational_metrics.svg")
    print("   - energy_balance_histogram.svg")
    print("   - balance_first_episode.svg")
    print("   - balance_last_episode.svg")
    print("="*80)


if __name__ == "__main__":
    main()
