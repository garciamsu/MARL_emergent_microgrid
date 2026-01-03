import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import sys

# Add project root to path
# This module lives under analysis/common/, so the repo root is two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.csv_handler import read_result_csv


def clear_directories(short: bool = True) -> Dict[str, List[str]]:
    """Limpia (solo archivos) varias carpetas de resultados agregando los mensajes en bloques.

    Directorios considerados:
        - results/
        - results/evolution/
        - results/plots/
        - results/logs/*agent (battery, grid, load, solar, wind)

    El comportamiento anterior imprimía una línea por cada evento (borrado, vacío, inexistente, ignorado).
    Ahora se agrupan por categoría para reducir el ruido:
        Ignored (not a file): <lista>
        Deleted: <lista>
        No files in: <lista>
        Directory does not exist: <lista>
        Could not delete: <archivo -> error>

    Returns:
        dict con las listas recopiladas (útil para pruebas o logging estructurado).
    """

    # Asegurar ciertos directorios base mínimos (evita muchos 'no existe')
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/evolution", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    directories = [
        "results/",
        "results/evolution/",
        "results/plots/",
        "results/logs/batteryagent",
        "results/logs/gridagent",
        "results/logs/loadagent",
        "results/logs/solaragent",
        "results/logs/windagent",
    ]

    collected: Dict[str, List[str]] = {
        "ignored": [],
        "deleted": [],
        "empty": [],
        "missing": [],
        "errors": [],  # formato: "ruta -> error"
    }

    for dir_path in directories:
        if not os.path.exists(dir_path):
            collected["missing"].append(dir_path)
            continue

        files = glob.glob(os.path.join(dir_path, "*"))
        if not files:
            collected["empty"].append(dir_path)
            continue

        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    collected["deleted"].append(file_path)
                except Exception as e:  # pragma: no cover - muy raro pero útil
                    collected["errors"].append(f"{file_path} -> {e}")
            else:
                collected["ignored"].append(file_path)

    # Construir salida agregada en orden similar al flujo original
    output_lines: List[str] = []
    if collected["ignored"]:
        output_lines.append("Ignored (not a file): " + ", ".join(collected["ignored"]))
    if collected["deleted"]:
        output_lines.append("Deleted: " + ", ".join(collected["deleted"]))
    if collected["empty"]:
        output_lines.append("No files in: " + ", ".join(collected["empty"]))
    if collected["missing"]:
        output_lines.append("Directory does not exist: " + ", ".join(collected["missing"]))
    if collected["errors"]:
        output_lines.append("Could not delete: " + ", ".join(collected["errors"]))

    if short:
        # Mensaje ultra resumido solo con conteos
        print(
            "Cleanup => "
            f"deleted:{len(collected['deleted'])} | "
            f"ignored:{len(collected['ignored'])} | "
            f"empty:{len(collected['empty'])} | "
            f"missing:{len(collected['missing'])} | "
            f"errors:{len(collected['errors'])}"
        )
    else:
        if output_lines:
            print(" | ".join(output_lines) + " | Cleanup completed.")
        else:
            print("Cleanup completed.")

    return collected


def load_episode_csvs(
    pattern: str = "results/evolution/episode_*.csv",
    max_episodes: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    Carga todos los CSVs de episodios que coincidan con el patrón.
    
    Args:
        pattern: Patrón glob para buscar archivos CSV de episodios.
        max_episodes: Límite de episodios a cargar (None = todos).
    
    Returns:
        Lista de DataFrames, uno por episodio.
    """
    files = sorted(glob.glob(pattern))
    if max_episodes:
        files = files[:max_episodes]
    
    dfs = []
    for f in files:
        try:
            df = read_result_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  No se pudo cargar {f}: {e}")
    
    return dfs


def load_all_episodes_metrics(
    pattern: str = "results/evolution/episode_*.csv",
    max_episodes: Optional[int] = None
) -> pd.DataFrame:
    """
    Carga todos los episodios y calcula métricas agregadas por episodio.
    
    Args:
        pattern: Patrón glob para buscar archivos CSV de episodios.
        max_episodes: Límite de episodios a cargar (None = todos).
    
    Returns:
        DataFrame con una fila por episodio y columnas de métricas agregadas.
    """
    files = sorted(glob.glob(pattern))
    if max_episodes:
        files = files[:max_episodes]
    
    episodes_data = []
    
    for file_path in files:
        ep_match = re.search(r"episode_(\d+)\.csv", file_path)
        if not ep_match:
            continue
        
        ep_num = int(ep_match.group(1))
        
        try:
            df = read_result_csv(file_path)
        except Exception as e:
            print(f"⚠️  Error cargando {file_path}: {e}")
            continue
        
        # Métricas agregadas por episodio
        metrics = {
            "episode": ep_num,
            "mean_demand_power": df["env_demand_power"].mean() if "env_demand_power" in df.columns else 0,
            "mean_energy_balance": df["env_energy_balance"].mean() if "env_energy_balance" in df.columns else 0,
            "mean_renewable_power": df["env_total_renewable"].mean() if "env_total_renewable" in df.columns else 0,
            "mean_total_power": df["env_total_power"].mean() if "env_total_power" in df.columns else 0,
        }
        
        # Recompensas por agente
        reward_cols = [c for c in df.columns if c.startswith("reward_")]
        for col in reward_cols:
            agent_name = col.replace("reward_", "").replace("#0", "")
            metrics[f"total_reward_{agent_name}"] = df[col].sum()
        
        # Tasas de activación
        if "action_solar#0" in df.columns:
            metrics["solar_activation_rate"] = (df["action_solar#0"] == 1).mean()
        if "action_wind#0" in df.columns:
            metrics["wind_activation_rate"] = (df["action_wind#0"] == 1).mean()
        if "action_battery#0" in df.columns:
            metrics["battery_charge_rate"] = (df["action_battery#0"] == 1).mean()
            metrics["battery_discharge_rate"] = (df["action_battery#0"] == 2).mean()
        if "action_grid#0" in df.columns:
            metrics["grid_import_rate"] = (df["action_grid#0"] == 1).mean()
        if "action_load#0" in df.columns:
            metrics["load_reduction_rate"] = (df["action_load#0"] == 0).mean()
        
        episodes_data.append(metrics)
    
    return pd.DataFrame(episodes_data)


def digitize_clip(value: float, bins: np.ndarray) -> int:
    """
    Discretiza un valor según bins y asegura que el índice esté dentro del rango.
    
    Args:
        value: Valor a discretizar.
        bins: Array de bins (bordes).
    
    Returns:
        Índice discretizado (0 a len(bins)-2).
    """
    idx = np.digitize([value], bins)[0] - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    return int(idx)