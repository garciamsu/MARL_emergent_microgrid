import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def plot_metric(df, field, ylabel, filename_svg):
    """Plot básico de una métrica vs episodios."""
    plt.figure()
    plt.plot(df["Episode"], df[field], drawstyle="steps-post")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename_svg, format="svg")
    plt.close()


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: Optional[str] = None
):
    """
    Genera un gráfico de serie temporal.
    
    Args:
        df: DataFrame con los datos.
        x_col: Nombre de la columna X.
        y_col: Nombre de la columna Y.
        title: Título del gráfico.
        xlabel: Etiqueta eje X.
        ylabel: Etiqueta eje Y.
        filename: Ruta para guardar (None = solo mostrar).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df[x_col], df[y_col], linewidth=2, alpha=0.8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_cumulative_rewards(
    df: pd.DataFrame,
    agents: List[str],
    filename: Optional[str] = None,
    window: int = 50
):
    """
    Grafica la evolución de recompensas acumuladas por agente con media móvil.
    
    Args:
        df: DataFrame con columnas episode y total_reward_<agent>.
        agents: Lista de nombres de agentes (sin prefijo total_reward_).
        filename: Ruta para guardar.
        window: Ventana de media móvil.
    """
    colors = {
        "solar": "gold",
        "wind": "skyblue",
        "battery": "green",
        "grid": "red",
        "load": "purple",
    }
    
    plt.figure(figsize=(14, 7))
    
    for agent in agents:
        col = f"total_reward_{agent}"
        if col not in df.columns:
            continue
        
        rolling_mean = df[col].rolling(window=window, center=True).mean()
        color = colors.get(agent, "gray")
        
        plt.plot(
            df["episode"],
            rolling_mean,
            label=agent.capitalize(),
            linewidth=2.5,
            alpha=0.9,
            color=color
        )
        plt.plot(
            df["episode"],
            df[col],
            alpha=0.15,
            linewidth=0.5,
            color=color
        )
    
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel(f"Reward Total (media móvil {window} ep)", fontsize=12)
    plt.title("Evolución de Rewards por Agente", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_metrics_bars(
    metrics: Dict[str, float],
    title: str,
    ylabel: str,
    filename: Optional[str] = None
):
    """
    Grafica métricas como barras.
    
    Args:
        metrics: Diccionario {métrica: valor}.
        title: Título del gráfico.
        ylabel: Etiqueta eje Y.
        filename: Ruta para guardar.
    """
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color="steelblue", alpha=0.8)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_penetrations(
    df: pd.DataFrame,
    filename: Optional[str] = None
):
    """
    Grafica penetraciones de renovables y red como barras apiladas o comparativas.
    
    Args:
        df: DataFrame con columnas episode, Renewable_Penetration, Grid_Penetration.
        filename: Ruta para guardar.
    """
    if "Renewable_Penetration" not in df.columns or "Grid_Penetration" not in df.columns:
        print("⚠️  Columnas de penetración no encontradas.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["episode"],
        df["Renewable_Penetration"] * 100,
        label="Penetración Renovable",
        linewidth=2.5,
        color="green",
        alpha=0.8
    )
    plt.plot(
        df["episode"],
        df["Grid_Penetration"] * 100,
        label="Penetración de Red",
        linewidth=2.5,
        color="red",
        alpha=0.8
    )
    
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel("Penetración (%)", fontsize=12)
    plt.title("Penetraciones de Energía por Episodio", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_energy_balance_histogram(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    bins: int = 50
):
    """
    Genera histograma del balance energético.
    
    Args:
        df: DataFrame con columna env_energy_balance.
        filename: Ruta para guardar.
        bins: Número de bins.
    """
    if "env_energy_balance" not in df.columns:
        print("⚠️  Columna env_energy_balance no encontrada.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(df["env_energy_balance"], bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Balance = 0")
    plt.xlabel("Balance Energético (W)", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución del Balance Energético", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_boxplots_by_episode(
    dfs: List[pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    filename: Optional[str] = None,
    max_episodes: int = 10
):
    """
    Genera boxplots de una columna para múltiples episodios.
    
    Args:
        dfs: Lista de DataFrames (uno por episodio).
        column: Nombre de la columna a graficar.
        title: Título del gráfico.
        ylabel: Etiqueta eje Y.
        filename: Ruta para guardar.
        max_episodes: Máximo de episodios a mostrar.
    """
    data = []
    labels = []
    
    for i, df in enumerate(dfs[:max_episodes]):
        if column in df.columns:
            data.append(df[column].values)
            labels.append(f"Ep {i}")
    
    if not data:
        print(f"⚠️  No se encontró la columna {column} en los episodios.")
        return
    
    plt.figure(figsize=(14, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Episodio", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, format="svg", dpi=150)
        plt.close()
    else:
        plt.show()
