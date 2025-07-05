import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import glob
import os


def load_latest_evolution_csv():
    """
    Busca el archivo learning_XXX.csv más reciente en /results/evolution/,
    lee su contenido con manejo robusto de separador y encoding,
    y devuelve un DataFrame listo para procesar.
    """
    import glob

    # 1. Buscar todos los archivos que coincidan
    files = glob.glob("results/evolution/learning_*.csv")
    if not files:
        raise FileNotFoundError("No se encontraron archivos learning_XXX.csv en /results/evolution/")

    # 2. Extraer los números y encontrar el máximo
    numbers = []
    for f in files:
        match = re.search(r"learning_(\d+)\.csv", f)
        if match:
            numbers.append(int(match.group(1)))

    if not numbers:
        raise ValueError("No se encontraron números de episodio en los nombres de archivos.")

    max_number = max(numbers)
    latest_file = f"results/evolution/learning_{max_number}.csv"
    print(f"Archivo seleccionado: {latest_file}")

    # 3. Intentar lectura robusta
    try:
        df = pd.read_csv(latest_file, sep=",", encoding="utf-8")
    except Exception as e1:
        print("Primera lectura falló, reintentando con ';' y latin-1...")
        try:
            df = pd.read_csv(latest_file, sep=";", encoding="latin-1")
        except Exception as e2:
            raise RuntimeError(
                f"No se pudo leer el archivo.\nPrimer error: {e1}\nSegundo error: {e2}"
            )

    return df

def plot_metric(df, field, ylabel, filename_svg):
    """
    Grafica la evolución de una métrica a lo largo de los episodios.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df['Episode'], df[field], marker='o')
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename_svg, format="svg")
    plt.close()

def compute_q_diff_norm(current_q, prev_q):
    """
    Calcula la norma L2 de la diferencia entre dos Q-tables.
    
    current_q: dict de dicts, Q-table actual.
    prev_q: dict de dicts, Q-table anterior.
    
    Devuelve: float
    """
    diffs = []
    for state in current_q:
        for action in current_q[state]:
            curr = current_q[state][action]
            prev = prev_q.get(state, {}).get(action, 0.0)
            diffs.append((curr - prev)**2)
    return np.sqrt(sum(diffs))

def check_stability(df, iae_threshold, var_threshold=1.0):
    """
    Verifica si la IAE y la varianza se mantienen estables en los últimos episodios.
    
    df: DataFrame con las métricas.
    iae_threshold: Umbral de IAE aceptable.
    var_threshold: Umbral de varianza aceptable.
    
    Devuelve: dict con resultados.
    """
    # Filtrar últimos 200 episodios
    df_recent = df[df['Episode'] >= df['Episode'].max() - 200]
    
    iae_mean = df_recent["IAE"].mean()
    var_mean = df_recent["Var_dif"].mean()
    
    result = {
        "IAE_mean": iae_mean,
        "Var_mean": var_mean,
        "IAE_stable": iae_mean <= iae_threshold,
        "Var_stable": var_mean <= var_threshold
    }
    return result

def process_evolution_data(df):
    """
    Aplica la transformación de bat_state y valida columnas necesarias.
    Devuelve el DataFrame listo para graficar.
    """
    required_columns = [
        "solar_state",
        "wind_state",
        "bat_state",
        "bat_soc",
        "grid_state",
        "dif",
        "demand"
    ]

    # Verificar columnas
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no está en el DataFrame.")

    # Copiar para no modificar el original
    df_plot = df.copy()

    # Transformar bat_state según reglas
    def transform_bat_state(x):
        if x == 0:
            return 0
        elif x == 1:
            return -1
        elif x == 2:
            return 1
        else:
            return 0  # Por seguridad

    df_plot["bat_state_transformed"] = df_plot["bat_state"].apply(transform_bat_state)

    return df_plot

def plot_coordination(df):
    """
    Genera una gráfica SVG con 6 subgráficas alineadas verticalmente:
    Solar, Wind, Battery (State + SOC), Grid, Demand, Dif.
    Usa barras y líneas según corresponda.
    """
    # Mapeo de colores fijos
    colors = {
        "solar": "orange",
        "wind": "blue",
        "bat_state": "green",
        "bat_soc": "green",
        "grid": "purple",
        "demand": "black",
        "dif": "red"
    }

    fig, axes = plt.subplots(
        6, 1,
        figsize=(12, 18),
        sharex=True,
        constrained_layout=True
    )

    # Eje X dinámico basado en el número de filas
    time = list(range(1, len(df)+1))

    # Subgráfica (A): Solar
    ax = axes[0]
    ax2 = ax.twinx()
    ax.bar(time, df["solar_state"], color=colors["solar"], label="Solar State")
    ax2.plot(
        time,
        df["solar"],
        linestyle="--",
        color=colors["solar"],
        linewidth=2.5,
        label="Solar Power"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Power")
    ax.set_title("(A)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subgráfica (B): Wind
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(time, df["wind_state"], color=colors["wind"], label="Wind State")
    ax2.plot(
        time,
        df["wind"],
        linestyle="--",
        color=colors["wind"],
        linewidth=2.5,
        label="Wind Power"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Power")
    ax.set_title("(B)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subgráfica (C): Battery State + SOC
    ax = axes[2]
    ax2 = ax.twinx()
    ax.bar(time, df["bat_state_transformed"], color=colors["bat_state"], label="Battery State")
    ax2.plot(
        time,
        df["bat_soc"],
        linestyle="--",
        color=colors["bat_soc"],
        linewidth=2.5,
        label="Battery SOC"
    )
    ax.set_ylabel("State (-1/0/1)")
    ax2.set_ylabel("SOC [0-1]")
    ax.set_title("(C)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subgráfica (D): Grid
    ax = axes[3]
    ax2 = ax.twinx()
    ax.bar(time, df["grid_state"], color=colors["grid"], label="Grid State")
    ax2.plot(
        time,
        df["price"],
        linestyle="--",
        color=colors["grid"],
        linewidth=2.5,
        label="Price"
    )
    ax.set_ylabel("State")
    ax2.set_ylabel("Price")
    ax.set_title("(D)", loc="center", pad=10)
    ax.grid(True, which='both')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Subgráfica (E): Demand (línea punteada)
    ax = axes[4]
    ax.plot(
        time,
        df["demand"],
        linestyle="--",
        color=colors["demand"],
        linewidth=2.5,
        label="Demand"
    )
    ax.set_ylabel("Power")
    ax.set_title("(E)", loc="center", pad=10)
    ax.grid(True, which='both')
    ax.legend(loc="upper right")

    # Subgráfica (F): Dif (área)
    ax = axes[5]
    ax.fill_between(
        time,
        0,
        df["dif"],
        color=colors["dif"],
        alpha=0.5,
        label="Energy Balance"
    )
    ax.set_ylabel("Power")
    ax.set_xlabel("Time Steps")
    ax.set_title("(F)", loc="center", pad=10)
    ax.grid(True, which='both')
    ax.legend(loc="upper right")

    # Eje X con ticks enumerados
    for ax in axes:
        ax.set_xticks(time)

    # Guardar SVG
    output_path = "results/plots/plot_coordination_last.svg"
    fig.savefig(output_path, format="svg")
    print(f"Gráfico guardado en {output_path}")

    plt.show()

def clear_results_directories():
    """
    Elimina todos los archivos dentro de los directorios:
    results/, results/evolution/, results/plots/, results/q_tables/.
    No elimina los directorios en sí, solo los contenidos.
    """
    directories = [
        "results/",
        "results/evolution/",
        "results/plots/",
        "results/q_tables/"
    ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directorio no existe: {dir_path}")
            continue

        files = glob.glob(os.path.join(dir_path, "*"))
        if not files:
            print(f"No hay archivos en {dir_path}")
            continue

        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Eliminado: {file_path}")
                except Exception as e:
                    print(f"No se pudo eliminar {file_path}: {e}")
            else:
                print(f"Ignorado (no es un archivo): {file_path}")

    print("Limpieza completada.")