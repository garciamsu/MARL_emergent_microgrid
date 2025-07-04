import numpy as np
import matplotlib.pyplot as plt
import re

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
