import numpy as np
import matplotlib.pyplot as plt

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
