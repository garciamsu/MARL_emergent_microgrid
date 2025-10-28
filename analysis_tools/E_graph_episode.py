import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

# --- CONFIGURACIÓN PRINCIPAL ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔧 Corrección: el CSV está un nivel arriba de analysis_tools/
BASE_DIRECTORY = os.path.join(SCRIPT_DIR, "..", "results", "evolution")

EPISODE_TO_PLOT = 499
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, "episode_dynamics.svg")

# --- PARÁMETROS DE ESTILO ---
STATE_FILL_TRANSPARENCY = 0.2
POWER_LINE_STYLE = '--'
POWER_LINE_WIDTH = 2.0
AXIS_FONT_WEIGHT = 'bold'
AXIS_FONT_SIZE = 11

# --- CONFIGURACIÓN DE LOS PANELES ---
PLOT_CONFIG = {
    'panel_1': {
        'title': '(A)',
        'color': '#FFA500',
        'left_Y': {'column': 'potential_solar#0', 'label': 'Potential (kW)'},
        'right_Y': {'column': 'action_solar#0', 'label': 'State'}
    },
    'panel_2': {
        'title': '(B)',
        'color': '#87CEEB',
        'left_Y': {'column': 'potential_wind#0', 'label': 'Potential (kW)'},
        'right_Y': {'column': 'action_wind#0', 'label': 'State'}
    },
    'panel_3': {
        'title': '(C)',
        'color': '#800080',
        'left_Y': {'column': 'soc_battery#0', 'label': 'SoC (%)'},
        'right_Y': {'column': 'action_battery#0', 'label': 'State'}
    },
    'panel_4': {
        'title': '(D)',
        'color': '#36454F',
        'left_Y': {'column': 'env_price', 'label': 'Price ($/kWh)'},
        'right_Y': {'column': 'action_grid#0', 'label': 'State'}
    },
    'panel_5': {
        'title': '(E)',
        'color': '#FF0000',
        'left_Y': {'column': 'env_demand_power', 'label': 'Power (kW)'}
    },
    'panel_6': {
        'title': '(F)',
        'left_Y': {'column': 'env_energy_balance', 'label': 'Energy Balance (kW)'},
        'color_positive': '#008000',
        'color_negative': '#8B0000'
    }
}


def plot_episode_dynamics(base_dir, episode_num, config):
    """Crea el gráfico del episodio y guarda un SVG junto al script."""
    # Buscar el archivo
    file_path = os.path.join(base_dir, f'episode_{episode_num}.csv')
    if not os.path.exists(file_path):
        recursive_path = os.path.join(base_dir, '**', f'episode_{episode_num}.csv')
        files = glob.glob(recursive_path, recursive=True)
        if files:
            file_path = files[0]
            print(f"✅ Archivo encontrado recursivamente en: {file_path}")
        else:
            print(f"⚠️  No se encontró episode_{episode_num}.csv en {base_dir}")
            return

    # Leer CSV con separador correcto
    try:
        df = pd.read_csv(file_path, sep=';')
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return

    print(f"✅ Archivo cargado: {file_path} ({len(df)} filas)")

    time_steps = np.arange(len(df))

    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-paper')

    # 📏 Figura más ancha y menos alta
    fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(15, 10))

    for i, (panel_key, panel_config) in enumerate(config.items()):
        ax = axes[i]
        color = panel_config.get('color', '#000000')
        ax.set_title(panel_config['title'], loc='center',
                     fontweight=AXIS_FONT_WEIGHT, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- Panel izquierdo ---
        if 'left_Y' in panel_config:
            left = panel_config['left_Y']
            col = left['column']
            if col in df.columns:
                if panel_key == 'panel_6':
                    vals = df[col]
                    bar_colors = [
                        panel_config['color_positive'] if v >= 0 else panel_config['color_negative']
                        for v in vals
                    ]
                    ax.bar(time_steps, vals, color=bar_colors, width=1.0)
                else:
                    ax.plot(time_steps, df[col], color=color,
                            linestyle=POWER_LINE_STYLE, linewidth=POWER_LINE_WIDTH)
                ax.set_ylabel(left['label'], fontweight='bold',
                              color='black', fontsize=AXIS_FONT_SIZE)
                ax.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)
            else:
                print(f"⚠️  Columna no encontrada: {col} (Panel {panel_key})")

        # --- Panel derecho ---
        if 'right_Y' in panel_config:
            right = panel_config['right_Y']
            col = right['column']
            if col in df.columns:
                ax_r = ax.twinx()
                ax_r.step(time_steps, df[col], where='post', color=color)
                ax_r.fill_between(time_steps, df[col], step='post',
                                  color=color, alpha=STATE_FILL_TRANSPARENCY)
                ax_r.set_ylabel(right['label'], fontweight='bold',
                                color='black', fontsize=AXIS_FONT_SIZE)
                ax_r.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)
                ax_r.set_ylim(min(0, df[col].min()), df[col].max() + 1)
            else:
                print(f"⚠️  Columna no encontrada: {col} (Panel {panel_key})")

        # 📊 Mostrar todos los números del eje X
        ax.set_xticks(time_steps)
        ax.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=AXIS_FONT_SIZE)

    axes[-1].set_xlabel("Time steps", fontweight='bold', fontsize=AXIS_FONT_SIZE)
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.98])

    try:
        plt.savefig(OUTPUT_FILENAME, format='svg', dpi=300, bbox_inches='tight')
        print(f"\n✅ ¡Gráfica guardada correctamente en:\n{OUTPUT_FILENAME}")
    except Exception as e:
        print(f"❌ Error al guardar la gráfica: {e}")


if __name__ == "__main__":
    plot_episode_dynamics(BASE_DIRECTORY, EPISODE_TO_PLOT, PLOT_CONFIG)
