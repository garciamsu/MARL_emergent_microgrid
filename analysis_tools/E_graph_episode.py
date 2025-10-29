import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import chardet

# --- CONFIGURACIÓN PRINCIPAL ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback para entornos interactivos (como Jupyter)
    SCRIPT_DIR = os.getcwd()

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
        'left_Y': {'column': 'soc_battery#0', 'label': 'SoC (0-1)'},
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
        'color_positive': '#28A745', 
        'color_negative': '#FF0000' 
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
    df = read_csv_auto(file_path)
    
    if df is None or df.empty:
        print(f"❌ Error: El archivo {file_path} está vacío o no se pudo leer.")
        return

    print(f"✅ Archivo cargado: {file_path} ({len(df)} filas)")

    time_steps = np.arange(len(df))

    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-paper')

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
                    ax.bar(time_steps, vals, 
                           color=bar_colors, 
                           width=1.0, 
                           alpha=STATE_FILL_TRANSPARENCY,
                           zorder=3) 
                    
                    # --- MODIFICACIÓN AQUÍ ---
                    # Cambiado el color de 'black' a '#28A745'
                    ax.axhline(y=0, color='#28A745', linestyle='-', linewidth=1.5, zorder=5) # <-- MODIFICADO

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
                
                y_min = min(0, df[col].min())
                y_max = max(1, df[col].max() + 1)
                ax_r.set_ylim(y_min, y_max)
                
                if np.issubdtype(df[col].dtype, np.integer):
                    ax_r.set_yticks(np.arange(int(y_min), int(y_max)))

            else:
                print(f"⚠️  Columna no encontrada: {col} (Panel {panel_key})")

        if len(time_steps) <= 50:
            ax.set_xticks(time_steps)
        
        ax.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=AXIS_FONT_SIZE)
        
        ax.set_xlim(time_steps[0], time_steps[-1])


    axes[-1].set_xlabel("Time steps", fontweight='bold', fontsize=AXIS_FONT_SIZE)
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.98])

    try:
        plt.savefig(OUTPUT_FILENAME, format='svg', dpi=300, bbox_inches='tight')
        print(f"\n✅ ¡Gráfica guardada correctamente en:\n{OUTPUT_FILENAME}")
    except Exception as e:
        print(f"❌ Error al guardar la gráfica: {e}")

def read_csv_auto(file_path):
    """
    Lee un archivo CSV detectando automáticamente:
      - El encoding (UTF-8, ISO-8859-1, Windows-1252, etc.)
      - El separador (',' o ';')
    Devuelve un DataFrame de pandas.
    """

    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(5000)
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'

        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            first_line = f.readline()

        if ';' in first_line and ',' not in first_line:
            sep = ';'
        elif ',' in first_line and ';' not in first_line:
            sep = ','
        else:
            try:
                df = pd.read_csv(file_path, sep=';', encoding=encoding, on_bad_lines='skip')
                if df.shape[1] > 1:
                    print(f"✅ Archivo leído con separador ';' y codificación '{encoding}'")
                    return df
            except Exception:
                pass
            sep = ',' 

        df = pd.read_csv(file_path, sep=sep, encoding=encoding, on_bad_lines='skip')
        print(f"✅ Archivo leído correctamente con separador '{sep}' y codificación '{encoding}'")
        return df

    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return None

if __name__ == "__main__":
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plot_episode_dynamics(BASE_DIRECTORY, EPISODE_TO_PLOT, PLOT_CONFIG)