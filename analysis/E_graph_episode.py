import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import chardet
import matplotlib.patches as mpatches # <-- Importado para la leyenda del Panel 6
import sys
from pathlib import Path

# Add root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.csv_handler import read_result_csv
from configs.loader import load_config


def get_power_unit_label():
    """
    Determine the correct power unit label based on power_scale_factor in config.
    
    Returns the appropriate unit string for power measurements:
    - If power_scale_factor = 1000.0: data is scaled from kW to W → label = 'W'
    - If power_scale_factor = 1.0: data remains in kW → label = 'kW'
    - For other factors, calculate the appropriate unit
    """
    try:
        cfg = load_config()
        scale_factor = cfg.get("simulation", {}).get("power_scale_factor", 1.0)
    except Exception:
        scale_factor = 1.0
    
    # Base unit in dataset is kW. After scaling by power_scale_factor:
    # factor=1000.0 → kW * 1000 = W
    # factor=1.0 → kW (unchanged)
    # factor=0.001 → MW
    if abs(scale_factor - 1000.0) < 1e-9:
        return "W"
    elif abs(scale_factor - 1.0) < 1e-9:
        return "kW"
    elif abs(scale_factor - 0.001) < 1e-9:
        return "MW"
    elif scale_factor > 1.0:
        # Scaled up from kW
        return "W" if scale_factor >= 100 else "kW"
    else:
        # Scaled down from kW
        return "MW" if scale_factor <= 0.01 else "kW"


# --- CONFIGURACIÓN PRINCIPAL ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback para entornos interactivos (como Jupyter)
    SCRIPT_DIR = os.getcwd()

# 🔧 Corrección: el CSV está un nivel arriba de analysis/
BASE_DIRECTORY = os.path.join(SCRIPT_DIR, "..", "results", "evolution")

# Directorio base para episodios offline (evaluación/explotación)
OFFLINE_BASE_DIRECTORY = os.path.join(BASE_DIRECTORY, "offline")

# Dynamic episode selection: find the latest episode file
def get_latest_episode_number(base_dir):
    """Find the episode number with the highest value in the directory."""
    pattern = os.path.join(base_dir, "episode_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("⚠️  No se encontraron archivos episode_*.csv")
        return None
    
    episode_numbers = []
    for file in files:
        basename = os.path.basename(file)
        try:
            # Extract number from "episode_123.csv"
            num_str = basename.replace("episode_", "").replace(".csv", "")
            episode_numbers.append(int(num_str))
        except ValueError:
            continue
    
    if not episode_numbers:
        print("⚠️  No se pudieron extraer números de episodio")
        return None
    
    return max(episode_numbers)

EPISODE_TO_PLOT = get_latest_episode_number(BASE_DIRECTORY) or 0
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, "..", "results", "plots", "episode_dynamics.svg")

# Nombre base para gráficos offline; se completará con el run_id
OFFLINE_OUTPUT_BASENAME = os.path.join(
    SCRIPT_DIR, "..", "results", "plots", "episode_dynamics_offline_{}.svg"
)

# --- PARÁMETROS DE ESTILO ---
STATE_FILL_TRANSPARENCY = 0.2
POWER_LINE_STYLE = '--'
POWER_LINE_WIDTH = 2.0
AXIS_FONT_WEIGHT = 'bold'
AXIS_FONT_SIZE = 11

# --- CONFIGURACIÓN DE LOS PANELES ---
# (Las etiquetas 'label' ahora solo se usarán para la LEYENDA, no para los ejes)
# NOTE: {power_unit} will be replaced dynamically based on power_scale_factor
PLOT_CONFIG = {
    'panel_1': {
        'title': '(A)',
        'color': '#FFA500',
        'left_Y': {'column': 'potential_solar#0', 'label': 'Potential solar ({power_unit})'},
        'right_Y': {'column': 'action_solar#0', 'label': 'Solar state'}
    },
    'panel_2': {
        'title': '(B)',
        'color': '#87CEEB',
        'left_Y': {'column': 'potential_wind#0', 'label': 'Potential wind ({power_unit})'},
        'right_Y': {'column': 'action_wind#0', 'label': 'Wind state'}
    },
    'panel_3': {
        'title': '(C)',
        'color': '#800080',
        'left_Y': {'column': 'soc_battery#0', 'label': 'SoC (0-1)'},
        'right_Y': {'column': 'action_battery#0', 'label': 'Battery state'}
    },
    'panel_4': {
        'title': '(D)',
        'color': '#36454F',
        'left_Y': {'column': 'env_price', 'label': 'Price ($/kWh)'},
        'right_Y': {'column': 'action_grid#0', 'label': 'Grid state'}
    },
    'panel_5': {
        'title': '(E)',
        'color': '#FF0000',
        'left_Y': {'column': 'env_demand_power', 'label': 'Demand ({power_unit})'},
        'right_Y': {'column': 'action_load#0', 'label': 'Load state'}
    },
    'panel_6': {
        'title': '(F)',
        'left_Y': {'column': 'env_energy_balance', 'label': 'Energy Balance ({power_unit})'},
        'color_positive': '#28A745', 
        'color_negative': '#FF0000',
        # Etiquetas para la nueva leyenda del Panel 6
        'label_positive': 'Surplus (+)',
        'label_negative': 'Deficit (-)'
    }
}


def get_active_panels(df, config):
    """
    Filter out panels that have no data and return only active panels.
    
    A panel is considered to have data if at least one of its columns
    (left_Y or right_Y) exists in the dataframe.
    
    Returns:
        list of tuples: [(panel_key, panel_config), ...] for panels with data
    """
    active_panels = []
    for panel_key, panel_config in config.items():
        has_data = False
        
        # Check left_Y column
        if 'left_Y' in panel_config:
            col = panel_config['left_Y']['column']
            if col in df.columns and not df[col].isna().all():
                has_data = True
        
        # Check right_Y column
        if 'right_Y' in panel_config:
            col = panel_config['right_Y']['column']
            if col in df.columns and not df[col].isna().all():
                has_data = True
        
        if has_data:
            active_panels.append((panel_key, panel_config))
    
    return active_panels


def apply_power_unit_labels(config):
    """
    Replace {power_unit} placeholder in all label strings with the actual unit.
    Returns a deep copy of the config with substituted labels.
    """
    import copy
    power_unit = get_power_unit_label()
    new_config = copy.deepcopy(config)
    
    for panel_key, panel_cfg in new_config.items():
        if 'left_Y' in panel_cfg and 'label' in panel_cfg['left_Y']:
            panel_cfg['left_Y']['label'] = panel_cfg['left_Y']['label'].format(power_unit=power_unit)
        if 'right_Y' in panel_cfg and 'label' in panel_cfg['right_Y']:
            panel_cfg['right_Y']['label'] = panel_cfg['right_Y']['label'].format(power_unit=power_unit)
    
    return new_config


def _build_time_ticks(time_steps: np.ndarray, max_ticks: int = 50):
    """Return a subset of time steps to use as x-ticks.

    Generates ticks every 2 hours (0, 2, 4, 6, ...) assuming dt_h=1.0.
    This ensures consistent spacing and prevents label overlap.
    """
    n = len(time_steps)
    if n == 0:
        return time_steps
    
    # Generate ticks every 2 hours (step = 2)
    step = 10
    return time_steps[::step]


def plot_episode_dynamics(base_dir, episode_num, config):
    """Crea el gráfico del episodio de entrenamiento y guarda un SVG."""
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

    # Apply power unit labels based on config
    config_with_units = apply_power_unit_labels(config)
    
    # Filter out panels without data and get only active panels
    active_panels = get_active_panels(df, config_with_units)
    
    if not active_panels:
        print("⚠️  No hay paneles con datos disponibles para graficar.")
        return
    
    num_panels = len(active_panels)
    print(f"📊 Paneles activos: {num_panels} de {len(config)}")

    time_steps = np.arange(len(df))

    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-paper')

    # Create figure with only the number of active panels
    fig, axes = plt.subplots(nrows=num_panels, ncols=1, sharex=True, 
                              figsize=(15, 2 + num_panels * 1.5))
    
    # Handle case of single panel (axes is not a list)
    if num_panels == 1:
        axes = [axes]

    # Dynamic title labels: A, B, C, D, E, F...
    title_letters = [chr(ord('A') + i) for i in range(num_panels)]

    for i, (panel_key, panel_config) in enumerate(active_panels):
        ax = axes[i]
        color = panel_config.get('color', '#000000')
        
        # Dynamic title based on position
        dynamic_title = f"({title_letters[i]})"
        ax.set_title(dynamic_title, loc='center',
                     fontweight=AXIS_FONT_WEIGHT, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax_r = None 
        handles, labels = [], [] # Para la leyenda

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
                    ax.axhline(y=0, color='#28A745', linestyle='-', linewidth=1.5, zorder=5)
                    
                    # --- MODIFICACIÓN: Crear leyenda manual para Panel 6 ---
                    pos_patch = mpatches.Patch(color=panel_config['color_positive'], 
                                               label=panel_config['label_positive'], 
                                               alpha=STATE_FILL_TRANSPARENCY)
                    neg_patch = mpatches.Patch(color=panel_config['color_negative'], 
                                               label=panel_config['label_negative'], 
                                               alpha=STATE_FILL_TRANSPARENCY)
                    handles.extend([pos_patch, neg_patch])
                    # --- FIN MODIFICACIÓN ---
                    
                else:
                    line, = ax.plot(time_steps, df[col], color=color,
                                    linestyle=POWER_LINE_STYLE, linewidth=POWER_LINE_WIDTH,
                                    label=left['label']) # 'label' para leyenda
                    handles.append(line)
                
                # --- MODIFICACIÓN: Etiqueta de eje Y eliminada ---
                # ax.set_ylabel(left['label'], fontweight='bold', ...) # <-- ELIMINADO
                ax.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)
            else:
                print(f"⚠️  Columna no encontrada: {col} (Panel {panel_key})")

        # --- Panel derecho ---
        if 'right_Y' in panel_config:
            right = panel_config['right_Y']
            col = right['column']
            if col in df.columns:
                ax_r = ax.twinx()
                line, = ax_r.step(time_steps, df[col], where='post', color=color, 
                                  label=right['label']) # 'label' para leyenda
                handles.append(line)
                
                ax_r.fill_between(time_steps, df[col], step='post',
                                  color=color, alpha=STATE_FILL_TRANSPARENCY)
                
                # --- MODIFICACIÓN: Etiqueta de eje Y eliminada ---
                # ax_r.set_ylabel(right['label'], fontweight='bold', ...) # <-- ELIMINADO
                ax_r.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)
                
                y_min = min(0, df[col].min())
                y_max = max(1, df[col].max() + 1)
                ax_r.set_ylim(y_min, y_max)
                
                if np.issubdtype(df[col].dtype, np.integer):
                    ax_r.set_yticks(np.arange(int(y_min), int(y_max)))

            else:
                print(f"⚠️  Columna no encontrada: {col} (Panel {panel_key})")

        # --- MODIFICACIÓN: Añadir Leyenda (combinada y reubicada) ---
        labels = [h.get_label() for h in handles]
        if handles:
            ax.legend(handles, labels, 
                      loc='upper left', 
                      bbox_to_anchor=(1.02, 1.0), # <-- Reubicada fuera del gráfico
                      fontsize=AXIS_FONT_SIZE - 1, 
                      frameon=True, 
                      shadow=False)
        # --- FIN MODIFICACIÓN ---

        # Ajuste de ticks en el eje X para evitar solapamiento de horas
        ticks = _build_time_ticks(time_steps, max_ticks=50)
        ax.set_xticks(ticks)
        
        ax.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=AXIS_FONT_SIZE)
        
        ax.set_xlim(time_steps[0], time_steps[-1])


    axes[-1].set_xlabel("Time steps [Hour]", fontweight='bold', fontsize=AXIS_FONT_SIZE)
    
    # Ajustar layout para dar espacio a la leyenda
    plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 0.98]) # <-- Ajustado rect[2] a 0.85

    try:
        plt.savefig(OUTPUT_FILENAME, format='svg', dpi=300, bbox_inches='tight')
        print(f"\n✅ ¡Gráfica guardada correctamente en:\n{OUTPUT_FILENAME}")
    except Exception as e:
        print(f"❌ Error al guardar la gráfica: {e}")

def read_csv_auto(file_path):
    """
    Lee un archivo CSV usando el formato estandarizado de la aplicación.
    Usa read_result_csv() para mantener consistencia en decimales y separadores.
    """
    try:
        df = read_result_csv(file_path)
        print(f"✅ Archivo leído correctamente con formato estandarizado")
        return df
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return None

if __name__ == "__main__":
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🔍 Episodio seleccionado automáticamente: {EPISODE_TO_PLOT}")
    print(f"📂 Directorio base (training): {BASE_DIRECTORY}")

    # Plot latest training episode dynamics
    plot_episode_dynamics(BASE_DIRECTORY, EPISODE_TO_PLOT, PLOT_CONFIG)

    # Additionally, if there are offline evolution files, plot those too.
    offline_pattern = os.path.join(OFFLINE_BASE_DIRECTORY, "episode_offline_*.csv")
    offline_files = glob.glob(offline_pattern)
    if offline_files:
        print(f"\n🔍 Encontrados {len(offline_files)} archivos offline para graficar")
        offline_output_dir = os.path.dirname(OFFLINE_OUTPUT_BASENAME)
        if not os.path.exists(offline_output_dir):
            os.makedirs(offline_output_dir)

        for csv_path in offline_files:
            basename = os.path.basename(csv_path)
            run_id = basename.replace("episode_offline_", "").replace(".csv", "")
            print(f"  ➜ Graficando offline_run={run_id} desde {csv_path}")

            df_off = read_csv_auto(csv_path)
            if df_off is None or df_off.empty:
                print(f"  ⚠️  Archivo vacío o ilegible: {csv_path}")
                continue

            # Apply power unit labels based on config
            config_with_units = apply_power_unit_labels(PLOT_CONFIG)
            
            # Filter out panels without data and get only active panels
            active_panels = get_active_panels(df_off, config_with_units)
            
            if not active_panels:
                print(f"  ⚠️  No hay paneles con datos para offline_run={run_id}")
                continue
            
            num_panels = len(active_panels)
            print(f"  📊 Paneles activos: {num_panels} de {len(PLOT_CONFIG)}")

            time_steps = np.arange(len(df_off))
            matplotlib.use('Agg')
            plt.style.use('seaborn-v0_8-paper')

            # Create figure with only the number of active panels
            fig, axes = plt.subplots(nrows=num_panels, ncols=1, sharex=True, 
                                      figsize=(15, 2 + num_panels * 1.5))
            
            # Handle case of single panel (axes is not a list)
            if num_panels == 1:
                axes = [axes]
            
            # Dynamic title labels: A, B, C, D, E, F...
            title_letters = [chr(ord('A') + i) for i in range(num_panels)]

            for i, (panel_key, panel_config) in enumerate(active_panels):
                ax = axes[i]
                color = panel_config.get('color', '#000000')
                
                # Dynamic title based on position
                dynamic_title = f"({title_letters[i]})"
                ax.set_title(dynamic_title, loc='center',
                             fontweight=AXIS_FONT_WEIGHT, fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)

                ax_r = None
                handles, labels = [], []

                # Panel izquierdo
                if 'left_Y' in panel_config:
                    left = panel_config['left_Y']
                    col = left['column']
                    if col in df_off.columns:
                        if panel_key == 'panel_6':
                            vals = df_off[col]
                            bar_colors = [
                                panel_config['color_positive'] if v >= 0 else panel_config['color_negative']
                                for v in vals
                            ]
                            ax.bar(time_steps, vals,
                                   color=bar_colors,
                                   width=1.0,
                                   alpha=STATE_FILL_TRANSPARENCY,
                                   zorder=3)
                            ax.axhline(y=0, color='#28A745', linestyle='-', linewidth=1.5, zorder=5)

                            pos_patch = mpatches.Patch(color=panel_config['color_positive'],
                                                       label=panel_config['label_positive'],
                                                       alpha=STATE_FILL_TRANSPARENCY)
                            neg_patch = mpatches.Patch(color=panel_config['color_negative'],
                                                       label=panel_config['label_negative'],
                                                       alpha=STATE_FILL_TRANSPARENCY)
                            handles.extend([pos_patch, neg_patch])
                        else:
                            line, = ax.plot(time_steps, df_off[col], color=color,
                                            linestyle=POWER_LINE_STYLE, linewidth=POWER_LINE_WIDTH,
                                            label=left['label'])
                            handles.append(line)

                        ax.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)
                    else:
                        print(f"⚠️  Columna no encontrada (offline): {col} (Panel {panel_key})")

                # Panel derecho
                if 'right_Y' in panel_config:
                    right = panel_config['right_Y']
                    col = right['column']
                    if col in df_off.columns:
                        ax_r = ax.twinx()
                        line, = ax_r.step(time_steps, df_off[col], where='post', color=color,
                                          label=right['label'])
                        handles.append(line)

                        ax_r.fill_between(time_steps, df_off[col], step='post',
                                          color=color, alpha=STATE_FILL_TRANSPARENCY)
                        ax_r.tick_params(axis='y', labelcolor='black', labelsize=AXIS_FONT_SIZE)

                        y_min = min(0, df_off[col].min())
                        y_max = max(1, df_off[col].max() + 1)
                        ax_r.set_ylim(y_min, y_max)

                        if np.issubdtype(df_off[col].dtype, np.integer):
                            ax_r.set_yticks(np.arange(int(y_min), int(y_max)))
                    else:
                        print(f"⚠️  Columna no encontrada (offline): {col} (Panel {panel_key})")

                labels = [h.get_label() for h in handles]
                if handles:
                    ax.legend(handles, labels,
                              loc='upper left',
                              bbox_to_anchor=(1.02, 1.0),
                              fontsize=AXIS_FONT_SIZE - 1,
                              frameon=True,
                              shadow=False)

                ticks = _build_time_ticks(time_steps, max_ticks=50)
                ax.set_xticks(ticks)
                ax.tick_params(axis='x', labelsize=AXIS_FONT_SIZE)
                ax.tick_params(axis='y', labelsize=AXIS_FONT_SIZE)
                ax.set_xlim(time_steps[0], time_steps[-1])

            axes[-1].set_xlabel("Time steps [Hour]", fontweight='bold', fontsize=AXIS_FONT_SIZE)

            plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 0.98])

            out_path = OFFLINE_OUTPUT_BASENAME.format(run_id)
            try:
                plt.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
                print(f"  ✅ Gráfica offline guardada en: {out_path}")
            except Exception as e:
                print(f"  ❌ Error al guardar gráfica offline para run_id={run_id}: {e}")
            finally:
                plt.close(fig)