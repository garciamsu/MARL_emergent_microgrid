import pandas as pd
import glob
import re
import os
import matplotlib
import matplotlib.pyplot as plt

def analyze_learning_curves(base_directory):
    """
    Lee todos los archivos 'episode_*.csv' de un directorio base, calcula:
      - Recompensa total por episodio y agente
      - Recompensa acumulada (cumulative reward)
      - Estadísticas descriptivas por agente
    Genera:
      - Un archivo Excel con tres hojas: Total_Reward, Cumulative_Reward, Statistics
      - Una gráfica SVG de las recompensas acumuladas
    """
    
    # 1. Buscar archivos de episodios
    search_pattern = os.path.join(base_directory, '**', 'episode_*.csv')
    episode_files = glob.glob(search_pattern, recursive=True)
    
    if not episode_files:
        print(f"❌ No se encontraron archivos 'episode_*.csv' en la ruta:\n{search_pattern}")
        return

    print(f"✅ Se encontraron {len(episode_files)} archivos. Procesando...")

    potential_agents = [
        'reward_solar#0', 
        'reward_wind#0', 
        'reward_battery#0', 
        'reward_grid#0', 
        'reward_load#0'
    ]
    
    all_episode_rewards = []

    # 2. Procesar cada archivo
    for filepath in episode_files:
        filename = os.path.basename(filepath)
        match = re.search(r'episode_(\d+).csv', filename)
        if not match:
            print(f"Aviso: Omitiendo archivo (nombre no válido): {filename}")
            continue
            
        episode_num = int(match.group(1))
        
        try:
            df = pd.read_csv(filepath)
            
            for reward_col in potential_agents:
                if reward_col in df.columns:
                    total_reward = df[reward_col].sum()  # Recompensa total del episodio
                    agent_name = reward_col.replace('reward_', '').replace('#0', '').capitalize()
                    all_episode_rewards.append({
                        'episode': episode_num,
                        'agent': agent_name,
                        'total_reward': total_reward
                    })
        
        except pd.errors.EmptyDataError:
            print(f"Aviso: Archivo vacío: {filename}")
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

    if not all_episode_rewards:
        print("❌ No se pudo procesar ningún dato de recompensa.")
        return

    # 3. Construir DataFrames
    df_results = pd.DataFrame(all_episode_rewards)
    df_grouped = df_results.groupby(['episode', 'agent'])['total_reward'].sum().reset_index()

    try:
        df_pivot = df_grouped.pivot(index='episode', columns='agent', values='total_reward')
    except Exception as e:
        print(f"Error al pivotar datos: {e}")
        return
        
    df_pivot = df_pivot.sort_index()
    df_cumulative = df_pivot.cumsum()

    print("\n📊 Primeras filas (Cumulative Reward):")
    print(df_cumulative.head())

    # 4. Calcular estadísticas descriptivas
    df_stats = pd.DataFrame({
        'mean': df_pivot.mean(),
        'std': df_pivot.std(),
        'min': df_pivot.min(),
        'max': df_pivot.max(),
        'final_cumulative': df_cumulative.iloc[-1]  # Último valor acumulado
    }).round(3)

    # 5. Guardar archivo Excel con tres hojas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_output = os.path.join(script_dir, 'agent_rewards_data.xlsx')

    try:
        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
            df_pivot.to_excel(writer, sheet_name='Total_Reward')
            df_cumulative.to_excel(writer, sheet_name='Cumulative_Reward')
            df_stats.to_excel(writer, sheet_name='Statistics')
        print(f"✅ Datos guardados en Excel: {excel_output}")
    except Exception as e:
        print(f"❌ Error al guardar Excel: {e}")

    # 6. Graficar la recompensa acumulada
    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-paper')

    num_agents = len(df_cumulative.columns)
    colormap = plt.get_cmap('tab10')
    colors = [colormap(i) for i in range(num_agents)]

    fig, ax = plt.subplots(figsize=(12, 7))
    df_cumulative.plot(kind='line', marker='o', markersize=4, ax=ax, color=colors)
    
    ax.set_title('Agent Learning Curve: Cumulative Reward', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.legend(title='Agent', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # Guardar gráfica en el mismo directorio
    svg_output = os.path.join(script_dir, 'agent_learning_curve_cumulative.svg')
    try:
        plt.savefig(svg_output, format='svg', dpi=300, bbox_inches='tight')
        print(f"✅ Gráfica guardada como: {svg_output}")
    except Exception as e:
        print(f"❌ Error al guardar la gráfica: {e}")

# --- INICIO ---
DIRECTORIO_BASE = os.path.join(os.getcwd(), 'results', 'evolution')
analyze_learning_curves(DIRECTORIO_BASE)
