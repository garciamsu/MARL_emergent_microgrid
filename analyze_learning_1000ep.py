#!/usr/bin/env python3
"""
Análisis completo de 1000 episodios - Experimento 3
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*80)
print("📊 ANÁLISIS COMPLETO: EXPERIMENTO 3 - 1000 EPISODIOS")
print("="*80)

# Leer todos los episodios
results_dir = Path('results/evolution')
episodes = []

print("\n⏳ Leyendo 1000 episodios...")
for ep_file in sorted(results_dir.glob('episode_*.csv')):
    ep_num = int(ep_file.stem.split('_')[1])
    df = pd.read_csv(ep_file)
    
    metrics = {
        'episode': ep_num,
        'mean_demand_power': df['env_demand_power'].mean(),
        'mean_energy_balance': df['env_energy_balance'].mean(),
        'mean_renewable_power': df['env_total_renewable'].mean(),
        'mean_total_power': df['env_total_power'].mean(),
        'total_reward_solar': df['reward_solar#0'].sum(),
        'total_reward_wind': df['reward_wind#0'].sum(),
        'total_reward_battery': df['reward_battery#0'].sum(),
        'total_reward_grid': df['reward_grid#0'].sum(),
        'total_reward_load': df['reward_load#0'].sum(),
        'solar_activation_rate': (df['action_solar#0'] == 1).mean(),
        'wind_activation_rate': (df['action_wind#0'] == 1).mean(),
        'battery_charge_rate': (df['action_battery#0'] == 1).mean(),
        'battery_discharge_rate': (df['action_battery#0'] == 2).mean(),
        'grid_import_rate': (df['action_grid#0'] == 1).mean(),
        'load_reduction_rate': (df['action_load#0'] == 0).mean(),
    }
    episodes.append(metrics)

df_episodes = pd.DataFrame(episodes)
print(f"✅ {len(df_episodes)} episodios cargados\n")

# Crear visualización mejorada
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

fig.suptitle('EXPERIMENTO 3: 1000 Episodios - Análisis Completo', 
             fontsize=20, fontweight='bold', y=0.995)

# 1. Evolución de Rewards con ventanas móviles
ax1 = fig.add_subplot(gs[0, :])
window = 50
for agent, color in [('solar', 'gold'), ('wind', 'skyblue'), 
                      ('battery', 'green'), ('grid', 'red'), ('load', 'purple')]:
    rolling_mean = df_episodes[f'total_reward_{agent}'].rolling(window=window, center=True).mean()
    ax1.plot(df_episodes['episode'], rolling_mean, 
            label=agent.capitalize(), linewidth=3, alpha=0.9, color=color)
    ax1.fill_between(df_episodes['episode'], 
                     df_episodes[f'total_reward_{agent}'].rolling(window=window*2, center=True).mean() - 
                     df_episodes[f'total_reward_{agent}'].rolling(window=window*2, center=True).std(),
                     df_episodes[f'total_reward_{agent}'].rolling(window=window*2, center=True).mean() + 
                     df_episodes[f'total_reward_{agent}'].rolling(window=window*2, center=True).std(),
                     alpha=0.1, color=color)

ax1.set_xlabel('Episodio', fontsize=13)
ax1.set_ylabel('Reward Total (media móvil 50 ep)', fontsize=13)
ax1.set_title('Evolución de Rewards: Tendencia y Variabilidad', fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# 2-5. Comparación en ventanas de 100 episodios
windows_100 = [
    (0, 100, 'darkred'),
    (200, 300, 'orange'),
    (400, 500, 'yellow'),
    (600, 700, 'lightgreen'),
    (800, 900, 'green')
]

for idx, (agent, title) in enumerate([
    ('solar', 'Solar'), ('wind', 'Wind'), ('battery', 'Battery'), 
    ('grid', 'Grid'), ('load', 'Load')
]):
    ax = fig.add_subplot(gs[1, idx % 3] if idx < 3 else gs[2, idx - 3])
    
    means = []
    stds = []
    labels = []
    colors_bar = []
    
    for start, end, color in windows_100:
        window_data = df_episodes[start:end][f'total_reward_{agent}']
        means.append(window_data.mean())
        stds.append(window_data.std())
        labels.append(f'{start}-{end}')
        colors_bar.append(color)
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, alpha=0.8, capsize=5, color=colors_bar)
    ax.set_ylabel('Reward Promedio', fontsize=11)
    ax.set_title(f'{title}: Evolución por Ventanas', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Agregar valor en cada barra
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
               fontsize=8)

# 6. Tasas de activación
ax6 = fig.add_subplot(gs[3, 0])
window = 50
for agent, label, color in [('solar', 'Solar', 'gold'), ('wind', 'Wind', 'skyblue')]:
    rate_roll = df_episodes[f'{agent}_activation_rate'].rolling(window=window, center=True).mean() * 100
    ax6.plot(df_episodes['episode'], rate_roll, label=label, linewidth=2.5, color=color)

ax6.set_xlabel('Episodio', fontsize=11)
ax6.set_ylabel('Tasa de Activación (%)', fontsize=11)
ax6.set_title('Renovables: Evolución de Activación', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 100])

# 7. Batería
ax7 = fig.add_subplot(gs[3, 1])
window = 50
charge_roll = df_episodes['battery_charge_rate'].rolling(window=window, center=True).mean() * 100
discharge_roll = df_episodes['battery_discharge_rate'].rolling(window=window, center=True).mean() * 100

ax7.plot(df_episodes['episode'], charge_roll, label='Carga', linewidth=2.5, color='green')
ax7.plot(df_episodes['episode'], discharge_roll, label='Descarga', linewidth=2.5, color='orange')
ax7.set_xlabel('Episodio', fontsize=11)
ax7.set_ylabel('Tasa de Acción (%)', fontsize=11)
ax7.set_title('Batería: Estrategia de Carga/Descarga', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.set_ylim([0, 100])

# 8. Load y Grid
ax8 = fig.add_subplot(gs[3, 2])
window = 50
load_roll = df_episodes['load_reduction_rate'].rolling(window=window, center=True).mean() * 100
grid_roll = df_episodes['grid_import_rate'].rolling(window=window, center=True).mean() * 100

ax8.plot(df_episodes['episode'], load_roll, label='Load Reducción', linewidth=2.5, color='purple')
ax8.plot(df_episodes['episode'], grid_roll, label='Grid Import', linewidth=2.5, color='red')
ax8.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (aleatorio)')
ax8.set_xlabel('Episodio', fontsize=11)
ax8.set_ylabel('Tasa (%)', fontsize=11)
ax8.set_title('Load y Grid: Comportamiento', fontsize=12, fontweight='bold')
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 100])

# 9. Convergencia - Varianza
ax9 = fig.add_subplot(gs[4, :])
window = 100
for agent, color in [('solar', 'gold'), ('wind', 'skyblue'), 
                      ('battery', 'green'), ('grid', 'red'), ('load', 'purple')]:
    rolling_std = df_episodes[f'total_reward_{agent}'].rolling(window=window, center=True).std()
    ax9.plot(df_episodes['episode'], rolling_std, 
            label=agent.capitalize(), linewidth=2.5, alpha=0.8, color=color)

ax9.set_xlabel('Episodio', fontsize=13)
ax9.set_ylabel('Desviación Estándar (ventana 100)', fontsize=13)
ax9.set_title('Convergencia: Estabilidad de Rewards', fontsize=15, fontweight='bold')
ax9.legend(loc='best', fontsize=11)
ax9.grid(True, alpha=0.3)

plt.savefig('results/plots/learning_analysis_1000ep.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: results/plots/learning_analysis_1000ep.png\n")

# Estadísticas detalladas
print("="*80)
print("📊 ESTADÍSTICAS DETALLADAS - EXPERIMENTO 3 (1000 EPISODIOS)")
print("="*80)

# Validación del fix
print("\n🎯 VALIDACIÓN DEL FIX (p_load=10W):")
print(f"   Demand Power mín:  {df_episodes['mean_demand_power'].min():.2f} W")
print(f"   Demand Power máx:  {df_episodes['mean_demand_power'].max():.2f} W")
print(f"   Demand Power mean: {df_episodes['mean_demand_power'].mean():.2f} W")
if df_episodes['mean_demand_power'].min() > 0:
    print("   ✅ CORRECTO: Demand Power > 0 en TODOS los episodios")

# Comparación temporal
print("\n🏆 EVOLUCIÓN DE REWARDS:")
first_100 = df_episodes.head(100)
last_100 = df_episodes.tail(100)

print(f"{'Agente':<12} {'Primeros 100':<15} {'Últimos 100':<15} {'Mejora':<12} {'Status'}")
print("-" * 80)

for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    mean_first = first_100[f'total_reward_{agent}'].mean()
    mean_last = last_100[f'total_reward_{agent}'].mean()
    improvement = ((mean_last - mean_first) / abs(mean_first)) * 100 if mean_first != 0 else 0
    
    if improvement > 20:
        status = "✅ MEJORA"
    elif improvement > -20:
        status = "⚠️ ESTABLE"
    else:
        status = "❌ EMPEORA"
    
    print(f"{agent.capitalize():<12} {mean_first:>13.2f}  {mean_last:>13.2f}  {improvement:>10.1f}%  {status}")

# Análisis de convergencia
print("\n🎯 ANÁLISIS DE CONVERGENCIA (últimos 200 episodios):")
last_200 = df_episodes.tail(200)

for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    std_last_200 = last_200[f'total_reward_{agent}'].std()
    mean_last_200 = last_200[f'total_reward_{agent}'].mean()
    cv = (std_last_200 / abs(mean_last_200)) * 100 if mean_last_200 != 0 else 0
    
    print(f"   {agent.capitalize():<10} - Std: {std_last_200:>7.2f} | CV: {cv:>7.1f}%", end='')
    if cv < 10:
        print("  ✅✅ Excelente")
    elif cv < 30:
        print("  ✅ Convergente")
    elif cv < 50:
        print("  ⚠️ Moderado")
    else:
        print("  ❌ Inestable")

# Tasas de activación
print("\n⚡ TASAS DE ACTIVACIÓN (últimos 100 episodios):")
print(f"   Solar:           {last_100['solar_activation_rate'].mean()*100:>6.1f}%")
print(f"   Wind:            {last_100['wind_activation_rate'].mean()*100:>6.1f}%")
print(f"   Battery Carga:   {last_100['battery_charge_rate'].mean()*100:>6.1f}%")
print(f"   Battery Descarga:{last_100['battery_discharge_rate'].mean()*100:>6.1f}%")
print(f"   Grid Import:     {last_100['grid_import_rate'].mean()*100:>6.1f}%")
print(f"   Load Reducción:  {last_100['load_reduction_rate'].mean()*100:>6.1f}%")

# Balance energético
print("\n📈 BALANCE ENERGÉTICO:")
print(f"   Promedio total:      {df_episodes['mean_energy_balance'].mean():>8.2f} W")
print(f"   Primeros 100 ep:     {first_100['mean_energy_balance'].mean():>8.2f} W")
print(f"   Últimos 100 ep:      {last_100['mean_energy_balance'].mean():>8.2f} W")
print(f"   Std últimos 100:     {last_100['mean_energy_balance'].std():>8.2f} W")

print("\n" + "="*80)
print("✅ Análisis completado")
print("="*80)
