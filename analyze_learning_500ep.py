#!/usr/bin/env python3
"""
Análisis completo de 500 episodios con comparación vs 100 episodios
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*70)
print("📊 ANÁLISIS: 500 EPISODIOS CON PARÁMETROS MEJORADOS")
print("="*70)

# Leer todos los episodios
results_dir = Path('results/evolution')
episodes = []

print("\n⏳ Leyendo episodios...")
for ep_file in sorted(results_dir.glob('episode_*.csv')):
    ep_num = int(ep_file.stem.split('_')[1])
    df = pd.read_csv(ep_file)
    
    # Calcular métricas por episodio
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
print(f"✅ {len(df_episodes)} episodios cargados")

# Crear visualizaciones mejoradas
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Análisis Completo - 500 Episodios (Parámetros Mejorados)', 
             fontsize=18, fontweight='bold', y=0.995)

# 1. Evolución de Rewards Totales con ventanas móviles
ax1 = fig.add_subplot(gs[0, :2])
window = 20
for agent, color in [('solar', 'gold'), ('wind', 'skyblue'), 
                      ('battery', 'green'), ('grid', 'red'), ('load', 'purple')]:
    rolling_mean = df_episodes[f'total_reward_{agent}'].rolling(window=window, center=True).mean()
    ax1.plot(df_episodes['episode'], rolling_mean, 
            label=agent.capitalize(), linewidth=2.5, alpha=0.9, color=color)
    ax1.plot(df_episodes['episode'], df_episodes[f'total_reward_{agent}'], 
            alpha=0.2, linewidth=0.5, color=color)

ax1.set_xlabel('Episodio', fontsize=12)
ax1.set_ylabel('Reward Total (media móvil 20 ep)', fontsize=12)
ax1.set_title('Evolución de Rewards con Tendencia', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# 2. Comparación: Primeros 50 vs Últimos 50
ax2 = fig.add_subplot(gs[0, 2])
first_50 = df_episodes.head(50)
last_50 = df_episodes.tail(50)

agents = ['solar', 'wind', 'battery', 'grid', 'load']
x = np.arange(len(agents))
width = 0.35

means_first = [first_50[f'total_reward_{a}'].mean() for a in agents]
means_last = [last_50[f'total_reward_{a}'].mean() for a in agents]

ax2.bar(x - width/2, means_first, width, label='Primeros 50', alpha=0.8, color='lightcoral')
ax2.bar(x + width/2, means_last, width, label='Últimos 50', alpha=0.8, color='lightgreen')

ax2.set_ylabel('Reward Promedio', fontsize=11)
ax2.set_title('Comparación: Inicio vs Final', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([a.capitalize() for a in agents], rotation=45, ha='right')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# 3. Demand Power - Validación crítica
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df_episodes['episode'], df_episodes['mean_demand_power'], 
        linewidth=1.5, color='red', alpha=0.7)
ax3.axhline(y=df_episodes['mean_demand_power'].mean(), 
           color='orange', linestyle='--', linewidth=2,
           label=f"Media: {df_episodes['mean_demand_power'].mean():.2f}W")
ax3.axhline(y=10, color='green', linestyle=':', linewidth=2, 
           label='Mínimo esperado: 10W')
ax3.fill_between(df_episodes['episode'], 0, df_episodes['mean_demand_power'], 
                 alpha=0.2, color='red')
ax3.set_xlabel('Episodio', fontsize=11)
ax3.set_ylabel('Demand Power Promedio (W)', fontsize=11)
ax3.set_title('✅ Validación: Demand Power > 0', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Tasas de Activación - Renovables
ax4 = fig.add_subplot(gs[1, 1])
window = 20
solar_roll = df_episodes['solar_activation_rate'].rolling(window=window, center=True).mean() * 100
wind_roll = df_episodes['wind_activation_rate'].rolling(window=window, center=True).mean() * 100

ax4.plot(df_episodes['episode'], solar_roll, label='Solar', linewidth=2.5, color='gold')
ax4.plot(df_episodes['episode'], wind_roll, label='Wind', linewidth=2.5, color='skyblue')
ax4.fill_between(df_episodes['episode'], solar_roll, alpha=0.3, color='gold')
ax4.fill_between(df_episodes['episode'], wind_roll, alpha=0.3, color='skyblue')
ax4.set_xlabel('Episodio', fontsize=11)
ax4.set_ylabel('Tasa de Activación (%) - Media móvil', fontsize=11)
ax4.set_title('Aprendizaje de Renovables', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 100])

# 5. Comportamiento de Batería
ax5 = fig.add_subplot(gs[1, 2])
window = 20
charge_roll = df_episodes['battery_charge_rate'].rolling(window=window, center=True).mean() * 100
discharge_roll = df_episodes['battery_discharge_rate'].rolling(window=window, center=True).mean() * 100

ax5.plot(df_episodes['episode'], charge_roll, 
        label='Carga (action=1)', linewidth=2.5, color='green')
ax5.plot(df_episodes['episode'], discharge_roll, 
        label='Descarga (action=2)', linewidth=2.5, color='orange')
ax5.fill_between(df_episodes['episode'], charge_roll, alpha=0.3, color='green')
ax5.fill_between(df_episodes['episode'], discharge_roll, alpha=0.3, color='orange')
ax5.set_xlabel('Episodio', fontsize=11)
ax5.set_ylabel('Tasa de Acción (%) - Media móvil', fontsize=11)
ax5.set_title('Coordinación de Batería', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 100])

# 6. Load - Reducción de Demanda
ax6 = fig.add_subplot(gs[2, 0])
window = 20
load_roll = df_episodes['load_reduction_rate'].rolling(window=window, center=True).mean() * 100

ax6.plot(df_episodes['episode'], load_roll, linewidth=2.5, color='purple')
ax6.fill_between(df_episodes['episode'], load_roll, alpha=0.3, color='purple')
ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (aleatorio)')
ax6.set_xlabel('Episodio', fontsize=11)
ax6.set_ylabel('Tasa de Reducción (%) - Media móvil', fontsize=11)
ax6.set_title('Load: Aprendizaje de Reducción', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, 100])

# 7. Balance Energético
ax7 = fig.add_subplot(gs[2, 1])
window = 20
balance_roll = df_episodes['mean_energy_balance'].rolling(window=window, center=True).mean()

ax7.plot(df_episodes['episode'], balance_roll, linewidth=2.5, color='teal')
ax7.fill_between(df_episodes['episode'], 0, balance_roll, 
                 where=(balance_roll >= 0), alpha=0.3, color='green', label='Surplus')
ax7.fill_between(df_episodes['episode'], 0, balance_roll, 
                 where=(balance_roll < 0), alpha=0.3, color='red', label='Deficit')
ax7.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax7.set_xlabel('Episodio', fontsize=11)
ax7.set_ylabel('Balance Promedio (W) - Media móvil', fontsize=11)
ax7.set_title('Balance Energético', fontsize=12, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Grid Import Rate
ax8 = fig.add_subplot(gs[2, 2])
window = 20
grid_roll = df_episodes['grid_import_rate'].rolling(window=window, center=True).mean() * 100

ax8.plot(df_episodes['episode'], grid_roll, linewidth=2.5, color='red')
ax8.fill_between(df_episodes['episode'], grid_roll, alpha=0.3, color='red')
ax8.set_xlabel('Episodio', fontsize=11)
ax8.set_ylabel('Tasa de Importación (%) - Media móvil', fontsize=11)
ax8.set_title('Grid: Uso de Red Externa', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_ylim([0, 100])

# 9. Convergencia - Varianza de Rewards
ax9 = fig.add_subplot(gs[3, :])
window = 50
for agent, color in [('solar', 'gold'), ('wind', 'skyblue'), 
                      ('battery', 'green'), ('grid', 'red'), ('load', 'purple')]:
    rolling_std = df_episodes[f'total_reward_{agent}'].rolling(window=window, center=True).std()
    ax9.plot(df_episodes['episode'], rolling_std, 
            label=agent.capitalize(), linewidth=2, alpha=0.8, color=color)

ax9.set_xlabel('Episodio', fontsize=12)
ax9.set_ylabel('Desviación Estándar de Rewards (ventana 50)', fontsize=12)
ax9.set_title('Convergencia: Estabilidad de Políticas', fontsize=14, fontweight='bold')
ax9.legend(loc='best', fontsize=10)
ax9.grid(True, alpha=0.3)

plt.savefig('results/plots/learning_analysis_500ep.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: results/plots/learning_analysis_500ep.png")

# Estadísticas detalladas
print("\n" + "="*70)
print("📊 ESTADÍSTICAS DETALLADAS - 500 EPISODIOS")
print("="*70)

# Validación del fix
print("\n🎯 VALIDACIÓN DEL FIX (p_load=10W):")
print(f"   Demand Power mín:  {df_episodes['mean_demand_power'].min():.2f} W")
print(f"   Demand Power máx:  {df_episodes['mean_demand_power'].max():.2f} W")
print(f"   Demand Power mean: {df_episodes['mean_demand_power'].mean():.2f} W")
if df_episodes['mean_demand_power'].min() > 0:
    print("   ✅ CORRECTO: Demand Power > 0 en TODOS los episodios")
else:
    print("   ❌ ERROR: Hay episodios con demand_power = 0")

# Comparación temporal
print("\n🏆 EVOLUCIÓN DE REWARDS:")
print(f"{'Agente':<12} {'Primeros 50':<14} {'Últimos 50':<14} {'Mejora':<12} {'Status'}")
print("-" * 70)

for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    mean_first = first_50[f'total_reward_{agent}'].mean()
    mean_last = last_50[f'total_reward_{agent}'].mean()
    improvement = ((mean_last - mean_first) / abs(mean_first)) * 100 if mean_first != 0 else 0
    
    if improvement > 10:
        status = "✅ MEJORA"
    elif improvement > -10:
        status = "⚠️ ESTABLE"
    else:
        status = "❌ EMPEORA"
    
    print(f"{agent.capitalize():<12} {mean_first:>12.2f}  {mean_last:>12.2f}  {improvement:>10.1f}%  {status}")

# Tasas de activación
print("\n⚡ TASAS DE ACTIVACIÓN (últimos 50 episodios):")
print(f"   Solar:           {last_50['solar_activation_rate'].mean()*100:>6.1f}%")
print(f"   Wind:            {last_50['wind_activation_rate'].mean()*100:>6.1f}%")
print(f"   Battery Carga:   {last_50['battery_charge_rate'].mean()*100:>6.1f}%")
print(f"   Battery Descarga:{last_50['battery_discharge_rate'].mean()*100:>6.1f}%")
print(f"   Grid Import:     {last_50['grid_import_rate'].mean()*100:>6.1f}%")
print(f"   Load Reducción:  {last_50['load_reduction_rate'].mean()*100:>6.1f}%")

# Balance energético
print("\n📈 BALANCE ENERGÉTICO:")
print(f"   Promedio total:      {df_episodes['mean_energy_balance'].mean():>8.2f} W")
print(f"   Primeros 50 ep:      {first_50['mean_energy_balance'].mean():>8.2f} W")
print(f"   Últimos 50 ep:       {last_50['mean_energy_balance'].mean():>8.2f} W")
print(f"   Std últimos 50:      {last_50['mean_energy_balance'].std():>8.2f} W")

# Convergencia
print("\n🎯 ANÁLISIS DE CONVERGENCIA:")
last_100 = df_episodes.tail(100)
for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    std_last_100 = last_100[f'total_reward_{agent}'].std()
    mean_last_100 = last_100[f'total_reward_{agent}'].mean()
    cv = (std_last_100 / abs(mean_last_100)) * 100 if mean_last_100 != 0 else 0
    
    print(f"   {agent.capitalize():<10} - Std: {std_last_100:>6.2f} | CV: {cv:>6.1f}%", end='')
    if cv < 20:
        print("  ✅ Convergente")
    elif cv < 50:
        print("  ⚠️ Moderado")
    else:
        print("  ❌ Inestable")

print("\n" + "="*70)
print("✅ Análisis completado")
print("="*70)
