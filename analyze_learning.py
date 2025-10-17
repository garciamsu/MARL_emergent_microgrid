#!/usr/bin/env python3
"""
Análisis del aprendizaje emergente después del fix de p_load
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Leer todos los episodios
results_dir = Path('results/evolution')
episodes = []

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

# Crear visualizaciones
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Análisis de Aprendizaje - 100 Episodios (post-fix p_load)', fontsize=16, fontweight='bold')

# 1. Evolución de Rewards Totales
ax = axes[0, 0]
for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    ax.plot(df_episodes['episode'], df_episodes[f'total_reward_{agent}'], 
            label=agent.capitalize(), linewidth=2, alpha=0.8)
ax.set_xlabel('Episodio')
ax.set_ylabel('Reward Total')
ax.set_title('Evolución de Rewards por Agente')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Demand Power Promedio
ax = axes[0, 1]
ax.plot(df_episodes['episode'], df_episodes['mean_demand_power'], 
        linewidth=2, color='red', label='Demand Power')
ax.axhline(y=df_episodes['mean_demand_power'].mean(), 
           color='orange', linestyle='--', label=f"Media: {df_episodes['mean_demand_power'].mean():.2f}W")
ax.set_xlabel('Episodio')
ax.set_ylabel('Demand Power Promedio (W)')
ax.set_title('Validación: Demand Power > 0')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Tasas de Activación de Renovables
ax = axes[1, 0]
ax.plot(df_episodes['episode'], df_episodes['solar_activation_rate'] * 100, 
        label='Solar', linewidth=2, color='gold')
ax.plot(df_episodes['episode'], df_episodes['wind_activation_rate'] * 100, 
        label='Wind', linewidth=2, color='skyblue')
ax.set_xlabel('Episodio')
ax.set_ylabel('Tasa de Activación (%)')
ax.set_title('Aprendizaje de Renovables: ¿Aprenden a Activarse?')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Comportamiento de Batería
ax = axes[1, 1]
ax.plot(df_episodes['episode'], df_episodes['battery_charge_rate'] * 100, 
        label='Charge (action=1)', linewidth=2, color='green')
ax.plot(df_episodes['episode'], df_episodes['battery_discharge_rate'] * 100, 
        label='Discharge (action=2)', linewidth=2, color='orange')
ax.set_xlabel('Episodio')
ax.set_ylabel('Tasa de Acción (%)')
ax.set_title('Coordinación de Batería')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Reducción de Carga
ax = axes[2, 0]
ax.plot(df_episodes['episode'], df_episodes['load_reduction_rate'] * 100, 
        linewidth=2, color='purple')
ax.set_xlabel('Episodio')
ax.set_ylabel('Tasa de Reducción (%)')
ax.set_title('Load: ¿Aprende a Reducir Demanda? (action=0)')
ax.grid(True, alpha=0.3)

# 6. Balance Energético
ax = axes[2, 1]
ax.plot(df_episodes['episode'], df_episodes['mean_energy_balance'], 
        linewidth=2, color='teal')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Episodio')
ax.set_ylabel('Balance Energético Promedio (W)')
ax.set_title('Balance: Surplus vs Deficit')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/learning_analysis_100ep.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: results/plots/learning_analysis_100ep.png")

# Estadísticas finales
print("\n" + "="*60)
print("📊 ESTADÍSTICAS DE APRENDIZAJE (100 EPISODIOS)")
print("="*60)

print("\n🎯 VALIDACIÓN DEL FIX:")
print(f"   Demand Power min:  {df_episodes['mean_demand_power'].min():.2f} W")
print(f"   Demand Power max:  {df_episodes['mean_demand_power'].max():.2f} W")
print(f"   Demand Power mean: {df_episodes['mean_demand_power'].mean():.2f} W")
if df_episodes['mean_demand_power'].min() > 0:
    print("   ✅ CORRECTO: Demand Power > 0 en todos los episodios")
else:
    print("   ❌ ERROR: Demand Power = 0 en algún episodio")

print("\n🏆 REWARDS TOTALES (primeros 10 vs últimos 10):")
first_10 = df_episodes.head(10)
last_10 = df_episodes.tail(10)

for agent in ['solar', 'wind', 'battery', 'grid', 'load']:
    mean_first = first_10[f'total_reward_{agent}'].mean()
    mean_last = last_10[f'total_reward_{agent}'].mean()
    improvement = ((mean_last - mean_first) / abs(mean_first)) * 100 if mean_first != 0 else 0
    
    print(f"   {agent.capitalize():8s}: {mean_first:8.2f} → {mean_last:8.2f} "
          f"({improvement:+.1f}%)")

print("\n⚡ TASAS DE ACTIVACIÓN (últimos 10 episodios):")
print(f"   Solar:      {last_10['solar_activation_rate'].mean()*100:.1f}%")
print(f"   Wind:       {last_10['wind_activation_rate'].mean()*100:.1f}%")
print(f"   Battery Ch: {last_10['battery_charge_rate'].mean()*100:.1f}%")
print(f"   Battery Dc: {last_10['battery_discharge_rate'].mean()*100:.1f}%")
print(f"   Grid:       {last_10['grid_import_rate'].mean()*100:.1f}%")
print(f"   Load Red:   {last_10['load_reduction_rate'].mean()*100:.1f}%")

print("\n📈 BALANCE ENERGÉTICO:")
print(f"   Promedio total:     {df_episodes['mean_energy_balance'].mean():.2f} W")
print(f"   Primeros 10 ep:     {first_10['mean_energy_balance'].mean():.2f} W")
print(f"   Últimos 10 ep:      {last_10['mean_energy_balance'].mean():.2f} W")

print("\n" + "="*60)
print("✅ Análisis completado")
print("="*60)
