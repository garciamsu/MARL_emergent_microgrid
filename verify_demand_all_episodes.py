#!/usr/bin/env python3
"""
Análisis detallado: Verificación de que demand_power > 0 en todas las filas
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("🔍 VERIFICACIÓN: Demand Power > 0 en TODAS las filas")
print("="*70)

results_dir = Path('results/evolution')
episodes_with_zero = []
total_rows = 0
rows_with_zero = 0

for ep_file in sorted(results_dir.glob('episode_*.csv')):
    ep_num = int(ep_file.stem.split('_')[1])
    df = pd.read_csv(ep_file)
    
    total_rows += len(df)
    
    # Verificar si hay filas con demand_power = 0
    zero_rows = df[df['env_demand_power'] == 0]
    
    if len(zero_rows) > 0:
        rows_with_zero += len(zero_rows)
        episodes_with_zero.append({
            'episode': ep_num,
            'rows_with_zero': len(zero_rows),
            'steps': zero_rows['step'].tolist()
        })

print(f"\n📊 Total de filas analizadas: {total_rows}")
print(f"📊 Filas con demand_power = 0: {rows_with_zero}")

if rows_with_zero == 0:
    print(f"\n✅ ¡PERFECTO! NO hay filas con demand_power = 0")
    print(f"✅ El fix de p_load funcionó correctamente en TODOS los casos")
else:
    print(f"\n⚠️  Encontradas {rows_with_zero} filas con demand_power = 0")
    print(f"    En {len(episodes_with_zero)} episodios diferentes:")
    for ep in episodes_with_zero[:5]:  # Mostrar primeros 5
        print(f"    - Episodio {ep['episode']}: {ep['rows_with_zero']} filas en steps {ep['steps']}")

# Análisis estadístico de demand_power
print(f"\n📈 ESTADÍSTICAS DE DEMAND_POWER:")

all_demand = []
for ep_file in sorted(results_dir.glob('episode_*.csv')):
    df = pd.read_csv(ep_file)
    all_demand.extend(df['env_demand_power'].tolist())

all_demand = np.array(all_demand)

print(f"   Mínimo:     {all_demand.min():.2f} W")
print(f"   Máximo:     {all_demand.max():.2f} W")
print(f"   Media:      {all_demand.mean():.2f} W")
print(f"   Mediana:    {np.median(all_demand):.2f} W")
print(f"   Std Dev:    {all_demand.std():.2f} W")

# Verificar acciones del load agent
print(f"\n🏠 ANÁLISIS DEL LOAD AGENT:")

all_load_actions = []
all_load_power = []
all_load_potential = []

for ep_file in sorted(results_dir.glob('episode_*.csv')):
    df = pd.read_csv(ep_file)
    all_load_actions.extend(df['action_load#0'].tolist())
    all_load_power.extend(df['power_load#0'].tolist())
    all_load_potential.extend(df['potential_load#0'].tolist())

all_load_actions = np.array(all_load_actions)
all_load_power = np.array(all_load_power)
all_load_potential = np.array(all_load_potential)

action_0_count = (all_load_actions == 0).sum()
action_1_count = (all_load_actions == 1).sum()

print(f"   Action 0 (reducir): {action_0_count} veces ({action_0_count/len(all_load_actions)*100:.1f}%)")
print(f"   Action 1 (completo): {action_1_count} veces ({action_1_count/len(all_load_actions)*100:.1f}%)")

# Analizar power cuando action=0
action_0_mask = all_load_actions == 0
power_at_action_0 = all_load_power[action_0_mask]
potential_at_action_0 = all_load_potential[action_0_mask]

print(f"\n   Cuando action=0 (reducir):")
print(f"      Power min:      {power_at_action_0.min():.2f} W")
print(f"      Power max:      {power_at_action_0.max():.2f} W")
print(f"      Power mean:     {power_at_action_0.mean():.2f} W")
print(f"      Potential mean: {potential_at_action_0.mean():.2f} W")

# Verificar que power = -(potential - p_load) cuando action=0
expected_power = -(potential_at_action_0 - 10)  # p_load = 10
difference = np.abs(power_at_action_0 - expected_power)

print(f"\n   Verificación: power = -(potential - 10)")
print(f"      Diferencia media: {difference.mean():.6f} W")
if difference.mean() < 0.01:
    print(f"      ✅ Cálculo CORRECTO")
else:
    print(f"      ❌ Cálculo INCORRECTO")

# Analizar power cuando action=1
action_1_mask = all_load_actions == 1
power_at_action_1 = all_load_power[action_1_mask]
potential_at_action_1 = all_load_potential[action_1_mask]

print(f"\n   Cuando action=1 (completo):")
print(f"      Power min:      {power_at_action_1.min():.2f} W")
print(f"      Power max:      {power_at_action_1.max():.2f} W")
print(f"      Power mean:     {power_at_action_1.mean():.2f} W")
print(f"      Potential mean: {potential_at_action_1.mean():.2f} W")

# Verificar que power = -potential cuando action=1
expected_power_1 = -potential_at_action_1
difference_1 = np.abs(power_at_action_1 - expected_power_1)

print(f"\n   Verificación: power = -potential")
print(f"      Diferencia media: {difference_1.mean():.6f} W")
if difference_1.mean() < 0.01:
    print(f"      ✅ Cálculo CORRECTO")
else:
    print(f"      ❌ Cálculo INCORRECTO")

print("\n" + "="*70)
print("✅ Verificación completada")
print("="*70)
