"""
Script para analizar específicamente la fila 15 del CSV y validar los cálculos.
"""

import pandas as pd

# Leer CSV (auto-detectar separador)
df = pd.read_csv('results/evolution/episode_0.csv')

print("\n" + "="*80)
print("ANÁLISIS DE FILA 15 (Step 14) - VALIDACIÓN DE CÁLCULOS")
print("="*80)

# Obtener fila 15 (index 14 porque es 0-based)
if len(df) < 15:
    print(f"\n❌ ERROR: El CSV solo tiene {len(df)} filas")
    exit(1)

row = df.iloc[14]  # Fila 15 es índice 14

print(f"\n📊 DATOS DE LA FILA 15 (Índice {14}):")
print("="*80)

# Solar
print(f"\n🌞 SOLAR:")
print(f"   Potential: {row['potential_solar#0']:.2f} W")
print(f"   Action:    {int(row['action_solar#0'])}")
print(f"   Power:     {row['power_solar#0']:.2f} W")

# Wind
print(f"\n💨 WIND:")
print(f"   Potential: {row['potential_wind#0']:.2f} W")
print(f"   Action:    {int(row['action_wind#0'])}")
print(f"   Power:     {row['power_wind#0']:.2f} W")

# Battery
print(f"\n🔋 BATTERY:")
print(f"   Potential: {row['potential_battery#0']:.2f} W")
print(f"   Action:    {int(row['action_battery#0'])}")
print(f"   Power:     {row['power_battery#0']:.2f} W")
print(f"   SOC:       {row['soc_battery#0']:.4f}")

# Grid
print(f"\n🔌 GRID:")
print(f"   Potential: {row['potential_grid#0']:.2f} W")
print(f"   Action:    {int(row['action_grid#0'])}")
print(f"   Power:     {row['power_grid#0']:.2f} W")

# Load
print(f"\n🏠 LOAD:")
print(f"   Potential: {row['potential_load#0']:.2f} W")
print(f"   Action:    {int(row['action_load#0'])}")
print(f"   Power:     {row['power_load#0']:.2f} W")

# Environment totals
print(f"\n" + "="*80)
print(f"🌍 TOTALES DEL ENTORNO:")
print("="*80)
print(f"   Renewable Power: {row['env_total_renewable']:.2f} W")
print(f"   Total Power:     {row['env_total_power']:.2f} W")
print(f"   Demand Power:    {row['env_demand_power']:.2f} W")
print(f"   Energy Balance:  {row['env_energy_balance']:.2f} W")
print(f"   Status:          {row['env_delta_power_idx']}")

# Validación manual de cálculos
print(f"\n" + "="*80)
print(f"🔍 VALIDACIÓN DE CÁLCULOS:")
print("="*80)

# Calcular generación esperada
solar_gen = row['power_solar#0']
wind_gen = row['power_wind#0']
battery_contrib = row['power_battery#0']
grid_import = row['power_grid#0']

expected_renewable = solar_gen + wind_gen
expected_total_gen = solar_gen + wind_gen + max(0, battery_contrib) + grid_import

print(f"\n1️⃣ GENERACIÓN RENOVABLE:")
print(f"   Solar + Wind = {solar_gen:.2f} + {wind_gen:.2f} = {expected_renewable:.2f} W")
print(f"   CSV indica:    {row['env_total_renewable']:.2f} W")
if abs(expected_renewable - row['env_total_renewable']) < 0.01:
    print(f"   ✅ CORRECTO")
else:
    print(f"   ❌ ERROR: Diferencia de {abs(expected_renewable - row['env_total_renewable']):.2f} W")

# Calcular consumo esperado
load_consumption = abs(row['power_load#0'])
battery_charging = abs(min(0, battery_contrib))

expected_total_demand = load_consumption + battery_charging

print(f"\n2️⃣ CONSUMO TOTAL:")
print(f"   Load = {load_consumption:.2f} W (abs de {row['power_load#0']:.2f})")
print(f"   Battery charging = {battery_charging:.2f} W")
print(f"   Total esperado = {load_consumption:.2f} + {battery_charging:.2f} = {expected_total_demand:.2f} W")
print(f"   CSV indica:    {row['env_demand_power']:.2f} W")
if abs(expected_total_demand - row['env_demand_power']) < 0.01:
    print(f"   ✅ CORRECTO")
else:
    print(f"   ❌ ERROR: Diferencia de {abs(expected_total_demand - row['env_demand_power']):.2f} W")

print(f"\n3️⃣ GENERACIÓN TOTAL:")
print(f"   Renewable = {expected_renewable:.2f} W")
print(f"   Battery discharge = {max(0, battery_contrib):.2f} W")
print(f"   Grid import = {grid_import:.2f} W")
print(f"   Total esperado = {expected_total_gen:.2f} W")
print(f"   CSV indica:    {row['env_total_power']:.2f} W")
if abs(expected_total_gen - row['env_total_power']) < 0.01:
    print(f"   ✅ CORRECTO")
else:
    print(f"   ❌ ERROR: Diferencia de {abs(expected_total_gen - row['env_total_power']):.2f} W")

# Balance energético
expected_balance = row['env_total_power'] - row['env_demand_power']

print(f"\n4️⃣ BALANCE ENERGÉTICO:")
print(f"   Total Power - Demand Power = {row['env_total_power']:.2f} - {row['env_demand_power']:.2f}")
print(f"   Balance esperado = {expected_balance:.2f} W")
print(f"   CSV indica:      {row['env_energy_balance']:.2f} W")
if abs(expected_balance - row['env_energy_balance']) < 0.01:
    print(f"   ✅ CORRECTO")
else:
    print(f"   ❌ ERROR: Diferencia de {abs(expected_balance - row['env_energy_balance']):.2f} W")

# Análisis del problema
print(f"\n" + "="*80)
print(f"🚨 ANÁLISIS DEL PROBLEMA:")
print("="*80)

if row['env_demand_power'] == 0:
    print(f"\n❌ PROBLEMA DETECTADO:")
    print(f"   env_demand_power = 0 W, pero load.power = {row['power_load#0']:.2f} W")
    print(f"\n💡 POSIBLE CAUSA:")
    print(f"   - La demanda NO se está acumulando en env.demand_power")
    print(f"   - Revisar que load.action={int(row['action_load#0'])} esté funcionando")
    print(f"   - Verificar que env.demand_power se resetee y acumule correctamente")
    
    # Ver fila anterior para contexto
    if len(df) > 1:
        prev_row = df.iloc[13]
        print(f"\n📋 COMPARACIÓN CON FILA ANTERIOR (Índice {13}):")
        print(f"   Prev demand_power: {prev_row['env_demand_power']:.2f} W")
        print(f"   Curr demand_power: {row['env_demand_power']:.2f} W")
        print(f"   Prev load.power:   {prev_row['power_load#0']:.2f} W")
        print(f"   Curr load.power:   {row['power_load#0']:.2f} W")

print(f"\n" + "="*80)
