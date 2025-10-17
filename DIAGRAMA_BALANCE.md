# 🔄 Diagrama de Flujo del Sistema de Balance Energético

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INICIO DE PASO DE SIMULACIÓN                     │
│                      (Cargar demand y price del dataset)                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FASE 1: RENOVABLES (Solar + Wind)                                      │
│  ────────────────────────────────────────────────────────────────────   │
│  • Entrada: potential (del dataset)                                     │
│  • Acción: 0 = no inyectar, 1 = inyectar                               │
│  • Salida: power = action * potential                                   │
│  • Actualiza: env.renewable_power += power                              │
│                env.total_power += power                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FASE 2: LOAD (Carga)                                                   │
│  ────────────────────────────────────────────────────────────────────   │
│  • Entrada: env.demand_power (del dataset)                              │
│  • Acción: 0 = reducir carga, 1 = demanda completa                      │
│  • Salida: power = -demand_power (si action=1)                          │
│            power = -(demand_power - p_load) (si action=0)               │
│  • Actualiza: env.demand_power = abs(power)                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Balance Preliminar    │
                    │  = renewable - demand  │
                    └────────┬───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FASE 3: BATTERY (Batería)                                              │
│  ────────────────────────────────────────────────────────────────────   │
│  • Entrada: preliminary_balance, SOC actual                             │
│  • Acción: 0 = idle, 1 = charge, 2 = discharge                          │
│                                                                          │
│  Si action=1 (charge) y surplus > 0:                                    │
│     power = -min(surplus, p_charge_max, soc_capacity)                   │
│                                                                          │
│  Si action=2 (discharge) y deficit > 0:                                 │
│     power = min(deficit, p_discharge_max, soc_energy)                   │
│                                                                          │
│  Si action=0:                                                            │
│     power = 0                                                            │
│                                                                          │
│  • Actualiza: SOC, env.soc_idx                                          │
│              env.total_power o env.demand_power (según signo)           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Balance Post-Battery  │
                    │  = total - demand      │
                    └────────┬───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FASE 4: GRID (Utility Grid - Último recurso)                           │
│  ────────────────────────────────────────────────────────────────────   │
│  • Entrada: deficit_residual = demand - total                           │
│  • Acción: 0 = no importar, 1 = importar                                │
│                                                                          │
│  Si action=1 y deficit > 0:                                              │
│     potential = deficit_residual                                         │
│     power = min(potential, p_max)                                        │
│                                                                          │
│  Si action=0 o no hay deficit:                                           │
│     power = 0                                                            │
│                                                                          │
│  • Actualiza: env.total_power += power (solo si power > 0)              │
│  • NOTA: Grid NO exporta, solo importa                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CÁLCULO DE BALANCE FINAL                                               │
│  ─────────────────────────────────────────────────────────────────────  │
│  • env.energy_balance = env.total_power - env.demand_power              │
│  • env.delta_power_idx = "surplus" si >= 0 else "deficit"               │
│                                                                          │
│  ESCENARIOS:                                                             │
│  ✓ Balance = 0      → Sistema balanceado (ideal)                        │
│  ⚠  Balance > 0      → Curtailment (exceso de renovables)               │
│  ⚠  Balance < 0      → Déficit (falta energía)                          │
└─────────────────────────────────────────────────────────────────────────┘

```

## 📊 Tabla de Convenciones

| Agente    | Power Positivo        | Power Negativo       | Actions         |
|-----------|----------------------|----------------------|-----------------|
| Solar     | Genera energía       | N/A                  | 0, 1            |
| Wind      | Genera energía       | N/A                  | 0, 1            |
| Load      | N/A                  | Consume energía      | 0, 1            |
| Battery   | Descarga (genera)    | Carga (consume)      | 0, 1, 2         |
| Grid      | Importa (genera)     | N/A (no exporta)     | 0, 1            |

## 🎯 Ejemplo Numérico

```
ESCENARIO: Déficit que requiere grid + batería

Inicio:
  dataset.demand = 1000 W
  dataset.solar = 300 W
  dataset.wind = 200 W

FASE 1: RENOVABLES
  solar.action = 1 → power = 300 W
  wind.action = 1  → power = 200 W
  env.renewable_power = 500 W
  env.total_power = 500 W

FASE 2: LOAD
  load.action = 1 → power = -1000 W (demanda completa)
  env.demand_power = 1000 W

BALANCE PRELIMINAR = 500 - 1000 = -500 W (déficit)

FASE 3: BATTERY
  battery.action = 2 (discharge)
  battery.SOC = 0.5
  deficit = 500 W
  battery.power = min(500, 100) = 100 W (limitado por p_discharge_max)
  env.total_power = 500 + 100 = 600 W

BALANCE POST-BATTERY = 600 - 1000 = -400 W (déficit residual)

FASE 4: GRID
  grid.action = 1 (import)
  deficit_residual = 400 W
  grid.power = min(400, 1000) = 400 W (limitado por p_max)
  env.total_power = 600 + 400 = 1000 W

BALANCE FINAL = 1000 - 1000 = 0 W ✓ (balanceado!)
```

## 🔍 Casos Especiales

### Caso 1: Curtailment (Battery llena)
```
Renovables = 800 W
Demanda = 500 W
Battery SOC = 1.0 (llena)

→ Battery no puede cargar (clip por SOC)
→ Grid no puede exportar
→ Balance final = 800 - 500 = +300 W (curtailment)
→ Penalizaciones en rewards
```

### Caso 2: Load Shedding (Grid desconectado)
```
Renovables = 300 W
Demanda = 1000 W
Battery SOC = 0.0 (vacía)
Grid action = 0 (no importa)

→ Battery no puede descargar
→ Grid no importa
→ Load puede reducir: action=0 → demand = 1000 - 200 = 800 W
→ Balance final = 300 - 800 = -500 W (déficit)
→ Penalizaciones fuertes en rewards
```

### Caso 3: Coordinación Óptima
```
Renovables = 700 W
Demanda = 1000 W
Battery SOC = 0.6

→ Battery descarga: power = 300 W
→ Balance = 700 + 300 - 1000 = 0 W ✓
→ Grid no necesita importar
→ Recompensas altas para todos
```
