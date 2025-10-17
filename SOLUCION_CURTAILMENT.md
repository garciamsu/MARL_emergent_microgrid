# Solución al Problema de Curtailment y Balance Energético

## 📋 Resumen de la Solución

Se ha implementado un sistema de actualización secuencial en cascada para emular un ecosistema energético de microgrid realista pero simple, que maneja:

- ✅ **Inyecciones simultáneas** de múltiples fuentes (solar, eólica, batería, utility grid)
- ✅ **Gestión de demanda** (load) con cargas controlables
- ✅ **Curtailment** (vertido de energía) cuando hay excedentes
- ✅ **Balance energético** con penalizaciones mediante rewards

---

## 🔄 Orden de Actualización Implementado

### **FASE 1: RENOVABLES** (Solar + Wind)
- **Orden:** Primera fase
- **Lógica:** `power = action * potential`
- **Características:**
  - `potential` viene del dataset histórico
  - `action = 1` → inyecta toda la potencia disponible
  - `action = 0` → no inyecta nada
  - No se ajustan físicamente por curtailment (excedentes se registran pero no modifican power)

### **FASE 2: LOAD** (Carga)
- **Orden:** Segunda fase (después de renovables)
- **Lógica:** 
  ```python
  if action == 1:
      power = -demand_base  # Consumo completo
  else:  # action == 0
      power = -(demand_base - p_load)  # Reduce carga controlable
  ```
- **Características:**
  - `action = 1` → demanda completa (carga base + controlable)
  - `action = 0` → desconecta carga controlable (solo carga base crítica)
  - `p_load` define cuánta carga es controlable (ej: 200W)

### **FASE 3: BATTERY** (Batería)
- **Orden:** Tercera fase (después de renovables y load)
- **Lógica:**
  ```python
  preliminary_balance = renewable_power - demand_power
  
  if action == 1:  # Charge
      surplus = max(0, preliminary_balance)
      power = -min(surplus, p_charge_max, soc_capacity_limit)
      
  elif action == 2:  # Discharge
      deficit = abs(min(0, preliminary_balance))
      power = min(deficit, p_discharge_max, soc_energy_available)
      
  else:  # action == 0 (idle)
      power = 0
  ```
- **Características:**
  - Reacciona al balance preliminar (renovables - demanda)
  - Clip por potencia máxima: `p_charge_max`, `p_discharge_max`
  - Clip por SOC: no puede cargar si SOC=1.0, no puede descargar si SOC=0.0
  - `action = 0` → idle
  - `action = 1` → carga (cuando hay surplus)
  - `action = 2` → descarga (cuando hay déficit)

### **FASE 4: GRID** (Utility Grid)
- **Orden:** Cuarta fase (último recurso)
- **Lógica:**
  ```python
  current_deficit = demand_power - total_power
  
  if action == 1:
      potential = max(0, current_deficit)
      power = min(potential, p_max)
  else:  # action == 0
      power = 0
  ```
- **Características:**
  - Solo puede **importar** (no exporta)
  - Cubre el déficit residual después de renovables, load y battery
  - Limitado por `p_max` (capacidad máxima de importación)
  - `action = 0` → no importa (permite desbalances)
  - `action = 1` → importa hasta cubrir déficit

---

## ⚡ Convenciones de Signos

- **Generación (positivo):** Solar, Wind, Battery discharge, Grid import
- **Consumo (negativo):** Load, Battery charge

---

## 📊 Variables de Balance en el Entorno

```python
env.renewable_power     # Generación renovable total (solar + wind)
env.demand_power        # Consumo total (load + battery charge)
env.total_power         # Generación total (renewables + battery discharge + grid)
env.energy_balance      # Balance final = total_power - demand_power
env.delta_power_idx     # "surplus" o "deficit"
```

---

## 🎯 Manejo de Escenarios Críticos

### **Escenario 1: Curtailment (Excedente de energía)**
- **Condición:** `renewable_power > demand_power` y battery llena (SOC=1.0)
- **Resultado:**
  - Battery no puede cargar → `action=1` pero `power=0` (clip por SOC)
  - Grid no puede exportar → `action=0`, `power=0`
  - `env.energy_balance > 0` (surplus)
  - Los agentes reciben **penalizaciones** en sus rewards por desperdicio
  - **No se ajusta físicamente** el power de las renovables

### **Escenario 2: Déficit sin Grid**
- **Condición:** `demand_power > total_power` y `grid.action=0`
- **Resultado:**
  - Grid no importa aunque haya déficit
  - `env.energy_balance < 0` (deficit)
  - Todos los agentes reciben **penalizaciones** en sus rewards
  - **No se fuerza** al grid a importar (no hay override)
  - **No se desconecta** carga automáticamente (load_shedding manual vía `load.action=0`)

### **Escenario 3: Balance Perfecto**
- **Condición:** `total_power ≈ demand_power`
- **Resultado:**
  - `env.energy_balance ≈ 0`
  - Los agentes reciben **recompensas** por balancear el sistema
  - Sistema operando óptimamente

---

## 🔧 Archivos Modificados

### 1. **`core/simulation.py`**
```python
# PHASE 1: Renewables
for agent in [solar, wind]:
    agent.update_power(env)
    env.renewable_power += agent.power
    env.total_power += agent.power

# PHASE 2: Load
for agent in [load]:
    agent.update_power(env)
    env.demand_power += abs(agent.power)

# PHASE 3: Battery
for agent in [battery]:
    agent.update_power(env)
    if agent.power >= 0:
        env.total_power += agent.power
    else:
        env.demand_power += abs(agent.power)

# PHASE 4: Grid
for agent in [grid]:
    agent.update_power(env)
    if agent.power > 0:
        env.total_power += agent.power
```

### 2. **`agents/load_agent.py`**
- Agregado: `p_load` (carga controlable)
- Modificado: `update_power()` con lógica de demand response

### 3. **`agents/battery_agent.py`**
- Modificado: `update_power()` con balance preliminar
- Agregado: Clips por SOC y potencia máxima
- Cálculo de `potential` basado en surplus/deficit

### 4. **`agents/grid_agent.py`**
- Agregado: `p_max`, `export_allowed`
- Modificado: `update_power()` como último recurso
- Cálculo de `potential` como déficit residual

### 5. **`core/environment.py`**
- Agregado: `dt_h` (paso de tiempo para cálculo de SOC)

---

## ✅ Verificación

El script `test_energy_balance.py` valida que:

1. ✓ El orden de actualización es correcto
2. ✓ Los cálculos de `power` y `potential` son consistentes
3. ✓ El balance energético se mantiene
4. ✓ Los clips por límites físicos funcionan

**Resultado de la prueba:**
```
Balance final: 0.00 W (surplus)
✓ Balance perfecto (error < 0.01 W)
```

---

## 🎓 Aprendizaje de Agentes

Los agentes aprenden mediante **Q-learning tabular** a:

- **Renovables:** Maximizar inyección cuando hay demanda, minimizar curtailment
- **Load:** Reducir carga cuando hay déficit crítico y precios altos
- **Battery:** Cargar en surplus, descargar en déficit, gestionar SOC óptimamente
- **Grid:** Importar solo cuando es necesario, minimizar costos de importación

**Crítico:** La solución **NO interfiere** con las estrategias de control (actions). Solo define el ecosistema físico donde los agentes operan y aprenden.

---

## 📝 Notas Importantes

1. **No hay ajustes forzados:** Si los agentes toman malas decisiones (ej: grid no importa en déficit), el sistema registra el desbalance pero no lo corrige automáticamente.

2. **Curtailment pasivo:** El excedente de renovables se desperdicia, no se ajusta la generación activamente.

3. **Restricciones físicas:** Los clips por SOC, potencia máxima, etc. son **hard constraints** que el agente no puede violar.

4. **Penalizaciones en rewards:** Los desbalances se reflejan en los rewards de cada agente, incentivando coordinación emergente.

---

## 🚀 Próximos Pasos (Sugerencias)

1. **Validación con datos reales:** Ejecutar simulaciones completas con el dataset real
2. **Métricas de curtailment:** Agregar tracking de energía desperdiciada
3. **Análisis de coordinación:** Estudiar cómo emergen patrones de cooperación entre agentes
4. **Visualización:** Crear gráficas de flujos energéticos por fase

---

**Fecha:** 17 de octubre de 2025  
**Versión:** 1.0  
**Branch:** refactor/estructura
