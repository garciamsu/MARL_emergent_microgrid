# Análisis del Bug en `env_delta_ph`

## Problema Detectado

El valor `env_delta_ph` registrado en `results/evolution/episode_0.csv` **NO corresponde** a la fórmula esperada:

```
delta_ph = (potential_solar + potential_wind) - demand
```

## Valores Observados (Step 0)

```
potential_solar#0:       0.00 W
potential_wind#0:        96,735.89 W
power_solar#0:           0.00 W (agente decidió NO generar)
power_wind#0:            0.00 W (agente decidió NO generar)
env_demand_power:        168,725.51 W
env_delta_ph:           -84,362.76 W ← INCORRECTO
```

## Análisis de la Fórmula Actual

La fórmula implementada en `agents/base_agent.py:119` es:

```python
env.delta_ph = env.renewable_power - env.demand_power
```

Donde:
- `env.renewable_power` = **potencia REAL generada** (power_solar + power_wind)
- `env.demand_power` = demanda efectiva después de las actualizaciones de agentes

## Valores Esperados vs Reales

### Opción 1: Delta basado en POTENCIA REAL

Si `delta_ph` representa el balance de **potencia real generada**:

```
delta_ph = renewable_power - demand_power
         = (0.0 + 0.0) - 168,725.51
         = -168,725.51 W
```

**Valor esperado**: -168,725.51 W
**Valor real en CSV**: -84,362.76 W
**Diferencia**: Factor de 2 → **BUG CONFIRMADO**

### Opción 2: Delta basado en POTENCIAL disponible

Si `delta_ph` representa el balance de **potencial disponible**:

```
delta_ph = renewable_potential - demand_power
         = (0.0 + 96,735.89) - 168,725.51
         = -71,989.62 W
```

**Valor esperado**: -71,989.62 W
**Valor real en CSV**: -84,362.76 W
**Diferencia**: No coincide → Tampoco es esta interpretación

## Causa del Bug

El problema está en el **flujo de actualización de `env.demand_power`** en `core/simulation.py`:

### Paso 1: Inicialización (línea 352)
```python
env.get_dataset("demand", index)
```

Esto ejecuta en `environment.py:176`:
```python
self.demand_power = row[field]  # demand_power = 168,725.51 W
self.base_demand = row[field]   # base_demand = 168,725.51 W
```

### Paso 2: Actualización del Load Agent (línea 403)
```python
if "load" in agent.name.lower():
    agent.update_power(env)
    env.demand_power += abs(agent.power)  # ← PROBLEMA AQUÍ
```

En `load_agent.py:47-48`:
```python
if self.action == 1:  # Load ON
    self.power = -base_demand  # power = -168,725.51 W
```

Entonces:
```python
env.demand_power += abs(agent.power)
                 = 168,725.51 + 168,725.51
                 = 337,451.02 W  # ¡DUPLICADO!
```

### Paso 3: Cálculo de delta_ph (base_agent.py:119)
```python
env.delta_ph = env.renewable_power - env.demand_power
             = 0.0 - 337,451.02
             = -337,451.02 W
```

**PERO** en el CSV aparece **-84,362.76 W**, que es aproximadamente **1/4** del valor real.

## Hipótesis Adicional

El valor **-84,362.76 W** es exactamente **la mitad de 168,725.51 W**.

Esto sugiere que en algún momento del flujo hay una **corrección parcial** o un **cálculo intermedio** que no se está propagando correctamente.

Posibles causas:
1. `env.demand_power` se está dividiendo por 2 en algún punto no visible
2. El `load_agent` está generando `power = -demand/2` en lugar de `-demand`
3. Hay una actualización adicional de `env.demand_power` que no se está registrando

## Solución Propuesta

### Opción A: Usar potencia real generada (implementación actual corregida)

```python
# En simulation.py, línea 323 (dentro del loop de steps)
env.demand_power = 0.0  # ← AÑADIR ESTA LÍNEA

# Luego, los agentes acumulan sus contribuciones:
# - Battery cargando: env.demand_power += abs(agent.power)
# - Load consumiendo: env.demand_power += abs(agent.power)
```

Y en `base_agent.py:119`:
```python
env.delta_ph = env.renewable_power - env.demand_power
```

### Opción B: Usar potencial disponible del dataset

Cambiar la fórmula en `base_agent.py:119`:
```python
env.delta_ph = env.renewable_potential - env.demand_power
```

Donde `env.renewable_potential` se calcula como:
```python
env.renewable_potential = potential_solar + potential_wind
```

## Recomendación

**Se recomienda Opción A** porque:

1. Es consistente con la arquitectura MARL: el balance debe reflejar las **decisiones** de los agentes
2. Los agentes aprenden sobre el impacto de sus **acciones reales**, no de potenciales teóricos
3. El potencial disponible ya está registrado en `env_renewable_potential`

## Acción Inmediata Requerida

1. **Añadir** `env.demand_power = 0.0` en la línea 323 de `simulation.py`
2. **Verificar** que el load agent NO esté sumando incorrectamente a `env.demand_power`
3. **Ejecutar test** con los cambios y verificar que:
   ```
   env_delta_ph == env_total_renewable - env_demand_power
   ```

## Estado Actual

- ❌ **Bug confirmado**: `env_delta_ph` no corresponde a ninguna fórmula esperada
- ❌ **Causa identificada**: Doble acumulación de `env.demand_power`
- ❌ **Impacto**: Las recompensas basadas en `delta_ph` están recibiendo valores incorrectos
- ⚠️ **Urgencia**: Alta - Afecta al aprendizaje de todos los agentes que usan `delta_ph` en su estado

---

**Fecha**: 11 de diciembre de 2025
**Autor**: GitHub Copilot
**Versión**: 1.0
