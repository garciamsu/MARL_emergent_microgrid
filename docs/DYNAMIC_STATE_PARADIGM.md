# Paradigma de Estado Dinámico con Consumo Estigmérgico

## Resumen

Este documento describe el cambio arquitectónico implementado para soportar el **paradigma de estado dinámico** en el sistema MARL de microred. El cambio principal es que el `delta_ph` (variable estigmérgica) ahora se actualiza dinámicamente conforme cada agente consume su porción del potencial renovable.

## Motivación

### Problema Anterior
En la implementación original:
1. Todos los agentes observaban el mismo `delta_ph` al inicio del paso
2. El `delta_ph` se calculaba una sola vez como `renewable_potential - demand`
3. Los agentes posteriores no "veían" que los anteriores ya habían consumido parte del potencial

### Solución Implementada
El nuevo flujo implementa **consumo estigmérgico**:
1. El potencial renovable disminuye conforme cada agente inyecta potencia
2. Cada agente observa el `delta_ph` actualizado al momento de decidir
3. Se mantiene la semántica original: `delta_ph = renewable_potential - demand`

## Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PASO t - ESTADO DINÁMICO                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FASE 0: CARGAR DATOS (env.load_timestep_data)                     │
│     └── demand, price, solar_potential, wind_potential              │
│     └── renewable_potential = solar_pot + wind_pot                  │
│     └── delta_ph = renewable_potential - demand                     │
│                                                                     │
│  FASE 1: SOLAR                                                      │
│     1. Observa estado con delta_ph actual                           │
│     2. Elige acción (epsilon-greedy)                                │
│     3. Ejecuta y acumula potencia                                   │
│     4. Consume potencial: renewable_potential -= solar.power        │
│     5. Recalcula delta_ph                                           │
│                                                                     │
│  FASE 2: WIND                                                       │
│     1. Observa estado con delta_ph reducido (post-solar)            │
│     2. Elige acción                                                 │
│     3. Ejecuta y acumula potencia                                   │
│     4. Consume potencial: renewable_potential -= wind.power         │
│     5. Recalcula delta_ph                                           │
│                                                                     │
│  FASE 3: BATTERY                                                    │
│     1. Observa delta_ph (potencial restante - demanda)              │
│     2. Elige acción (idle/charge/discharge)                         │
│     3. Ejecuta                                                      │
│                                                                     │
│  FASE 4: GRID                                                       │
│     1. Observa estado actual                                        │
│     2. Elige acción (import/idle)                                   │
│     3. Ejecuta - cubre déficit residual                             │
│                                                                     │
│  FASE 5: LOAD                                                       │
│     1. Observa estado actual                                        │
│     2. Elige acción (full demand/shed)                              │
│     3. Ejecuta - ajusta demanda                                     │
│                                                                     │
│  CÁLCULO DE RECOMPENSAS                                             │
│     └── Basadas en el estado que CADA agente observó al decidir     │
│                                                                     │
│  NEXT_STATE                                                         │
│     └── Usa datos del timestep t+1 + SOC actualizado                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Cambios en Archivos

### 1. `core/environment.py`

**Nuevos métodos añadidos:**

#### `load_timestep_data(index: int)`
Carga los datos base del dataset para el timestep actual. DEBE llamarse al inicio de cada paso, ANTES de que cualquier agente observe o actúe.

```python
def load_timestep_data(self, index: int) -> None:
    """Load base data from dataset for the current timestep."""
    # Carga: demand, price, solar_potential, wind_potential
    # Inicializa renewable_potential y calcula delta_ph inicial
```

#### `update_delta_ph()`
Recalcula `delta_ph` basándose en el `renewable_potential` y `demand_power` actuales.

```python
def update_delta_ph(self) -> None:
    """Recalculate delta_ph based on current renewable_potential and demand."""
    self.delta_ph = self.renewable_potential - self.demand_power
    self.delta_ph_norm = self.delta_ph / self.max_value
    self.delta_ph_idx = discretize_ternary(self.delta_ph_norm, threshold=0.01)
```

#### `consume_renewable_potential(power_consumed: float, source: str)`
Reduce el potencial renovable después de que un agente consume su porción.

```python
def consume_renewable_potential(self, power_consumed: float, source: str) -> None:
    """Reduce renewable_potential after an agent consumes its portion."""
    self.renewable_potential -= power_consumed
    self.renewable_potential = max(0, self.renewable_potential)
    self.update_delta_ph()  # Recalcula delta_ph
```

### 2. `agents/base_agent.py`

**Modificación en `get_discretized_state()`:**
- Ya no carga datos del dataset (eso lo hace `env.load_timestep_data`)
- Lee directamente las variables del environment que ya fueron actualizadas
- Simplificado y más eficiente

### 3. `core/simulation.py`

**Cambios en el bucle principal:**
- Añadida FASE 0: `env.load_timestep_data(index)` al inicio de cada paso
- Fusión de observe-decide-execute por fase de agente
- Llamada a `env.consume_renewable_potential()` después de renovables
- Nuevo cálculo de `next_state` usando `env.load_timestep_data(index + 1)`

**Nuevas columnas en CSV de evolución:**
- `env_delta_ph_initial`: delta_ph al inicio del paso (antes de cualquier acción)
- `env_delta_ph_final`: delta_ph al final del paso (después de consumo)

## Semántica del Estado

### Para Agentes Renovables (Solar, Wind)
El `delta_ph` que observan representa:
- **Solar (primero)**: `(solar_pot + wind_pot) - demand` → Potencial completo
- **Wind (segundo)**: `(potencial_restante) - demand` → Potencial menos lo que solar ya inyectó

### Para Batería
El `delta_ph` representa el potencial restante después de que los renovables actuaron:
- Positivo → Hay excedente potencial → Oportunidad de carga
- Negativo → Hay déficit → Oportunidad de descarga

### Para Grid y Load
Ven el estado final del sistema, actuando como respaldo.

## Compatibilidad

### Hacia Atrás
- Los CSVs de evolución mantienen todas las columnas anteriores
- Se añaden `env_delta_ph_initial` y `env_delta_ph_final` (renombrado de `env_delta_ph`)
- Las funciones de recompensa siguen usando `state_tuple` que incluye `delta_ph_idx`

### Scripts de Análisis
Los scripts en `analysis/` deberían funcionar sin cambios. Si alguno usaba `env_delta_ph`, ahora puede usar:
- `env_delta_ph_initial` para el valor pre-acciones
- `env_delta_ph_final` para el valor post-acciones
- `env_real_balance` para el balance real de potencia (renovable - demanda)

## Señales Diferenciadas por Tipo de Agente

### Problema Detectado
Tras implementar el consumo estigmérgico, se identificó una discrepancia crítica:
- **delta_ph (estigmérgico)**: Calculado como `renewable_potential - demand` → Se vuelve NEGATIVO después de que los renovables "consumen" su potencial
- **real_balance**: Calculado como `renewable_power - demand` → Refleja el balance REAL de potencia inyectada

**Ejemplo del problema:**
```
Step 0:
  delta_ph_initial = +86,623 W (SURPLUS de potencial)
  Wind inyecta 108,453 W
  Después de consume_renewable_potential():
    delta_ph_final = -21,830 W (DEFICIT de potencial restante)
  
  Pero la realidad física:
    real_balance = 108,453 - 76,214 = +32,239 W (SURPLUS real)
```

La batería veía `delta_ph = -21,830` (deficit) pero el balance real era `+32,239` (surplus). Esto causaba que recibiera **penalización** por cargar correctamente.

### Solución: Diferenciación por Tipo de Agente

Se implementó el concepto de **señales diferenciadas**:

| Tipo de Agente | Señal para Recompensa | Justificación |
|----------------|----------------------|---------------|
| **Solar, Wind** | `delta_ph` (estigmérgico) | Coordinan acceso al potencial renovable |
| **Battery, Grid** | `real_balance` (potencia real) | Responden al balance físico real |

### Nuevas Variables en Environment

```python
# Señal estigmérgica (para renovables)
env.delta_ph = env.renewable_potential - env.demand_power
env.delta_ph_norm = delta_ph / max_value
env.delta_ph_idx = discretize_ternary(delta_ph_norm)

# Balance real (para batería/grid)
env.real_balance = env.renewable_power - env.demand_power
env.real_balance_norm = real_balance / max_value
env.real_balance_idx = discretize_ternary(real_balance_norm)
```

### Cambios en Funciones de Recompensa

**DefaultBatteryReward** y **DefaultGridReward** ahora usan `real_balance_idx`:
```python
def compute(self, agent, env, state_tuple):
    # Usa balance REAL en lugar de delta_ph estigmérgico
    real_balance_idx = getattr(env, 'real_balance_idx', 0)
    
    if real_balance_idx > 0 and agent.action == 1:  # Surplus → carga correcta
        reward = +self.psi * (1 - soc_norm)
    # ...
```

### Columnas Adicionales en CSV

- `env_real_balance`: Balance real en Watts
- `env_real_balance_norm`: Balance normalizado [-1, 1]
- `env_real_balance_idx`: Índice discretizado {-1, 0, +1}

### Verificación

Para verificar que la batería ahora recibe recompensa correcta:
```python
# Cuando hay surplus real Y la batería carga:
assert real_balance_idx > 0 and battery.action == 1
# → reward debe ser POSITIVO
```

## Ejemplo de Ejecución

```
Timestep 5:
  Datos cargados: demand=100kW, solar_pot=80kW, wind_pot=40kW
  renewable_potential = 120kW
  delta_ph_inicial = 120 - 100 = +20kW (surplus)

  SOLAR observa delta_ph=+20kW (surplus) → decide inyectar → power=80kW
  renewable_potential = 120 - 80 = 40kW
  delta_ph = 40 - 100 = -60kW (deficit)

  WIND observa delta_ph=-60kW (deficit) → decide no inyectar → power=0kW
  renewable_potential = 40 - 0 = 40kW
  delta_ph = 40 - 100 = -60kW (deficit)

  BATTERY observa delta_ph=-60kW → decide descargar
  ...
```

## Fundamento Teórico

Este diseño implementa un **juego secuencial de Stackelberg** donde:
1. El orden de ejecución está predefinido (Solar → Wind → Battery → Grid → Load)
2. Los agentes posteriores tienen ventaja informacional (ven efectos de anteriores)
3. El equilibrio resultante es un **Subgame Perfect Equilibrium (SPE)**

La variable `delta_ph` actúa como **señal estigmérgica**:
- Es modificada por las acciones de los agentes
- Es observada por agentes posteriores
- Facilita coordinación indirecta sin comunicación explícita

## Referencias Internas

- [copilot-instructions.md](.github/copilot-instructions.md): Reglas generales del proyecto
- [STABILITY_ANALYSIS.md](docs/STABILITY_ANALYSIS.md): Análisis de estabilidad MARL
- [core/rewards.py](core/rewards.py): Funciones de recompensa (usan delta_ph_idx del estado)
