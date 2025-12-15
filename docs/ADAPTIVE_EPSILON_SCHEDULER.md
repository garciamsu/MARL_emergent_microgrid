# Adaptive Epsilon Scheduler

## Overview

El **Adaptive Epsilon Scheduler** ajusta dinámicamente la tasa de exploración (epsilon) basándose en el **comportamiento emergente del sistema multi-agente**, específicamente en el balance energético. Este enfoque permite que la exploración se adapte automáticamente al nivel de estabilidad del sistema, favoreciendo la **autoorganización** y la **resiliencia**.

## Motivación

En sistemas multi-agente complejos como microredes, el comportamiento emergente del sistema (coordinación entre agentes) es tan importante como el aprendizaje individual. Un scheduler adaptativo permite:

1. **Respuesta a inestabilidad**: Cuando el sistema está desbalanceado, aumenta la exploración para encontrar mejores estrategias de coordinación.

2. **Consolidación de comportamiento**: Cuando el sistema alcanza estabilidad, reduce la exploración para consolidar el comportamiento emergente exitoso.

3. **Resiliencia ante cambios**: Detecta automáticamente cuando el entorno cambia (aumento de inestabilidad) y responde incrementando la exploración.

4. **Autoorganización**: Los agentes aprenden a coordinarse sin intervención externa, guiados por la señal de estabilidad del sistema.

## Funcionamiento

### Índice de Estabilidad Emergente

El scheduler calcula un **índice de estabilidad** basado en el balance energético del episodio anterior:

```python
stability_index = mean(|energy_balance|) / max_reference_power
```

Donde:
- `energy_balance = total_power - demand_power`
- Valor normalizado en [0, 1]
- **0** = balance perfecto (estabilidad máxima)
- **1** = desbalance máximo (inestabilidad máxima)

### Bandas de Ajuste

El scheduler opera en tres bandas definidas por umbrales:

#### 1. Alta Inestabilidad (stability_index > `high_instability_threshold`)
- **Acción**: ↑ **AUMENTAR epsilon**
- **Razón**: El sistema está descoordinado, necesita más exploración
- **Incremento**: `epsilon += increase_rate`
- **Límite superior**: `max_eps` (típicamente 1.0)

**Ejemplo**: Si `stability_index = 0.20` y `high_instability_threshold = 0.15`
→ Epsilon aumenta de 0.30 a 0.35 (con `increase_rate = 0.05`)

#### 2. Zona Intermedia (`stability_threshold` < stability_index < `high_instability_threshold`)
- **Acción**: → **MANTENER epsilon**
- **Razón**: El sistema está en transición, mantener nivel actual de exploración
- **Cambio**: ninguno

**Ejemplo**: Si `stability_index = 0.10` (entre 0.05 y 0.15)
→ Epsilon se mantiene constante

#### 3. Alta Estabilidad (stability_index < `stability_threshold`)
- **Acción**: ↓ **REDUCIR epsilon**
- **Razón**: El sistema está bien coordinado, consolidar comportamiento
- **Decremento**: `epsilon -= decrease_rate`
- **Límite inferior**: `min_eps` (típicamente 0.15)

**Ejemplo**: Si `stability_index = 0.03` y `stability_threshold = 0.05`
→ Epsilon disminuye de 0.25 a 0.23 (con `decrease_rate = 0.02`)

## Configuración

### En `configs/default.yaml`

```yaml
simulation:
  epsilon:
    schedule: adaptive       # Activar scheduler adaptativo
    start: 1.0               # Epsilon inicial (exploración total)
    min: 0.15                # Epsilon mínimo (nunca va a 0)
    max: 1.0                 # Epsilon máximo
    
    # Umbrales de estabilidad
    high_instability_threshold: 0.15    # Límite superior para aumentar ε
    stability_threshold: 0.05           # Límite inferior para reducir ε
    
    # Tasas de ajuste
    increase_rate: 0.05      # Incremento cuando hay inestabilidad
    decrease_rate: 0.02      # Decremento cuando hay estabilidad
    
    # Ventana de análisis
    stability_window: 50     # Pasos a considerar (no implementado aún)
```

### Parámetros Explicados

| Parámetro | Rango Típico | Descripción |
|-----------|--------------|-------------|
| `high_instability_threshold` | 0.10 - 0.20 | Umbral superior: por encima, el sistema se considera inestable |
| `stability_threshold` | 0.03 - 0.08 | Umbral inferior: por debajo, el sistema se considera estable |
| `increase_rate` | 0.03 - 0.10 | Magnitud del incremento de epsilon (mayor = respuesta más agresiva) |
| `decrease_rate` | 0.01 - 0.05 | Magnitud del decremento de epsilon (menor = consolidación más gradual) |

### Ajuste de Parámetros

**Para sistemas más reactivos** (respuesta rápida a cambios):
```yaml
increase_rate: 0.08
decrease_rate: 0.03
high_instability_threshold: 0.12
```

**Para sistemas más conservadores** (cambios graduales):
```yaml
increase_rate: 0.03
decrease_rate: 0.01
high_instability_threshold: 0.18
```

**Para favorecer estabilidad temprana**:
```yaml
stability_threshold: 0.08  # Más permisivo para empezar a reducir
decrease_rate: 0.03        # Reducción más rápida
```

## Ejemplo de Evolución

### Episodio por Episodio

| Episode | Stability Index | Epsilon Before | Action | Epsilon After |
|---------|----------------|----------------|--------|---------------|
| 0 | - | 1.00 | START | 1.00 |
| 1 | 0.35 | 1.00 | ↑ INCREASE | 1.00 (max) |
| 2 | 0.28 | 1.00 | ↑ INCREASE | 1.00 (max) |
| 5 | 0.18 | 1.00 | ↑ INCREASE | 1.00 (max) |
| 10 | 0.12 | 1.00 | → MAINTAIN | 1.00 |
| 20 | 0.09 | 0.95 | → MAINTAIN | 0.95 |
| 50 | 0.04 | 0.80 | ↓ DECREASE | 0.78 |
| 100 | 0.03 | 0.65 | ↓ DECREASE | 0.63 |
| 200 | 0.02 | 0.45 | ↓ DECREASE | 0.43 |
| 500 | 0.01 | 0.25 | ↓ DECREASE | 0.23 |
| 1000 | 0.02 | 0.18 | ↓ DECREASE | 0.16 |
| 1200 | 0.01 | 0.15 | ↓ DECREASE | 0.15 (min) |

### Interpretación

1. **Episodios 0-10**: Alta inestabilidad inicial, epsilon se mantiene alto para exploración intensiva.
2. **Episodios 10-50**: Sistema empieza a estabilizarse, epsilon se mantiene para consolidar.
3. **Episodios 50-200**: Mejora progresiva, epsilon disminuye gradualmente.
4. **Episodios 200+**: Sistema estable, epsilon alcanza mínimo y se mantiene para exploración continua.

## Ventajas vs. Schedulers Tradicionales

### Scheduler Exponential (tradicional)
```yaml
schedule: exponential
decay: 0.9985
```
- ✗ Decay fijo independiente del desempeño
- ✗ No responde a cambios en el entorno
- ✗ Puede reducir exploración cuando aún hay inestabilidad
- ✓ Predecible y simple

### Scheduler Adaptive (nuevo)
```yaml
schedule: adaptive
high_instability_threshold: 0.15
stability_threshold: 0.05
```
- ✓ Responde al comportamiento emergente real
- ✓ Se adapta automáticamente a cambios
- ✓ Mantiene exploración mientras sea necesario
- ✓ Favorece autoorganización
- ✗ Más parámetros que configurar

## Monitoreo

Durante el entrenamiento, el scheduler imprime información en cada episodio:

```
   Adaptive ε: ↑ INCREASE | Stability=0.182 | ε=0.850
   Adaptive ε: → MAINTAIN | Stability=0.112 | ε=0.850
   Adaptive ε: ↓ DECREASE | Stability=0.043 | ε=0.830
```

Esta información se puede usar para:
- Verificar que el sistema responde correctamente
- Ajustar umbrales si es necesario
- Identificar episodios de alta inestabilidad

## Análisis de Resultados

### Métricas Clave

1. **Correlación Stability-Epsilon**: 
   - Esperado: correlación positiva (más inestabilidad → más exploración)

2. **Tasa de Convergencia**:
   - Medir cuántos episodios toma alcanzar epsilon mínimo

3. **Resiliencia**:
   - Verificar que epsilon aumenta cuando hay cambios súbitos

### Gráficos Recomendados

```python
import pandas as pd
import matplotlib.pyplot as plt

# Leer logs de entrenamiento
# (asumiendo que se guarden stability_index y epsilon por episodio)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Evolución de epsilon
ax1.plot(episodes, epsilon_values, label='Epsilon')
ax1.axhline(y=0.15, color='r', linestyle='--', label='Min Epsilon')
ax1.set_ylabel('Epsilon')
ax1.set_title('Adaptive Epsilon Evolution')
ax1.legend()
ax1.grid(True)

# Índice de estabilidad
ax2.plot(episodes, stability_values, label='Stability Index', color='orange')
ax2.axhline(y=0.15, color='r', linestyle='--', label='High Instability')
ax2.axhline(y=0.05, color='g', linestyle='--', label='Stability')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Stability Index')
ax2.set_title('System Stability Evolution')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('adaptive_epsilon_analysis.png')
```

## Casos de Uso

### 1. Entornos No Estacionarios
Ideal para microredes donde:
- La demanda varía estacionalmente
- La generación renovable es intermitente
- Se añaden o retiran agentes dinámicamente

### 2. Sistemas con Comportamiento Emergente Complejo
Cuando:
- La coordinación entre agentes es crítica
- El desempeño individual no garantiza desempeño global
- Se busca autoorganización sin control centralizado

### 3. Aprendizaje Continuo (Lifelong Learning)
Para sistemas que:
- Operan indefinidamente
- Deben adaptarse a nuevas condiciones
- No tienen fase de preentrenamiento clara

## Limitaciones y Consideraciones

1. **Ruido en la Señal**:
   - El balance energético puede tener variabilidad intrínseca
   - Considerar usar promedios móviles en lugar de episodio individual

2. **Horizonte de Evaluación**:
   - Un episodio puede no ser suficiente para evaluar estabilidad
   - Considerar ventanas de múltiples episodios

3. **Inicialización**:
   - El primer episodio no tiene historial para ajustar
   - Epsilon inicial debe ser suficientemente alto

4. **Interacción con Hiperparámetros**:
   - Alpha y gamma también afectan la estabilidad
   - Búsqueda de hiperparámetros debe considerar el scheduler adaptativo

## Extensiones Futuras

1. **Multi-señal**:
   - Combinar balance energético con otras señales (recompensa acumulada, entropía de acciones)

2. **Predicción**:
   - Anticipar inestabilidad futura y ajustar proactivamente

3. **Per-Agent Epsilon**:
   - Epsilon diferente para cada agente según su contribución a la inestabilidad

4. **Bandas Dinámicas**:
   - Los umbrales también se adaptan según la evolución del sistema

## Referencias

- Aprendizaje multi-agente y comportamiento emergente
- Control adaptativo en sistemas de energía
- Epsilon-greedy con annealing contextual

## Ver También

- [ONLINE_QLEARNING_CONFIG.md](ONLINE_QLEARNING_CONFIG.md) - Configuración base de Q-learning
- [configs/default.yaml](../configs/default.yaml) - Configuración completa
- [core/simulation.py](../core/simulation.py) - Implementación del scheduler
