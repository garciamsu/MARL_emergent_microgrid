# Optimizadores de Hiperparámetros - MARL Microgrid

Este directorio contiene los scripts de optimización de hiperparámetros para los 5 agentes del sistema MARL (Multi-Agent Reinforcement Learning) para gestión de microrredes.

## 📋 Requisitos

### Dependencias Python

```bash
pip install -r requirements.txt
```

**Paquetes principales:**
- `numpy`, `pandas` - Manipulación de datos
- `matplotlib` - Visualización
- `tqdm` - Barras de progreso
- `pyyaml` - Configuración
- `scikit-optimize` - Optimización bayesiana (opcional pero recomendado)

### Verificación de Instalación

Ejecuta el script de validación para verificar que todas las dependencias estén correctamente instaladas:

```bash
python test/validate_optimizers.py
```

Si todas las validaciones pasan (✓), los optimizadores están listos para usarse.

## 🚀 Scripts Disponibles

### 1. Battery Agent
```bash
python test/battery/B_battery_hyperparam_optimizer.py
```
- **Hiperparámetros:** psi, sigma, nu, beta, xi, mu (6 parámetros)
- **Acciones:** 0=descargar, 1=mantener, 2=cargar

### 2. Grid Agent
```bash
python test/grid/B_grid_hyperparam_optimizer.py
```
- **Hiperparámetros:** psi, sigma, nu, xi (4 parámetros)
- **Acciones:** 0=no comprar, 1=comprar

### 3. Load Agent
```bash
python test/load/B_load_hyperparam_optimizer.py
```
- **Hiperparámetros:** sigma, psi, nu, beta (4 parámetros)
- **Acciones:** 0=no satisfacer, 1=satisfacer

### 4. Solar Agent
```bash
python test/solar/B_solar_hyperparam_optimizer.py
```
- **Hiperparámetros:** theta, beta, eta, nu, xi (5 parámetros)
- **Acciones:** 0=no suministrar, 1=suministrar

### 5. Wind Agent
```bash
python test/wind/B_wind_hyperparam_optimizer.py
```
- **Hiperparámetros:** theta, beta, eta, nu, xi (5 parámetros)
- **Acciones:** 0=no suministrar, 1=suministrar

## 🔧 Configuración

Cada script tiene constantes configurables al inicio del archivo:

```python
# Método de optimización
OPTIMIZATION_METHOD = "random_search"  # "random_search" | "bayesian" | "evolutionary"

# Configuración de optimización
MAX_ITERATIONS = 500     # Número de iteraciones
POP_SIZE = 24           # Tamaño de población (evolutivo)
MUTATION_RATE = 0.2     # Tasa de mutación (evolutivo)
CROSSOVER_RATE = 0.7    # Tasa de cruce (evolutivo)

# Rangos de hiperparámetros (ajustables)
PARAM_BOUNDS = {
    "param1": (0.1, 5.0),
    "param2": (0.1, 5.0),
    # ...
}
```

## 📊 Métodos de Optimización

### 1. Random Search (`random_search`)
- **Descripción:** Búsqueda aleatoria uniforme en el espacio de hiperparámetros
- **Ventajas:** Simple, rápida, sin dependencias externas
- **Uso recomendado:** Exploración inicial rápida

### 2. Bayesian Optimization (`bayesian`)
- **Descripción:** Optimización bayesiana usando Gaussian Processes
- **Requisito:** `scikit-optimize` instalado
- **Ventajas:** Más eficiente, encuentra óptimos con menos evaluaciones
- **Uso recomendado:** Cuando se necesita precisión con presupuesto limitado

### 3. Evolutionary Algorithm (`evolutionary`)
- **Descripción:** Algoritmo genético simple
- **Ventajas:** Buena exploración del espacio, sin dependencias externas
- **Uso recomendado:** Espacios de búsqueda complejos con múltiples óptimos locales

## 📁 Estructura de Archivos

```
test/
├── validate_optimizers.py           # Script de validación
├── optimizer_utils.py               # Funciones comunes
├── battery/
│   ├── B_battery_hyperparam_optimizer.py
│   ├── input/
│   │   └── Battery_Agent_Reward_Table.csv
│   └── output/
│       ├── hyperparam_optimization_log.csv
│       └── fitness_convergence.svg
├── grid/
│   ├── B_grid_hyperparam_optimizer.py
│   ├── input/
│   │   └── GridAgent_Reward_Table.csv
│   └── output/
│       ├── hyperparam_optimization_log.csv
│       └── fitness_convergence.svg
├── [solar, wind, load con estructura similar]
```

## 📈 Outputs

Cada ejecución genera:

### 1. Log CSV (`hyperparam_optimization_log.csv`)
```csv
timestamp,param1,param2,...,margin_total
2025-11-03T10:30:00Z,1.234,2.456,...,150.23
2025-11-03T10:30:01Z,1.567,2.789,...,152.45
...
```

### 2. Gráfico de Convergencia (`fitness_convergence.svg`)
- Visualización de la evolución del margen total
- Formato SVG (escalable, alta calidad)

## 🎯 Criterio de Optimización

Los optimizadores maximizan el **margen total**, definido como:

```
Margen = Σ (Recompensa_Acción_Correcta - max(Recompensa_Acciones_Incorrectas))
```

Para cada estado único en el dataset.

### Acción Correcta

Las reglas de acción correcta están definidas en `test/optimizer_utils.py`:

- **Battery:** Basado en SOC y balance energético
- **Grid:** Comprar cuando hay déficit y batería baja
- **Load:** Satisfacer si hay suficiente energía y confort adecuado
- **Solar/Wind:** Suministrar cuando hay potencial y demanda

## 🔍 Ejemplo de Uso Completo

```bash
# 1. Validar instalación
python test/validate_optimizers.py

# 2. Ejecutar optimización bayesiana para Battery (editar OPTIMIZATION_METHOD en el script)
python test/battery/B_battery_hyperparam_optimizer.py

# 3. Revisar resultados
cat test/battery/output/hyperparam_optimization_log.csv | tail -n 10
```

## 📝 Notas Importantes

1. **Datos de Entrada:** Los CSV de entrada deben estar en `/test/<agent>/input/`
2. **Backup Automático:** Si ya existe un log previo, se respalda automáticamente con timestamp
3. **Semilla:** La semilla se lee de `configs/default.yaml` para reproducibilidad
4. **Tiempo de Ejecución:** Depende del método y número de iteraciones:
   - Random: ~1-5 min
   - Bayesian: ~5-15 min
   - Evolutionary: ~3-10 min

## 🐛 Solución de Problemas

### Error: `No module named 'skopt'`
```bash
pip install scikit-optimize
```

### Error: CSV no encontrado
Verificar que existan los archivos:
- `test/battery/input/Battery_Agent_Reward_Table.csv`
- `test/grid/input/GridAgent_Reward_Table.csv`
- etc.

### Proceso muy lento
1. Reducir `MAX_ITERATIONS`
2. Usar método `random_search` en lugar de `bayesian`
3. Verificar tamaño del dataset de entrada

## 📚 Referencias

- **Proyecto:** MARL Emergent Microgrid
- **Arquitectura:** Ver `/docs/` para documentación completa
- **Configuración:** `configs/default.yaml`
