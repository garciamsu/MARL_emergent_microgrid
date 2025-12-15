# 📊 Analysis Tools - Flujo Manual de Análisis

Herramientas para pruebas incrementales manuales orientadas a afinar funciones de recompensas e hiperparámetros de agentes MARL en el sistema de microgrid.

---

## 🎯 Objetivo

Proporcionar un flujo de análisis **manual** y **secuencial** que permite:

1. **Validar** datasets y columnas antes de ejecutar entrenamientos.
2. **Ejecutar** entrenamientos controlados desde `configs/default.yaml`.
3. **Consolidar** resultados de múltiples episodios.
4. **Calcular** métricas operativas (MEAN, ISE, IAE, Variability, Penetraciones) y de aprendizaje (Cumulative Rewards).
5. **Visualizar** evolución y comparativas para evaluar el desempeño.

---

## 📂 Estructura del Directorio

```
analysis_tools/
├── README.md                      # 📖 Esta guía
├── __init__.py                    # Paquete Python
├── utils.py                       # Utilidades: carga de CSVs, discretización
├── metrics.py                     # Cálculo de métricas (MEAN, ISE, IAE, penetraciones, rewards)
├── plotting.py                    # Generación de gráficos (series, barras, histogramas, boxplots)
│
├── run_full_pipeline.py           # 🚀 Pipeline completo automatizado
│
├── A_data_check.py                # 🔍 Validar datasets y columnas
├── B_run_training.py              # 🏋️ Ejecutar entrenamiento (wrapper de main.py)
├── C_collect_episodes.py          # 📦 Consolidar episodios en un CSV agregado
├── D_compute_metrics.py           # 📊 Calcular métricas con PASS/FAIL
├── E_accumulated_reward.py        # 📈 Análisis completo de rewards + validación + visualización
├── E_graph_episode.py             # 📉 Generar visualizaciones clave
│
├── stability_analysis.py          # 🔬 Analizadores de estabilidad MARL (Bellman + Consenso)
├── collect_qtables_per_episode.py # 📸 Recolección de Q-tables por episodio
└── run_stability_analysis.py      # 📊 Ejecutar análisis de estabilidad completo
```

---

## 🔄 Flujo de Ejecución Manual

### **Paso 1: Validar Datos** (`A_data_check.py`)

Verifica que el dataset contenga las columnas necesarias y que los episodios previos (si existen) tengan la estructura correcta.

```bash
python analysis_tools/A_data_check.py
```

**Salida esperada:**
- ✅ Dataset válido con columnas requeridas: `['demand', 'solar', 'wind']`
- ✅ Episodios válidos con columnas requeridas (si ya existen).
- ⚠️ Sin episodios para validar si no se ha ejecutado entrenamiento.

---

### **Paso 2: Ejecutar Entrenamiento** (`B_run_training.py`)

Ejecuta `main.py` con la configuración definida en `configs/default.yaml`. La limpieza de `results/` está delegada a `main.py` (usa `analysis_tools.utils.clear_directories`).

```bash
python analysis_tools/B_run_training.py
```

**Parámetros clave (leídos de `configs/default.yaml`):**
- `simulation.episodes`: Número de episodios.
- `simulation.dataset`: Nombre del dataset (sin extensión `.csv`).
- `simulation.seed`: Semilla aleatoria.
- `simulation.epsilon`: Configuración de exploración (schedule, start, end, decay, min).
- `agents.<tipo>.reward`: Funciones de recompensa por agente.

**Política de resultados:**
- **Ephemeral**: cada ejecución de `B_run_training.py` borra los resultados previos automáticamente (vía `main.py`).
- Para preservar una corrida, copia manualmente `results/evolution/` y `results/logs/` a otro lugar antes de la siguiente ejecución.

**Salida esperada:**
- CSVs de episodios en `results/evolution/episode_<n>.csv`.
- Logs de agentes en `results/logs/run_*.log`.
- Archivo `results/logs/episode_rewards.csv` con rewards por episodio.

---

**Nota:** La validación de rewards ahora está integrada en `E_accumulated_reward.py` (Paso 5), que automáticamente verifica la corrección de los datos antes del análisis.
- ✅ Comparación de rewards almacenados vs. suma manual de timesteps
- ✅ Detección de acumulación incorrecta entre episodios
- ✅ Verificación de que valores fluctúan (no son monótonos)

**Salida esperada:**
- ✅ VERIFICATION PASSED: Todos los episodios verificados coinciden
- 📊 Estadísticas de rewards por agente
- 🔍 Verificación de acumulación incorrecta

**Nota:** Este script es **crítico** para validar que la implementación de rewards sigue el patrón correcto:
```python
episode_reward = 0.0  # Reset al inicio de cada episodio
for step in episode:
    episode_reward += timestep_reward  # Acumular dentro del episodio
episode_rewards.append(episode_reward)  # Guardar al final
```

**Referencia:** Los rewards por episodio están implementados correctamente en `metrics.py` y validados automáticamente por `E_accumulated_reward.py`.

---

### **Paso 3: Consolidar Episodios** (`C_collect_episodes.py`)

Agrupa todos los episodios en un único CSV con métricas por episodio (recompensas totales, tasas de activación, etc.).

```bash
python analysis_tools/C_collect_episodes.py
```

**Salida:**
- `results/evolution/episodes_consolidated.csv` con columnas:
  - `episode`, `mean_demand_power`, `mean_energy_balance`, `mean_renewable_power`, `mean_total_power`
  - `total_reward_<agent>` (solar, wind, battery, grid, load)
  - Tasas de activación: `solar_activation_rate`, `wind_activation_rate`, etc.

---

### **Paso 4: Calcular Métricas** (`D_compute_metrics.py`)

Calcula métricas operativas y de aprendizaje para todos los episodios e imprime resumen en consola con evaluación PASS/FAIL según umbrales configurados.

```bash
python analysis_tools/D_compute_metrics.py
```

**Métricas Operativas (por episodio):**
- **MEAN**: Promedio de `env_energy_balance` (supply − demand).
- **ISE**: Integral del Error Cuadrático: `sum(env_energy_balance²)`.
- **IAE**: Integral del Error Absoluto: `sum(|env_energy_balance|)`.
- **Variability**: Desviación estándar de `env_energy_balance`.

**Penetraciones:**
- **Renewable_Penetration**: `sum(env_total_renewable) / sum(env_demand_power)`.
- **Grid_Penetration**: `sum(max(power_grid#0, 0)) / sum(env_demand_power)`.

**Aprendizaje:**
- **Cumulative_Reward**: Suma de rewards por agente y total por episodio.

**Umbrales PASS/FAIL:**
Configurables en `configs/default.yaml` bajo `analysis.thresholds`:

```yaml
analysis:
  thresholds:
    IAE: 10000
    ISE: 50000
    Variability: 500
    Renewable_Penetration_min: 0.3
    Grid_Penetration_max: 0.7
    Cumulative_Reward_min: -5000
```

**Salida esperada:**
- Resumen en consola con valores y estado PASS/FAIL.
- Recompensas promedio por agente.

---

### **Paso 5: Gráficos de Reward Acumulado** (`E_accumulated_reward.py`)

Genera análisis de recompensas acumuladas a través de los episodios para visualizar tendencias de aprendizaje.

```bash
python analysis_tools/E_accumulated_reward.py
```

**Funcionalidad:**
- Lee `episode_rewards.csv` (generado por el entrenamiento)
- Calcula suma acumulativa ENTRE episodios (para análisis de tendencias)
- Genera estadísticas y visualizaciones

**Archivos generados en `results/plots/`:**

1. **`agent_rewards_cumulative.xlsx`**: Excel con 3 hojas:
   - `Episode_Rewards`: Rewards por episodio (cada episodio independiente)
   - `Cumulative_Sum`: Suma acumulativa entre episodios (para tendencia)
   - `Statistics`: Estadísticas descriptivas

2. **`cumulative_rewards.svg`**: Gráfico de suma acumulativa
   - Muestra tendencia de aprendizaje a lo largo del entrenamiento
   - Running sum de episode rewards (NO confundir con rewards por episodio)
   - Útil para visualizar si los agentes están mejorando globalmente

**Análisis impreso:**
- 📊 Estadísticas por agente (mean, std, min, max por episodio)
- 📈 Progreso de aprendizaje (primeros 10% vs últimos 10%)
- 💹 Mejora porcentual y absoluta

**Nota importante:** 
- La suma acumulativa es solo para **visualización de tendencias**
- Los episode rewards en sí NO son acumulativos (se resetean cada episodio)
- Este gráfico complementa las visualizaciones de rewards que muestran las tendencias de aprendizaje

---

### **Paso 6: Generar Visualizaciones** (`E_plot_metrics.py`)

Crea gráficos clave en formato SVG para análisis visual.

```bash
python analysis_tools/E_plot_metrics.py
```

**Gráficos generados en `results/plots/`:**

1. **`penetrations.svg`**: Serie temporal de penetraciones renovable y de red.
2. **`operational_metrics.svg`**: Barras de métricas operativas (MEAN, IAE, ISE, Variability) promediadas sobre últimos 10 episodios.
3. **`energy_balance_histogram.svg`**: Histograma del balance energético (último episodio).
4. **`balance_first_episode.svg`**: Serie temporal del balance en el primer episodio.
5. **`balance_last_episode.svg`**: Serie temporal del balance en el último episodio.

---

**Nota:** Todas las visualizaciones de rewards (episodio, promedio móvil, acumulativo) están integradas en `E_accumulated_reward.py` (Paso 5).

---

## 📐 Convenciones de Signos y Balance Energético

### **Columnas clave en CSVs de episodios:**

- **`env_energy_balance`**: Balance neto (supply − demand). **Objetivo: ≈ 0**.
  - `supply = solar + wind + grid_import + battery_discharge`
  - `demand = env_demand_power` (siempre positivo)

- **Generación (positiva):**
  - `power_solar#0`, `power_wind#0` ≥ 0
  - `power_battery#0` > 0 cuando descarga
  - `power_grid#0` > 0 cuando importa

- **Consumo (negativo en agentes, positivo en entorno):**
  - `power_load#0` < 0 (consumo del agente)
  - `env_demand_power` > 0 (acumulador del entorno)

- **Exportaciones:**
  - `power_grid#0` < 0 cuando exporta a la red.

### **Orden de actualización en `simulation.py`:**

1. **Renovables** (solar, wind) → generan `env.renewable_power`.
2. **Carga** (load) → actualiza `env.demand_power`.
3. **Batería** → usa balance preliminar (renovables − demanda).
4. **Red** (grid) → ajusta balance final.

---

## 🛠️ Modificar Configuración

Para ajustar experimentos, edita **`configs/default.yaml`**:

```yaml
simulation:
  episodes: 100           # Número de episodios
  dataset: "Case1"        # Dataset a usar (sin .csv)
  seed: 42                # Semilla aleatoria
  dt_h: 1.0               # Delta temporal (fijo a 1.0)
  epsilon:
    schedule: "linear"    # linear|exponential|constant|custom
    start: 0.9
    end: 0.1
    decay: 0.995          # Para exponential
    min: 0.01
```

**Recompensas por agente** (ejemplo para batería):

```yaml
agents:
  battery:
    reward:
      w_balance: 1.0
      w_soc: 0.5
      w_overcharge: -10.0
```

**Nota:** NO uses flags CLI; toda la configuración es declarativa en YAML.

---

## 📊 Ejemplo de Sesión Manual Completa

```bash
# 1. Validar datos
python analysis_tools/A_data_check.py

# 2. Editar configs/default.yaml según el experimento
nano configs/default.yaml  # Ajustar episodes, epsilon, rewards, etc.

# 3. Ejecutar entrenamiento
python analysis_tools/B_run_training.py

# 4. Consolidar episodios
python analysis_tools/C_collect_episodes.py

# 5. Calcular métricas y revisar PASS/FAIL
python analysis_tools/D_compute_metrics.py

# 6. Generar gráficos
python analysis_tools/E_plot_metrics.py

# 7. Revisar plots en results/plots/*.svg
```

---

## 🚀 Pipeline Automatizado

Para ejecutar todos los pasos de forma secuencial y automática:

```bash
python analysis_tools/run_full_pipeline.py
```

**Secuencia de ejecución:**
1. ✅ A_data_check.py - Validación de dataset
2. ✅ B_run_training.py - Entrenamiento
3. ✅ C_collect_episodes.py - Consolidación
4. ✅ D_compute_metrics.py - Métricas
5. ✅ E_accumulated_reward.py - Análisis de rewards & visualización
6. ✅ E_graph_episode.py - Gráficos por episodio

**Características:**
- ⚠️ El pipeline se detiene si algún paso falla
- 📊 Muestra resumen de pasos completados
- 📁 Indica ubicación de todos los resultados generados

---

## 📚 Archivo de Referencia

### **Scripts de Soporte Eliminados**

- `episode_reward_reference.py`: Tutorial eliminado - la lógica correcta está en `metrics.py`
- `verify_episode_rewards.py`: Integrado en `E_accumulated_reward.py` para validación automática
- `plot_episode_rewards.py`: Integrado en `E_accumulated_reward.py` para visualización completa

Este archivo es una **guía de referencia y documentación**, NO se ejecuta en el pipeline.

**Propósito:**
- Demuestra la forma **CORRECTA** de calcular episode rewards
- Documenta patrones **INCORRECTOS** a evitar
- Proporciona ejemplos de implementación

**Principios clave documentados:**
```python
# ✅ CORRECTO
episode_rewards = []
for episode in range(num_episodes):
    episode_reward = 0.0  # Reset al inicio
    for step in range(max_steps):
        timestep_reward = compute_reward(...)
        episode_reward += timestep_reward  # Acumular dentro del episodio
    episode_rewards.append(episode_reward)  # Guardar al final

# ❌ INCORRECTO
global_cumulative = 0.0  # NO hacer esto
for episode in range(num_episodes):
    for step in range(max_steps):
        global_cumulative += reward  # Acumula entre episodios ❌
```

**Uso:**
- Consultar cuando se implementan nuevas funciones de reward
- Referencia para entender el patrón correcto
- Documentación para nuevos desarrolladores

---

## 🧪 Tests y Validación

```bash
# Smoke test general
python scripts/self_check.py

# Tests unitarios
pytest -q
```

---

## 🔍 Notas Técnicas

### **Métricas ISE, IAE e Importancia**

- **ISE** (Integral del Error Cuadrático): penaliza errores grandes cuadráticamente, útil para detectar desequilibrios severos.
- **IAE** (Integral del Error Absoluto): suma total de desviaciones, más robusto ante outliers que ISE.
- **Variability** (desviación estándar): mide estabilidad del balance; valores bajos indican control consistente.

### **Penetraciones y Objetivos**

- **Renewable_Penetration ↑**: maximizar uso de renovables.
- **Grid_Penetration ↓**: minimizar dependencia de la red.

### **Curtailment**

El curtailment (energía renovable no aprovechada) se maneja mediante acciones de los agentes solar/wind. Si `action_solar#0 = 0`, no se inyecta generación aunque haya `potential_solar#0 > 0`.

---

## � Análisis de Estabilidad (Nuevo)

### **Análisis de Estabilidad para Sistemas MARL Distribuidos**

Dos estudios independientes para validar convergencia y consenso:

#### **1. Estabilidad por Contracción de Bellman**

Mide convergencia de funciones de valor entre episodios consecutivos:

```bash
# Paso 1: Recolectar Q-tables por episodio
python analysis_tools/collect_qtables_per_episode.py

# Paso 2: Ejecutar análisis de estabilidad
python analysis_tools/run_stability_analysis.py
```

**Métrica:** `ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞`

**Interpretación:**
- ✅ `ΔV → 0`: Convergencia estable
- ⚠️ `ΔV oscila`: Inestabilidad
- ❌ `ΔV diverge`: No converge

#### **2. Estabilidad por Consenso Distribuido**

Mide consenso entre agentes:

**Métrica:** `D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2`

**Interpretación:**
- ✅ `D → 0`: Consenso logrado
- ⚠️ `D > 0 persistente`: Falta de coordinación
- ❌ `D diverge`: Inestabilidad del sistema

**Archivos de Salida:**
- `results/stability/bellman_contraction_stability.csv`
- `results/stability/bellman_contraction_stability.png`
- `results/stability/consensus_stability.csv`
- `results/stability/consensus_stability.png`

**Documentación detallada:** Ver [docs/STABILITY_ANALYSIS.md](../docs/STABILITY_ANALYSIS.md) y [docs/STABILITY_ANALYSIS_QUICK_REF.md](../docs/STABILITY_ANALYSIS_QUICK_REF.md)

---

## 📖 Referencias y Archivos Clave

- **Simulación**: `core/simulation.py`, `core/environment.py`
- **Agentes**: `agents/base_agent.py`, `agents/*_agent.py`
- **Políticas**: `core/policies.py`, `core/rewards.py`
- **Config**: `configs/default.yaml`, `configs/loader.py`
- **Análisis de Estabilidad**: `analysis_tools/stability_analysis.py`, `docs/STABILITY_ANALYSIS.md`
- **Instrucciones generales**: `.github/copilot-instructions.md`

---

## 💡 Consejos para Afinar Recompensas

1. **Iteración corta**: Usa `episodes: 50` para pruebas rápidas.
2. **Umbrales dinámicos**: Ajusta `analysis.thresholds` según resultados previos.
3. **Análisis por agente**: Revisa `total_reward_<agent>` para identificar agentes con bajo desempeño.
4. **Balance vs Penetración**: Si IAE es bajo pero Grid_Penetration alta, considera aumentar penalizaciones en `grid_agent.reward`.
5. **Epsilon decay**: Para convergencia más rápida, usa `schedule: exponential` con `decay: 0.99`.

---

## 📝 Changelog y Versiones

- **v2.0** (2025-10-17): Refactorización completa con scripts A–E, métricas consolidadas, política ephemeral.
- **v1.x**: Scripts dispersos en raíz (deprecated).

---

**¿Dudas o mejoras?** Consulta `.github/copilot-instructions.md` o revisa los scripts individuales con comentarios inline.
