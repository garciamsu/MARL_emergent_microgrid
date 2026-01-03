## Resumen del Proyecto

Este repositorio implementa un marco de **aprendizaje por refuerzo multi‑agente (MARL)** para la operación de una microred con:
- Agentes: `solar`, `wind`, `battery`, `grid`, `load` (ver `agents/`).
- Núcleo de entorno y bucle de entrenamiento: `core/environment.py`, `core/simulation.py`, `main.py`.
- Datos: datasets horarios en CSV en `assets/datasets/`, seleccionados vía `configs/default.yaml`.

El código es completamente orientado a objetos; cada componente físico es un agente con su propia política y configuración de recompensas.

## Arquitectura y Módulos Clave

- `main.py`: punto de entrada principal. Carga `configs/default.yaml`, construye entorno y agentes, ejecuta entrenamiento y escribe salidas en `results/`.
- `core/environment.py`: entorno multi‑agente (construcción del estado, avance temporal, acceso a dataset, evolución de SOC, balance de potencia).
- `core/simulation.py`: bucle de entrenamiento (episodios, programación de epsilon, actualización de Q‑tables, logging, exportación a CSV).
- `agents/*.py`: clases concretas de agentes (solar, eólica, batería, red, carga) registradas mediante decoradores; implementan `update_power` y ganchos de política.
- `core/rewards.py`: cálculo centralizado de recompensas impulsado por la configuración YAML (`agents.<type>.reward`). Los agentes no implementan su propia función de recompensa.
- `utils/discretization.py`: utilidades de discretización de estado‑acción; las Q‑tables son tabulares sobre estos espacios discretos.
- `analysis/`: análisis post‑hoc (chequeos de datos, corridas de entrenamiento, métricas, gráficos) que asumen los formatos actuales de CSV/log.

## Datos, Paso de Tiempo y Episodios

- El paso de tiempo `dt_h` está fijado a 1.0 h (ver `configs/loader.py`); mantener siempre la semántica 1 paso ⇾ 1 hora.
- Los datasets son CSV indexados por hora en `assets/datasets/`; la selección se controla con `simulation.dataset` en `configs/default.yaml`.
- Los episodios de entrenamiento recorren índices de tiempo contiguos; evita barajar filas o reordenar el tiempo cuando añadas nueva lógica.

## Convenciones de Configuración

- Fuente única de verdad: `configs/default.yaml`.
- Bloques importantes:
  - `simulation.*`: episodios, dataset, semilla, programación de epsilon, posible escalado de demanda.
  - `agents.<type>.policy.{alpha,gamma}`: hiperparámetros de Q‑learning tabular.
  - `agents.<type>.reward`: pesos y parámetros de recompensa consumidos en `core/rewards.py`.
  - `agents.battery.limits.*`: límites de SOC y comportamiento del SOC inicial por episodio.
  - `discretization.*`: número de bins y rangos usados por `utils/discretization.py`.
  - `stability.window`: tamaño de ventana para promedios móviles en análisis (usado por `analysis/operative/E_accumulated_reward.py`).
  - `io.results_dir`: directorio base para todos los archivos de salida (usado por `core/simulation.py` y scripts de análisis).
- Al añadir nuevas opciones, extiende este YAML y léelo en `configs/loader.py` o `core/utils` en lugar de usar constantes hard‑codeadas.
- **Principio**: Evita valores hardcodeados; lee siempre desde `default.yaml` cuando el parámetro afecta múltiples módulos o análisis.

## Bucle de Entrenamiento y Resultados

- Ejecución estándar de entrenamiento: `python main.py`.
- Flujo típico dentro de `core/simulation.py`:
  1. Fijar la semilla global con `simulation.seed`.
  2. Por episodio: SOC inicial de la batería desde el YAML.
  3. Reiniciar entorno y agentes, luego avanzar el episodio (1 paso por hora).
  4. Seleccionar acciones con política epsilon‑greedy (`policies/epsilon_greedy.py`) y actualizar Q‑tables.
  5. Registrar la evolución en `results/evolution/episode_<n>.csv` y el estado general en `results/logs/`.
- No cambies los nombres de archivos ni la estructura de directorios en `results/` sin actualizar también los scripts en `analysis/`.

## Paradigma de Estado Dinámico con Consumo Estigmérgico

El sistema implementa un **flujo de estado dinámico** donde la variable estigmérgica `delta_ph` se actualiza conforme cada agente consume su porción del potencial renovable:

### Flujo de Ejecución por Paso
1. **FASE 0**: `env.load_timestep_data(index)` - Carga demand, price, potenciales del dataset
2. **FASE 1**: Solar observa → decide → ejecuta → consume potencial → recalcula delta_ph
3. **FASE 2**: Wind observa (ve delta_ph reducido) → decide → ejecuta → consume potencial
4. **FASE 3**: Battery observa → decide → ejecuta
5. **FASE 4**: Grid observa → decide → ejecuta (cubre déficit residual)
6. **FASE 5**: Load observa → decide → ejecuta (demand response)

### Variable Estigmérgica: delta_ph
- **Definición**: `delta_ph = renewable_potential - demand_power`
- **Comportamiento**: Se actualiza dinámicamente tras cada agente renovable
- **Semántica**:
  - Positivo → Potencial excedente disponible
  - Cero → Balance exacto
  - Negativo → Déficit de potencial

### Balance Real: real_balance
- **Definición**: `real_balance = renewable_power - demand_power`
- **Uso**: Para recompensas de batería y grid (no estigmérgico)
- **Justificación**: La batería/grid responden al balance físico real, no al potencial restante

### Señales Diferenciadas por Tipo de Agente
| Agente | Señal para Recompensa | Razón |
|--------|----------------------|-------|
| Solar, Wind | `delta_ph` (estigmérgico) | Coordinan acceso al potencial renovable |
| Battery, Grid | `real_balance` (potencia real) | Responden al balance físico real |

### Métodos Clave en Environment
- `load_timestep_data(index)`: Carga datos base del dataset (DEBE llamarse primero)
- `update_delta_ph()`: Recalcula delta_ph y real_balance con valores actuales
- `consume_renewable_potential(power, source)`: Reduce potencial tras inyección renovable

### Columnas en CSV de Evolución
- `env_delta_ph_initial`: delta_ph antes de cualquier acción
- `env_delta_ph_final`: delta_ph después de consumo estigmérgico
- `env_delta_ph_norm`, `env_delta_ph_idx`: Valores normalizados y discretizados
- `env_real_balance`: Balance real de potencia (renewables - demand)
- `env_real_balance_norm`, `env_real_balance_idx`: Valores normalizados y discretizados

### Documentación Detallada
- **Guía completa**: `docs/DYNAMIC_STATE_PARADIGM.md`

## Recompensas y Agentes

- Las recompensas están centralizadas: modifica siempre la lógica en `core/rewards.py` y en las secciones YAML correspondientes, nunca dentro de las clases de agentes.
- Cada agente expone al menos `update_power` y reutiliza utilidades compartidas para discretización y actualización de Q‑tables.
- Para introducir un nuevo agente:
  - Crea `agents/<name>_agent.py`.
  - Regístralo en `core/registry.py` con el decorador existente.
  - Añade su configuración bajo `agents.<name>` en `configs/default.yaml`.

## Estilo de Código e Invariantes

- Mantén el código orientado a objetos; no conviertas los módulos núcleo en scripts puramente procedimentales.
- Comentarios y docstrings deben estar en inglés.
- Preserva las APIs públicas existentes entre módulos (por ejemplo, `reset/step` del entorno, `update_power` de los agentes, utilidades de discretización y funciones de registro).
- Mantén el comportamiento determinista cuando `simulation.seed` esté definido (usa las utilidades de `core/utils.py`).
- Mantén la aplicación lo más simple posible, evitando complejidades innecesarias.
- Todos los archivos generados (CSV, XLSX, gráficos, etc.) deben ubicarse en el directorio `results/` correspondiente.
- Toda la documentación relevante debe actualizarse para reflejar cambios de comportamiento, incluyendo ejemplos y guías de interpretación en `docs/`.
- **No** dejes scripts de pruebas o debug en el código final; úsalos solo para validar y elimínalos después.

Cada nueva funcionalidad o cambio de comportamiento relevante que desarrollen los agentes de IA debe reflejarse y resumirse apropiadamente en este archivo `copilot-instructions.md`, para mantener estas reglas siempre alineadas con el estado real del proyecto.

Las respuestas generadas por GitHub Copilot u otros agentes de IA para este repositorio deben estar redactadas en **español**, salvo que se trate de comentarios/docstrings en el código, que siguen siendo en inglés.

## Herramientas y Flujos de Trabajo

- Self‑check rápido: `python scripts/self_check.py` (corrida corta de validación).
- Pipeline completo y análisis: `analysis/run_full_pipeline.py` y scripts individuales en `analysis/`.
- Tests de humo con pytest (cuando existan): `pytest -q` o `python -m pytest -k smoke -q` desde la raíz del repositorio.

## Análisis de Estabilidad MARL (Nuevo)

El repositorio ahora incluye herramientas de análisis de estabilidad para sistemas MARL distribuidos:

### Módulos de Análisis de Estabilidad

- `analysis/stability/stability_analysis.py`: Clases analizadoras para estudios de estabilidad (Bellman y Consenso).
- `analysis/collect_qtables_per_episode.py`: Recolecta snapshots de Q-tables por episodio durante el entrenamiento.
- `analysis/run_stability_analysis.py`: Ejecuta ambos análisis de estabilidad sobre datos recolectados.
- `analysis/tests/test_stability_analysis.py`: Suite de validación con datos sintéticos.

### Dos Estudios de Estabilidad

**1. Estabilidad por Contracción de Bellman:**
- Métrica: `ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞`
- Mide convergencia de funciones de valor entre episodios consecutivos.
- `ΔV → 0` indica aprendizaje estable y convergencia.

**2. Estabilidad por Consenso Distribuido:**
- Métrica: `D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2`
- Mide consenso entre agentes respecto a la representación de valor promedio.
- `D → 0` indica consenso distribuido y coordinación emergente.

### Flujo de Uso

```bash
# 1. Recolectar Q-tables por episodio
python analysis/collect_qtables_per_episode.py

# 2. Ejecutar análisis de estabilidad
python analysis/run_stability_analysis.py

# 3. Validar implementación (opcional)
python analysis/test_stability_analysis.py
```

### Salidas

- `results/stability/qtables_per_episode.npz`: Historial de Q-tables comprimido.
- `results/stability/bellman_contraction_stability.csv` y `.png`: Métricas y gráfico de contracción.
- `results/stability/consensus_stability.csv` y `.png`: Métricas y gráfico de consenso.

### Documentación

- **Guía completa**: `docs/STABILITY_ANALYSIS.md` (fundamentos teóricos, interpretación, uso avanzado).
- **Referencia rápida**: `docs/STABILITY_ANALYSIS_QUICK_REF.md` (comandos, umbrales, troubleshooting).
- **Integración en pipeline**: Ver `analysis/README.md` sección "Análisis de Estabilidad".

### Características Clave

- **No modifica código existente**: Scripts autocontenidos que no alteran agentes, recompensas ni entrenamiento.
- **Análisis post-hoc**: Opera sobre datos recolectados; no interfiere con el bucle de entrenamiento.
- **Validación teórica**: Métricas alineadas con teoría de programación dinámica distribuida y sistemas emergentes.
- **Manejo robusto de heterogeneidad**: Soporta Q-tables de tamaños variables entre agentes y episodios (padding automático).
- **Interpretación clara**: Umbrales y guías de interpretación incluidos en documentación y plots.

## Manejo de CSV y Formato de Datos

El proyecto utiliza un sistema **estandarizado** para el manejo de archivos CSV que previene conflictos con símbolos decimales:

### Formato Estandarizado
- **INPUT (datasets)**: Formato europeo - `sep=';'`, `decimal=','`
  - Ubicación: `assets/datasets/*.csv`
  - Función: `read_dataset_csv()` de `core/csv_handler.py`
- **OUTPUT (results)**: Formato internacional - `sep=','`, `decimal='.'`
  - Ubicación: `results/**/*.csv`
  - Funciones: `write_result_csv()` y `read_result_csv()` de `core/csv_handler.py`

### Reglas Obligatorias
1. **NUNCA** usar `pd.read_csv()` o `.to_csv()` directamente en código de producción
2. **SIEMPRE** usar las funciones del módulo `core/csv_handler.py`:
   - `read_dataset_csv()` - para leer datasets de entrada
   - `read_result_csv()` - para leer archivos de resultados
   - `write_result_csv()` - para escribir archivos de resultados
3. **VALIDAR** cambios ejecutando: `python scripts/validate_csv_consistency.py`

### Archivos Actualizados (No Modificar sin Justificación)
Los siguientes archivos ya usan el formato estandarizado:
- `core/environment.py`, `core/simulation.py`
- `analysis/operative/A_data_check.py`, `analysis/operative/C_collect_episodes.py`, `analysis/operative/D_compute_metrics.py`, `analysis/operative/E_accumulated_reward.py`, `analysis/operative/E_graph_episode.py`, `analysis/common/utils.py`, `analysis/stability/stability_analysis.py`
- `scripts/hyperparameter_search.py`, `validate_load_agent.py`

### Documentación
- **Guía completa**: `docs/CSV_FORMAT_STANDARDIZATION.md`
- **Resumen de cambios**: `docs/CSV_STANDARDIZATION_SUMMARY.md`
- **Validación**: `scripts/validate_csv_consistency.py`

## Mecanismos de Seguridad y Fallback

### Inicialización Optimista de Q-tables
Las Q-tables se inicializan con un valor configurable para incentivar exploración:

- **Configuración**: `simulation.q_init_value` (default: 5.0)
- **Efecto**: Valores altos incentivan exploración de acciones no probadas
- **Rango típico**: 0.0 (neutral) a 10.0 (muy optimista)

## Cómo Extender de Forma Segura

- Al añadir funcionalidades, prioriza:
  - Nuevos campos de configuración en `configs/default.yaml`.
  - Helpers pequeños y enfocados en `core/utils.py`, `utils/discretization.py` o `analysis/common/utils.py`.
  - Reutilizar el sistema de logging y las utilidades de limpieza existentes en lugar de scripts ad‑hoc.
  - **Usar siempre `core/csv_handler.py`** para leer/escribir CSVs (ver sección anterior).
- Evita romper la compatibilidad hacia atrás de los formatos de CSV/log; los scripts de `analysis/` dependen de su esquema actual.

Si alguna asunción arquitectónica no es evidente (flujo de recompensas, manejo de datasets o registro de agentes), revisa en conjunto `core/environment.py`, `core/rewards.py` y el paquete `agents/` antes de hacer refactors grandes.
