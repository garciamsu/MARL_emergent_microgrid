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
- `analysis_tools/`: análisis post‑hoc (chequeos de datos, corridas de entrenamiento, métricas, gráficos) que asumen los formatos actuales de CSV/log.

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
- Al añadir nuevas opciones, extiende este YAML y léelo en `configs/loader.py` o `core/utils` en lugar de usar constantes hard‑codeadas.

## Bucle de Entrenamiento y Resultados

- Ejecución estándar de entrenamiento: `python main.py`.
- Flujo típico dentro de `core/simulation.py`:
  1. Fijar la semilla global con `simulation.seed`.
  2. Por episodio: SOC inicial de la batería desde el YAML.
  3. Reiniciar entorno y agentes, luego avanzar el episodio (1 paso por hora).
  4. Seleccionar acciones con política epsilon‑greedy (`policies/epsilon_greedy.py`) y actualizar Q‑tables.
  5. Registrar la evolución en `results/evolution/episode_<n>.csv` y el estado general en `results/logs/`.
- No cambies los nombres de archivos ni la estructura de directorios en `results/` sin actualizar también los scripts en `analysis_tools/`.

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

## Herramientas y Flujos de Trabajo

- Self‑check rápido: `python scripts/self_check.py` (corrida corta de validación).
- Pipeline completo y análisis: `analysis_tools/run_full_pipeline.py` y scripts individuales en `analysis_tools/`.
- Tests de humo con pytest (cuando existan): `pytest -q` o `python -m pytest -k smoke -q` desde la raíz del repositorio.

## Cómo Extender de Forma Segura

- Al añadir funcionalidades, prioriza:
  - Nuevos campos de configuración en `configs/default.yaml`.
  - Helpers pequeños y enfocados en `core/utils.py`, `utils/discretization.py` o `analysis_tools/utils.py`.
  - Reutilizar el sistema de logging y las utilidades de limpieza existentes en lugar de scripts ad‑hoc.
- Evita romper la compatibilidad hacia atrás de los formatos de CSV/log; los scripts de `analysis_tools/` dependen de su esquema actual.

Si alguna asunción arquitectónica no es evidente (flujo de recompensas, manejo de datasets o registro de agentes), revisa en conjunto `core/environment.py`, `core/rewards.py` y el paquete `agents/` antes de hacer refactors grandes.
