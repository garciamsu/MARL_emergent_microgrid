# Guía corta para agentes de IA (MARL microgrid)

## Qué es y cómo corre
- Punto de entrada: `main.py` → limpia (`analysis_tools.utils.clear_directories`), carga config (`configs/loader.py`), ejecuta `core/simulation.run_training`.
- Resultados por episodio: `results/evolution/episode_<n>.csv`; logs en `results/logs/run_*.log`.
- Ejecutar: `python main.py`. Self-check: `python scripts/self_check.py`. Tests: `pytest -q` (ver `test/test_smoke.py`).

## Arquitectura esencial
- `core/`:
  - `environment.MultiAgentEnv` carga CSV de `assets/datasets/`, crea bins (`discretization.bins_power`) y expone índices discretizados (`get_value`, `get_dataset`). Usa `dt_h` para SOC.
  - `simulation.run_training` itera episodios y pasos; orden de actualización por paso: 1) `solar`/`wind` → 2) `load` → 3) `battery` → 4) `grid`.
  - `policies.TabularQL` (epsilon-greedy), `registry` (decoradores), `utils` (semilla+logger).
- `agents/`: `BaseAgent` define `get_discretized_state`, `choose_action`, `update_q_table`. Agentes concretos implementan `update_power` y `calculate_reward` (la recompensa efectiva vive aquí).
- Autoregistro: `agents/__init__.py` importa todos los `*_agent.py` del folder; basta con crear el archivo y anotar con `@register_agent`.

## Estado, acciones y acumuladores
- `state_space` declarativo por agente: fuentes `local|env|global|self|external`.
  - `local` espera columnas `<var>_<idx>` (ej.: `solar_power_0`, `wind_power_0`).
  - `env.get_dataset('demand', i)` también fija `env.price` y `env.demand_power`.
  - `external` lee atributos crudos del entorno (ej.: `soc_idx`).
- Convenciones de potencia: generación positiva; consumo negativo en agentes, pero el entorno acumula demanda como positivo en `env.demand_power`.
- Orden importa: la batería usa el balance preliminar (renovables − demanda) antes de que actúe la red.

## Configuración clave (`configs/default.yaml`)
- `simulation`: `episodes`, `dt_h`, `seed`, `dataset`, `epsilon`.
- `discretization.bins_power` usado por el entorno (las claves `power_bins` repetidas actualmente no se usan).
- `agents.<tipo>`: `policy` (p.ej. `tabular_ql`), `state_space`, `limits`.

## Gotchas reales del código
- Epsilon: el loop asigna `decay = epsilon_cfg.get('decay', 'linear')` y luego compara `decay` con cadenas (`"linear"|"exponential"`). Si en YAML `decay` es numérico (ej.: `0.997`), NO habrá decaimiento (no entra en ninguna rama) y, en caso de exponencial, usa fijo `0.99`. Soluciones: cambiar el código para respetar `schedule`+`decay`, o en YAML definir sólo `schedule` y ajustar el código; no confiar en `end`.
- `reward_fn` del YAML se instancia pero NO se invoca; se usa `agent.calculate_reward(...)` de cada clase.
- `initialize_q_table` existe pero no se usa; la Q-table se crea bajo demanda.
- `dt_h` impacta SOC y límites efectivos por paso en batería; cambiarlo sin revisar `p_charge_max`/`p_discharge_max` puede saturar/ahogar SOC.
- Bloques de config no utilizados hoy: `validation`, `stability`, `metrics`, parte de `io` (rutas de salida están codificadas como `results/...`). Documenta si empiezas a usarlos.

## Patrones para extender
- Nuevo agente: `agents/<name>_agent.py` con `@register_agent("<name>")`, heredar `BaseAgent`, implementar `update_power` y `calculate_reward`; agregar bloque en YAML con `state_space`, `limits`, `policy`. Nombrado de instancias: `<tipo>#<idx>`.
- Al leer `local`, la columna debe incluir sufijo del índice (ej.: `wind_power_0`).

## Archivos de referencia
`core/simulation.py`, `core/environment.py`, `agents/base_agent.py`, `agents/*_agent.py`, `core/policies.py`, `core/registry.py`, `configs/default.yaml`.
