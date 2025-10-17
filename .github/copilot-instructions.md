# Guía corta para agentes de IA (MARL microgrid)

## Qué es y cómo corre
- Punto de entrada: `main.py` → limpia (`analysis_tools.utils.clear_directories`), carga config (`configs/loader.py`), ejecuta `core/simulation.run_training`.
- Resultados por episodio: `results/evolution/episode_<n>.csv`; logs en `results/logs/run_*.log`.
- Ejecutar: `python main.py`. Self-check: `python scripts/self_check.py`. Tests: `pytest -q` (ver `test/test_smoke.py`).

## Arquitectura esencial
- `core/`:
  - `environment.MultiAgentEnv` carga CSV de `assets/datasets/`, crea bins (`discretization.bins_power`) y expone índices discretizados (`get_value`, `get_dataset`). Usa `dt_h` (fijado a 1.0) para SOC.
  - `simulation.run_training` itera episodios y pasos; orden de actualización por paso: 1) `solar`/`wind` → 2) `load` → 3) `battery` → 4) `grid`.
  - `policies.TabularQL` (epsilon-greedy), `registry` (decoradores), `utils` (semilla+logger).
- `agents/`: `BaseAgent` define `get_discretized_state`, `choose_action`, `update_q_table`. Las recompensas se consumen desde `reward_fn` (YAML) — no hay `calculate_reward` en los agentes. Q-table se inicializa con `initialize_q_table` a partir de `state_space`.
- Autoregistro: `agents/__init__.py` importa todos los `*_agent.py`.

## Estado, acciones y acumuladores
- `state_space` declarativo por agente: fuentes `local|env|global|self|external`.
  - `local` espera columnas `<var>_<idx>` (ej.: `solar_power_0`, `wind_power_0`).
  - `env.get_dataset('demand', i)` también fija `env.price` y `env.demand_power`.
  - `external` lee atributos crudos del entorno (ej.: `soc_idx`).
- Convenciones de potencia: generación positiva; consumo negativo en agentes, pero el entorno acumula demanda como positivo en `env.demand_power`.
- Orden importa: la batería usa el balance preliminar (renovables − demanda) antes de que actúe la red.

## Configuración clave (`configs/default.yaml`)
- `simulation`: `episodes`, `dt_h=1.0` (validado), `seed`, `dataset`, `epsilon`.
- `discretization.bins_power` (la clave `power_bins` repetida no se usa).
- `agents.<tipo>`: `policy`, `state_space`, `limits`, `reward` (OBLIGATORIO ahora).

## Gotchas reales del código
- Epsilon: ahora se usa un scheduler según `simulation.epsilon.schedule` (`linear|exponential|constant|custom`) que respeta `start`, `end`, `decay`, `min` y `values` (sin 0.99 fijo). Si falta `decay` en `exponential` y hay `start/end`, se deriva; siempre se clippea a `min`.
- Recompensas: el loop exige `reward_fn` y llama `reward_fn.compute(agent, env, state_tuple)`. Si falta, se lanza error indicando el bloque YAML requerido.
- Q-table: se inicializa al instanciar cada agente con `initialize_q_table(env)` en base a `state_space`; si un estado no está, se lanza error (no hay creación bajo demanda).
- `dt_h` está fijado a 1.0 por validación para evitar inconsistencias en la cinemática de la batería.
- Bloques de config hoy no utilizados: `validation`, `stability`, `metrics`, parte de `io`.

## Patrones para extender
- Nuevo agente: `agents/<name>_agent.py` con `@register_agent("<name>")`, heredar `BaseAgent`, implementar `update_power`; en YAML define `state_space`, `limits`, `policy`, `reward`.
- Para `local`, incluye el sufijo de índice en el CSV (ej.: `wind_power_0`).

## Archivos de referencia
`core/simulation.py`, `core/environment.py`, `agents/base_agent.py`, `agents/*_agent.py`, `core/policies.py`, `core/registry.py`, `core/rewards.py`, `configs/default.yaml`.
