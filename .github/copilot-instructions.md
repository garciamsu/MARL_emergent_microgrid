# Guía para agentes de IA (MARL microgrid)

## Qué es y cómo corre
- **Punto de entrada**: `main.py` → limpia directorios, carga config (`configs/loader.py`), ejecuta `core/simulation.run_training`.
- **Resultados**: CSVs por episodio en `results/evolution/episode_<n>.csv`; logs en `results/logs/run_*.log`.
- **Comandos**: `python main.py` (entrenamiento), `python scripts/self_check.py` (validación), `pytest -q` (tests).

## Arquitectura esencial

### `core/environment.py` - MultiAgentEnv
- Carga dataset CSV desde `assets/datasets/` (config: `simulation.dataset`).
- Crea `power_bins` (de 0 a max del dataset, excluye `price`, `demand`, `Datetime`).
- **Atributos acumuladores** (actualizados en cada paso):
  - `renewable_potential`, `renewable_power`, `demand_power`, `total_power`, `energy_balance`
  - `price`, `soc_idx`, `delta_power_idx` ("surplus"/"deficit")
  - Índices discretizados: `*_idx` para cada acumulador
- `dt_h`: paso de simulación en horas (fijado a 1.0, validado en config).
- `scale_demand`: factor de escalado de demanda por episodio (config: `simulation.demand_scale`).
- **Métodos clave**:
  - `reset()`: reinicia acumuladores al inicio de cada episodio.
  - `get_dataset(field, index)`: devuelve índice discretizado; si `field='demand'` también actualiza `price` y `demand_power`.
  - `get_value(var)`: devuelve índice discretizado de variable global (`potential`, `renewable`, `demand`, `total`).

### `core/simulation.py` - run_training
Loop principal de entrenamiento episódico:
1. **Configuración inicial**: semilla global, instanciación de agentes, scheduler de epsilon.
2. **Por episodio**:
   - `env.reset()`
   - **Escalado de demanda**: modo `fixed` o `random` (U[min, max]); en último episodio fuerza valor fijo.
   - **SOC inicial batería**: modo `fixed` (lista/float) o `random` (U[initial_soc_min, initial_soc_max]); en último episodio fuerza valor fijo.
3. **Por paso** (index 0 a `max_steps-1`):
   - Carga `base_demand` del dataset y actualiza `env.price`.
   - Resetea acumuladores de potencia (`total_power=0`, `renewable_power=0`, etc.).
   - **FASE 1**: Actualiza agentes renovables (`solar`, `wind`) → acumula en `renewable_potential` y `renewable_power`.
   - **FASE 2**: Actualiza agentes `load` → acumula consumo en `demand_power`.
   - **FASE 3**: Actualiza agentes `battery` (usa balance preliminar renovables − demanda).
   - **FASE 4**: Actualiza agentes `grid` (último recurso).
   - Calcula `energy_balance = total_power - demand_power` y `delta_power_idx`.
   - Calcula recompensas (`reward_fn.compute`) y actualiza Q-tables.
   - Guarda registro del paso en `step_record`.
4. **Exporta** CSV del episodio y actualiza epsilon según scheduler.

**Scheduler de epsilon**: configurado por `simulation.epsilon`:
- `schedule`: `linear`, `exponential`, `constant`, `custom`.
- `linear`: interpola linealmente de `start` a `end` en `episodes-1` pasos.
- `exponential`: multiplica por `decay` cada episodio; si falta `decay` y hay `start/end`, se deriva automáticamente.
- `constant`: usa siempre `start`.
- `custom`: usa lista `values` (rellena con último valor si es corta).
- `min`: clip inferior aplicado a todos los modos.

### `agents/` - Sistema de agentes

#### `base_agent.py` - BaseAgent
Clase base con lógica común:
- **Atributos**: `name`, `actions`, `q_table`, `action`, `power`, `potential`, `idx`, `reward_fn`, `state_space`.
- **Métodos clave**:
  - `get_discretized_state(env, index)`: construye tupla de estado según `state_space`.
    - `local`: lee columna `<var>_<idx>` del dataset (ej: `solar_power_0`).
    - `env`: llama `env.get_dataset(var, index)`.
    - `global`: llama `env.get_value(var)`.
    - `self`: usa `self.idx`.
    - `external`: lee atributo crudo de `env` (ej: `env.soc_idx`).
  - `choose_action(state, epsilon)`: epsilon-greedy. Lanza error si estado no está en Q-table.
  - `update_q_table(state, action, reward, next_state)`: actualización tabular Q-learning.
  - `initialize_q_table(env)`: construye Q-table como producto cartesiano de dimensiones declaradas en `state_space`:
    - Si `bins` es lista/tupla → cardinalidad = `len(bins)`.
    - Si `bins == 'auto'` o ausente → cardinalidad = `len(env.power_bins)`.
    - Caso especial SOC: si `var in {'soc', 'soc_idx'}` y existe `self.battery_soc_bins` → usa esos bins.
    - Inicializa todos los estados con Q-values = 0.0.
  - `update_power(env)`: método abstracto, implementado por cada agente específico.

#### Agentes específicos (`solar_agent.py`, `wind_agent.py`, `battery_agent.py`, `grid_agent.py`, `load_agent.py`)
- Decorados con `@register_agent("<tipo>")`.
- Implementan `update_power(env)` con lógica de negocio:
  - **Solar/Wind**: `action=0` → no aporta; `action=1` → aporta `potential * efficiency` (clipeado a `p_max`).
  - **Battery**: `action=0` → idle; `action=1` → carga (power<0); `action=2` → descarga (power>0). Actualiza SOC con `update_soc`.
  - **Grid**: `action=0` → no importa; `action=1` → importa déficit faltante (clipeado a `p_max`).
  - **Load**: `action=0` → no consume; `action=1` → consume demanda base del entorno.

#### `agents/__init__.py` - instantiate_agents
- Importa dinámicamente todos los `*_agent.py`.
- Función `instantiate_agents(config, env)`:
  - Itera sobre `config["agents"]` y crea `count` instancias de cada tipo.
  - Construye `policy` (`create_policy`), `reward_fn` (`create_reward`), y `agent` (`create_agent`).
  - Llama `agent.initialize_q_table(env)` para cada agente.

### `core/rewards.py` - Sistema de recompensas
Clases decoradas con `@register_reward("<tipo>")` que heredan de `RewardFn`:
- **Interfaz**: `compute(agent, env, state_tuple) → float`.
- **Implementadas**:
  - `DefaultSolarReward`, `DefaultWindReward`: premian suministro con potencial, castigan suministro sin potencial o inacción con déficit.
  - `DefaultBatteryReward`: premia carga con excedente y descarga con déficit; castiga acciones ilógicas.
  - `DefaultGridReward`: premia importación necesaria (inversamente proporcional a precio), castiga importación innecesaria o inacción con déficit.
  - `DefaultLoadReward`: premia uso de energía interna/excedente, castiga compra cara o desperdicio.
- **Parámetros**: definidos en YAML (`agents.<tipo>.reward.params`), ej: `{theta: 4.926, beta: 4.211, ...}`.

### `core/policies.py` - TabularQL
- Policy epsilon-greedy con Q-table tabular (numpy + defaultdict).
- Métodos: `select_action(state, epsilon)`, `update(state, action, reward, next_state)`.
- Registrada como `@register_policy("tabular_ql")`.

### `core/registry.py` - Sistema de registro
- Registros globales: `AGENT_REGISTRY`, `POLICY_REGISTRY`, `REWARD_REGISTRY`.
- Decoradores: `@register_agent`, `@register_policy`, `@register_reward`.
- Factories: `create_agent`, `create_policy`, `create_reward`.

### `utils/discretization.py` - digitize_clip
Función auxiliar: `digitize_clip(value, bins) → int`. Discretiza valor en bins y clipea índice a rango válido.

## Configuración (`configs/default.yaml`)

### Bloques principales
- **`simulation`**:
  - `episodes`: número de episodios.
  - `dt_h`: paso temporal (debe ser 1.0).
  - `seed`: semilla aleatoria.
  - `dataset`: archivo CSV en `assets/datasets/`.
  - `demand_scale`: escalado de demanda (`mode: fixed|random`, `fixed`, `min`, `max`).
  - `epsilon`: scheduler (`schedule`, `start`, `end`, `decay`, `min`, `values`).
- **`discretization`**:
  - `power_bins`: número de bins para potencia.
- **`agents.<tipo>`**: (ej: `solar`, `wind`, `battery`, `grid`, `load`)
  - `count`: número de instancias.
  - `policy`: `{type: tabular_ql, alpha, gamma}`.
  - `reward`: `{type: Default<Tipo>Reward, params: {...}}`.
  - `state_space`: lista de `{var, source, bins}`.
  - `limits`: parámetros físicos (ej: `p_max`, `capacity_ah`, `soc_min/max`, `initial_soc`, etc.).

### Gotchas de configuración
- **`reward` es OBLIGATORIO**: si falta, lanza error en runtime indicando el bloque YAML requerido.
- **`state_space`**: define dimensiones de Q-table; cardinalidad de cada dimensión según `bins` o `auto`.
- **`initial_soc` en batería**: puede ser float o lista; modo `fixed|random` controla muestreo por episodio; último episodio siempre usa valor fijo.
- **Bloques no usados actualmente**: `validation`, `stability`, `metrics` (parcial), `io` (parcial).

## Convenciones de datos

### Potencia
- **Generación**: valores positivos (solar, wind, grid importando, battery descargando).
- **Consumo**: valores negativos en código de agentes; el entorno acumula demanda como positivo en `env.demand_power`.
- **Balance**: `energy_balance = total_power - demand_power`; surplus si ≥0, déficit si <0.

### Estado discretizado
- Cada agente construye tupla de estado combinando fuentes (`local`, `env`, `global`, `self`, `external`).
- La Q-table se indexa por estas tuplas (producto cartesiano de dimensiones).

### Dataset
- CSV con columnas: `Datetime`, `demand`, `price`, `solar_power_<i>`, `wind_power_<i>`, etc.
- `demand` se clipea a ≥0 al cargar.
- Máximo del dataset se calcula sumando todas las columnas numéricas excepto `price`, `demand`, `Datetime`.

## Patrones para extender

### Nuevo agente
1. Crear `agents/<nombre>_agent.py` heredando `BaseAgent`.
2. Decorar con `@register_agent("<nombre>")`.
3. Implementar `__init__` (procesar `limits`, inicializar atributos) y `update_power(env)`.
4. En YAML: agregar bloque `agents.<nombre>` con `count`, `policy`, `reward`, `state_space`, `limits`.
5. Para fuente `local` en `state_space`, asegurar columnas `<var>_<idx>` en dataset.

### Nueva recompensa
1. Crear clase en `core/rewards.py` heredando `RewardFn`.
2. Decorar con `@register_reward("<nombre>")`.
3. Implementar `__init__` (parámetros) y `compute(agent, env, state_tuple)`.
4. En YAML: usar `reward: {type: <nombre>, params: {...}}`.

### Nueva policy
1. Crear clase en `core/policies.py` heredando `Policy`.
2. Decorar con `@register_policy("<nombre>")`.
3. Implementar `select_action` y `update`.
4. En YAML: usar `policy: {type: <nombre>, ...}`.

## Archivos clave
- **Core**: `main.py`, `core/simulation.py`, `core/environment.py`, `core/policies.py`, `core/rewards.py`, `core/registry.py`, `core/utils.py`.
- **Agentes**: `agents/base_agent.py`, `agents/{solar,wind,battery,grid,load}_agent.py`, `agents/__init__.py`.
- **Utils**: `utils/discretization.py`, `configs/loader.py`.
- **Config**: `configs/default.yaml`.
- **Tests**: `test/test_smoke.py`, `scripts/self_check.py`.
