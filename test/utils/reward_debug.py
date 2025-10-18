"""Utilidades de depuración para funciones de recompensa.

Incluye:
- explain_and_compute: imprime hiperparámetros, inputs y traza del cálculo.
- build_reward_from_config: construye la reward leyendo params de configs/default.yaml.
"""

from typing import Tuple
import sys
import tty
import termios

# REPO_ROOT lo agrega cada test antes de importar este módulo
try:
    from configs.loader import load_config as _load_config
except ImportError:  # pragma: no cover - import dinámico segun sys.path
    _load_config = None


def _read_single_key():
    fd_handle = sys.stdin.fileno()
    old = termios.tcgetattr(fd_handle)
    try:
        tty.setraw(fd_handle)
        ch_key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd_handle, termios.TCSADRAIN, old)
    return ch_key


def _wait_for_space(skip_wait=False):
    # If caller requested skip, or inner explains are suppressed, return immediately
    if skip_wait or globals().get('_suppress_inner_waits', False):
        print("[DEBUG] Skipping wait for space due to test mode.")
        return True

    print("Presiona BARRA ESPACIADORA para continuar, 'q' para salir...", end='', flush=True)
    while True:
        k = _read_single_key()
        if k == ' ':
            print('')
            # Call any registered callback for space press only if enabled
            invoke = globals().get('_invoke_on_space_callback', True)
            if invoke:
                callback = globals().get('_on_space_callback')
                if callable(callback):
                    callback()
            return True
        if k.lower() == 'q':
            print('\nInterrumpido por usuario.')
            raise KeyboardInterrupt()


def explain_and_compute(reward_fn, agent, env, state_tuple: Tuple, step: bool = False):
    """
    Imprime hiperparámetros, significado de los inputs y una traza paso a paso
    que replica la lógica de las funciones de recompensa definidas en
    `core.rewards`. Devuelve el valor de recompensa calculado.
    """

    # Diccionario de descripciones de variables por agente
    state_desc = {
        'battery': [
            ('soc_idx', 'Índice discreto del estado de carga de la batería, rango [0, soc_max]'),
            ('demand_power_idx', 'Índice discreto de la demanda de potencia'),
            ('total_power_idx', 'Índice discreto de la potencia total disponible')
        ],
        'grid': [
            ('soc_idx', 'Índice discreto del estado de carga de la batería'),
            ('demand_power_idx', 'Índice discreto de la demanda de potencia'),
            ('total_power_idx', 'Índice discreto de la potencia total disponible')
        ],
        'load': [
            ('soc_idx', 'Índice discreto del estado de carga de la batería'),
            ('demand_power_idx', 'Índice discreto de la demanda de potencia'),
            ('renewable_potential_idx', 'Índice discreto del potencial renovable disponible')
        ],
        'solar': [
            ('solar_potential_idx', 'Índice discreto del potencial solar'),
            ('demand_power_idx', 'Índice discreto de la demanda de potencia'),
            ('total_power_idx', 'Índice discreto de la potencia renovable agregada')
        ],
        'wind': [
            ('wind_potential_idx', 'Índice discreto del potencial eólico'),
            ('demand_power_idx', 'Índice discreto de la demanda de potencia'),
            ('total_power_idx', 'Índice discreto de la potencia renovable agregada')
        ]
    }

    # Diccionario de descripciones de hiperparámetros por clase
    param_desc = {
        'theta': 'Ponderación para penalización por déficit renovable',
        'beta': 'Ponderación para bonificación por excedente renovable',
        'eta': 'Ponderación para penalización por excedente no usado',
        'xi': 'Bonificación por no activar renovable cuando no hay excedente',
        'psi': 'Bonificación por descarga útil de batería',
        'sigma': 'Penalización por descarga no útil de batería',
        'nu': 'Bonificación por carga útil de batería',
        'mu': 'Alias de nu en YAML',
        'C_M': 'Costo marginal de la red',
        'comfort_threshold': 'Umbral de confort para el agente de carga',
    }

    cls_name = reward_fn.__class__.__name__
    print(f"\n=== Depuración de recompensa: {cls_name} ===")
    print("\n--- Hiperparámetros ---")
    for key, val in vars(reward_fn).items():
        desc = param_desc.get(key, '')
        if desc:
            print(f"  {key}: {val}  # {desc}")
        else:
            print(f"  {key}: {val}")

    print("\n--- Atributos del agente ---")
    try:
        for key, val in vars(agent).items():
            print(f"  {key}: {val}")
    except TypeError:
        print("  (El agente no tiene atributos introspectables)")

    if env is not None:
        print("\n--- Atributos del entorno (env) ---")
        try:
            for key, val in vars(env).items():
                print(f"  {key}: {val}")
        except TypeError:
            print("  (El entorno no es introspectable)")

    # Detectar tipo de agente para mostrar descripciones
    agent_type = None
    if reward_fn.__class__.__name__.lower().startswith('defaultbattery'):
        agent_type = 'battery'
    elif reward_fn.__class__.__name__.lower().startswith('defaultgrid'):
        agent_type = 'grid'
    elif reward_fn.__class__.__name__.lower().startswith('defaultload'):
        agent_type = 'load'
    elif reward_fn.__class__.__name__.lower().startswith('defaultsolar'):
        agent_type = 'solar'
    elif reward_fn.__class__.__name__.lower().startswith('defaultwind'):
        agent_type = 'wind'

    print("\n--- Inputs (state_tuple) ---")
    if agent_type and agent_type in state_desc:
        for (var, desc), val in zip(state_desc[agent_type], state_tuple):
            print(f"  {var}: {val}  # {desc}")
    else:
        for i, val in enumerate(state_tuple):
            print(f"  [{i}]: {val}")

    # Prepare concise summary callback (used when stepping)
    def _print_summary():
        val = globals().get('_last_computed_value')
        if val is None:
            return
        print(f"State: {state_tuple}, Action: {getattr(agent, 'action', None)}, Reward: {val}")

    # Ensure callback and invoke flag exist
    globals()['_on_space_callback'] = None
    globals()['_invoke_on_space_callback'] = True

    # Disable inner waits (they should not trigger the final summary prompt)
    if step:
        globals()['_suppress_inner_waits'] = True

    # Dispatch por tipo de reward para replicar la lógica y mostrar pasos
    if cls_name == 'DefaultBatteryReward':
        val = _explain_battery(reward_fn, agent, env, state_tuple, step)
    elif cls_name == 'DefaultGridReward':
        val = _explain_grid(reward_fn, agent, env, state_tuple, step)
    elif cls_name == 'DefaultLoadReward':
        val = _explain_load(reward_fn, agent, env, state_tuple, step)
    elif cls_name == 'DefaultSolarReward' or cls_name == 'DefaultWindReward':
        val = _explain_solar_wind(reward_fn, agent, env, state_tuple, step)
    else:
        # Fallback: usar compute directo si no se reconoce el tipo
        print("\n--- Tipo de recompensa no reconocido para trazado; usando compute() directo ---")
        val = reward_fn.compute(agent, env, state_tuple)
        print(f"Resultado: {val}")

    # store last computed value
    globals()['_last_computed_value'] = val

    # Print concise result before showing the final prompt (if stepping)
    if step:
        # Ensure inner waits were suppressed
        globals()['_suppress_inner_waits'] = False
        # show final prompt and concise summary will be triggered by space press
        globals()['_on_space_callback'] = _print_summary
        try:
            _wait_for_space(skip_wait=False)
        finally:
            globals()['_on_space_callback'] = None

    return val


def _explain_battery(rwd, agent, _env, state_tuple, step):
    soc, demand_idx, total_idx = state_tuple
    print(f"  soc (continuo) = {soc}")
    print(f"  demand_idx = {demand_idx}")
    print(f"  total_idx = {total_idx}")
    delta_p = total_idx - demand_idx
    print(f"  delta_p = total_idx - demand_idx = {delta_p}")
    print(
        f"  action del agente = {agent.action}  (0: nada, 1: cargar?, 2: descargar?)"
    )
    if step:
        _wait_for_space()

    # Reproducir condiciones tal como en core/rewards.py
    if agent.action == 2 and delta_p < 0 and soc > 0:
        val = rwd.psi * abs(delta_p) * soc
        print(
            "Caso: action==2 & delta_p<0 & soc>0 -> "
            f"psi * |delta_p| * soc = {rwd.psi} * {abs(delta_p)} * {soc} = {val}"
        )
        return val
    if agent.action == 2 and (delta_p >= 0 or soc == 0):
        val = -rwd.sigma
        print(
            "Caso: action==2 & (delta_p>=0 or soc==0) -> "
            f"-sigma = -{rwd.sigma} = {val}"
        )
        return val
    if agent.action == 1 and delta_p > 0:
        soc_max = getattr(agent, 'soc_max', 1.0)
        val = rwd.nu * delta_p * (soc_max - soc)
        print(
            "Caso: action==1 & delta_p>0 -> "
            f"nu * delta_p * (soc_max - soc) = {rwd.nu} * {delta_p} * ({soc_max} - {soc}) = {val}"
        )
        return val
    if agent.action == 1 and delta_p <= 0:
        val = -rwd.beta * abs(delta_p)
        print(
            "Caso: action==1 & delta_p<=0 -> "
            f"-beta * |delta_p| = -{rwd.beta} * {abs(delta_p)} = {val}"
        )
        return val
    if agent.action == 0 and abs(delta_p) > 0:
        val = -rwd.xi * abs(delta_p)
        print(
            "Caso: action==0 & |delta_p|>0 -> "
            f"-xi * |delta_p| = -{rwd.xi} * {abs(delta_p)} = {val}"
        )
        return val
    print("Caso por defecto -> 0.0")
    return 0.0


def _explain_grid(rwd, agent, _env, state_tuple, step):
    soc_idx, demand_idx, total_idx = state_tuple
    print(f"  soc_idx = {soc_idx}")
    print(f"  demand_idx = {demand_idx}")
    print(f"  total_idx = {total_idx}")
    delta_p = total_idx - demand_idx
    print(f"  delta_p = {delta_p}")
    print(f"  action = {agent.action} (1: comprar?, 0: vender/ninguna?)")
    if step:
        _wait_for_space()

    if agent.action == 1 and delta_p < 0 and soc_idx == 0:
        val = rwd.psi / rwd.C_M
        print(
            "Caso: action==1 & delta_P<0 & soc_idx==0 -> "
            f"psi / C_M = {rwd.psi}/{rwd.C_M} = {val}"
        )
        return val
    if agent.action == 1 and (delta_p >= 0 or soc_idx > 0):
        val = -rwd.sigma * rwd.C_M
        print(
            "Caso: action==1 & (delta_P>=0 or soc_idx>0) -> "
            f"-sigma * C_M = -{rwd.sigma} * {rwd.C_M} = {val}"
        )
        return val
    if agent.action == 0 and delta_p < 0 and soc_idx == 0:
        val = -rwd.nu * rwd.C_M
        print(
            "Caso: action==0 & delta_P<0 & soc_idx==0 -> "
            f"-nu * C_M = -{rwd.nu} * {rwd.C_M} = {val}"
        )
        return val
    val = -rwd.xi
    print(f"Caso por defecto -> -xi = -{rwd.xi} = {val}")
    return val


def _explain_load(rwd, agent, env, state_tuple, step):
    soc_idx, demand_idx, renewable_idx = state_tuple
    print(f"  soc_idx = {soc_idx}")
    print(f"  demand_idx = {demand_idx}")
    print(f"  renewable_idx = {renewable_idx}")
    market_cost = getattr(env, 'price', None)
    print(f"  market_cost (env.price) = {market_cost}")
    print(f"  action = {agent.action} (1: consumir?, 0: reducir?)")
    if step:
        _wait_for_space()

    if agent.action == 1 and (soc_idx > 0 or renewable_idx > demand_idx):
        val = rwd.sigma * market_cost
        print(
            "Caso: action==1 & (soc_idx>0 or renewable_idx>demand_idx) -> "
            f"sigma * market_cost = {rwd.sigma} * {market_cost} = {val}"
        )
        return val
    if agent.action == 1 and getattr(agent, 'comfort_threshold', 0) < market_cost:
        if market_cost:
            val = -rwd.psi / market_cost
            print(
                "Caso: action==1 & comfort_threshold < market_cost -> "
                f"-psi / market_cost = -{rwd.psi} / {market_cost} = {val}"
            )
        else:
            val = -rwd.psi
            print(f"Caso: action==1 & market_cost==0 -> -psi = -{rwd.psi} = {val}")
        return val
    if agent.action == 0 and (soc_idx > 0 or renewable_idx > demand_idx):
        val = -rwd.nu * soc_idx * renewable_idx
        print(
            "Caso: action==0 & (soc_idx>0 or renewable_idx>demand_idx) -> "
            f"-nu * soc_idx * renewable_idx = -{rwd.nu} * {soc_idx} * {renewable_idx} = {val}"
        )
        return val
    val = rwd.beta
    print(f"Caso por defecto -> beta = {val}")
    return val


def _explain_solar_wind(rwd, agent, _env, state_tuple, step):
    idx, demand_idx, renewable_idx = state_tuple
    weight = (idx / renewable_idx) if renewable_idx != 0 else 1.0
    delta_abs = abs(renewable_idx - demand_idx)
    print(f"  idx = {idx}")
    print(f"  demand_idx = {demand_idx}")
    print(f"  renewable_idx = {renewable_idx}")
    print(f"  weight (idx/renewable) = {weight}")
    print(f"  delta_abs = |renewable_idx - demand_idx| = {delta_abs}")
    print(f"  action = {agent.action} (1: usar/activar?, 0: no activar?)")
    if step:
        _wait_for_space()

    if agent.action == 1:
        if renewable_idx <= demand_idx:
            val = -rwd.theta * weight * delta_abs
            print(
                "Caso: action==1 & renewable_idx<=demand_idx -> "
                f"-theta * weight * delta_abs = -{rwd.theta} * {weight} * {delta_abs} = {val}"
            )
            return val
        else:
            val = rwd.beta * weight * delta_abs
            print(
                "Caso: action==1 & renewable_idx>demand_idx -> "
                f"beta * weight * delta_abs = {rwd.beta} * {weight} * {delta_abs} = {val}"
            )
            return val
    else:
        if renewable_idx > demand_idx:
            val = -rwd.eta * weight * delta_abs
            print(
                "Caso: action!=1 & renewable_idx>demand_idx -> "
                f"-eta * weight * delta_abs = -{rwd.eta} * {weight} * {delta_abs} = {val}"
            )
            return val
        else:
            val = rwd.xi * weight
            print(
                "Caso: action!=1 & renewable_idx<=demand_idx -> "
                f"xi * weight = {rwd.xi} * {weight} = {val}"
            )
            return val


def build_reward_from_config(agent_key: str, default_class):
    """Construye una instancia de reward leyendo params del YAML.

    - agent_key: 'battery' | 'grid' | 'load' | 'solar' | 'wind'
    - default_class: clase de recompensa a instanciar si no hay override
    """
    if _load_config is None:
        # fallback: devolver instancia por defecto
        return default_class()
    cfg = _load_config()
    agent_cfg = cfg.get('agents', {}).get(agent_key, {})
    reward_cfg = agent_cfg.get('reward', {})
    params = dict(reward_cfg.get('params', {}))
    # Mapeo de alias comunes en YAML -> nombres reales de kwargs
    if 'mu' in params and 'nu' not in params:
        params['nu'] = params.pop('mu')
    # Cualquier otro parámetro faltante usa el valor por defecto de la clase
    try:
        return default_class(**params)
    except TypeError:
        # En caso de params inesperados, ignorarlos de forma permisiva
        safe = {}
        for k in params:
            if k in getattr(default_class.__init__, '__code__').co_varnames:
                safe[k] = params[k]
        return default_class(**safe)
