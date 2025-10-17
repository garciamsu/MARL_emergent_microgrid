# Entorno Distribuido de Aprendizaje por Refuerzo Multi‑Agente (Microred)

Este repositorio provee un marco de simulación en Python para estudiar la **coordinación distribuida de recursos energéticos renovables** usando **aprendizaje por refuerzo multi‑agente (MARL)**. El sistema modela la interacción entre generación solar, generación eólica, almacenamiento en baterías, cargas controlables y la red eléctrica principal, aplicando Q‑Learning tabular sobre espacios de estado‐acción discretizados.

El código está orientado a **ingenieros e investigadores** familiarizados con Python y RL, priorizando reproducibilidad y extensibilidad.

---

## Características Clave

- Simulación multi‑agente de una microred bajo condiciones dinámicas.
- Arquitectura modular con clases de agentes especializadas.
- Discretización configurable de variables de estado.
- Políticas tabulares por agente (Q‑Learning).
- Generación automática de:
  - Logs de evolución por paso (`CSV`)
  - Métricas por episodio (`Excel`)
  - Q‑tables (snapshots) (`Excel` en futuras extensiones)
  - Gráficas vectoriales (`SVG`) (base preparada)
- Utilidades de análisis y limpieza de resultados.

---

## Instalación

**Plataforma recomendada:** Linux Ubuntu 22.04.5 LTS  
**Versión Python:** 3.11.7  
**Gestor de entorno:** Anaconda

### Crear entorno

```bash
conda create -n marl_env python=3.11.7
conda activate marl_env
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución Básica

Punto de entrada: `main.py`.

```bash
python main.py
```

Al ejecutar se realiza:
1. Limpieza de directorios de resultados previos.
2. Carga de configuración (`configs/default.yaml`).
3. Creación del entorno y agentes.
4. Entrenamiento episodio a episodio.
5. Exportación de logs a `results/` y logging estructurado.

### Parámetros principales (en `configs/default.yaml`)
| Bloque | Claves |
|--------|-------|
| Simulación | `simulation.episodes`, `simulation.dataset`, `simulation.seed` |
| Exploración | `simulation.epsilon.start`, `simulation.epsilon.end`, `simulation.epsilon.schedule`, `decay` |
| Política | `agents.<tipo>.policy.{alpha,gamma}` |
| Recompensas | `agents.<tipo>.reward` + parámetros internos |
| Discretización | `discretization.bins_power` |

---

## Estructura de Carpetas (resultados)

Tras una corrida típica:
```
results/
├── evolution/
│   └── episode_<n>.csv            # Traza paso a paso del episodio
├── logs/
│   └── run_YYYYMMDD_HHMMSS.log    # Log consolidado
└── plots/                         # Reservado para figuras
```

---

## Métricas (concepto)

Indicadores de desempeño energético (implementables / extendibles):
- ΔP: Balance instantáneo (generación – consumo).
- IAE / ISE: Errores integrales (absoluto / cuadrático).
- REP / GEP: Penetración renovable / de red.
- Recompensas promedio por episodio.

Los CSV en `evolution/` permiten derivar estas métricas externamente.

---

## Afinación de Hiperparámetros (Tuning)

Campos clave en `configs/default.yaml`:

| Área | Claves |
|------|-------|
| Episodios & dataset | `simulation.episodes`, `simulation.dataset` |
| Exploración | `simulation.epsilon.{schedule,start,end,decay}` |
| Aprendizaje | `agents.<tipo>.policy.alpha`, `agents.<tipo>.policy.gamma` |
| Discretización | `discretization.bins_power` |
| Reproducibilidad | `simulation.seed` |

Ejemplo (cambiar a decaimiento exponencial):
```yaml
simulation:
  episodes: 50
  seed: 123
  epsilon:
    schedule: exponential
    start: 1.0
    end: 0.05
    decay: 0.97
```

### Flujo sugerido
1. Comenzar con pocos bins para iterar rápido.
2. Explorar `alpha`: 0.05, 0.1, 0.2, 0.3.
3. Ajustar `epsilon` para mantener exploración útil en ~30–40% inicial.
4. Refinar funciones de recompensa (ver agentes / `core/rewards.py`).
5. Aumentar resolución (más bins) sólo tras estabilización.

### Barrido manual
```bash
for cfg in configs/exp_*.yaml; do python main.py --config "$cfg"; done
```
(El flag `--config` puede añadirse fácilmente con argparse — pendiente si se requiere.)

---

### Tiempo de paso (dt_h) [Importante]
- En este proyecto `dt_h` está fijado a 1.0 h por validación (ver `configs/loader.py`). Cualquier valor distinto producirá un error de configuración.
- Motivo: mantener estable la cinemática de la batería y evitar cambios implícitos en la energía transferida por paso (E_step = P · dt_h).
- Si en el futuro necesitas permitir `dt_h ≠ 1.0`, considera implementar uno de estos modos en el arranque:
  - physical: usa `P_max` tal cual (límite físico del hardware).
  - per_step_invariant: `P_max_eff = P_max_base × (1.0 / dt_h)` para mantener Wh por paso constante.
  - c_rate: `P_max_eff = c_rate_max × (capacity_ah × v_nom)`.

---

## Logging y Reproducibilidad

- Semilla global: `simulation.seed` (controla Python, NumPy y Torch si disponible).
- Logger central: `core.utils.build_logger()` genera archivo en `results/logs/`.
- Cada episodio registra `epsilon` residual.
- Limpieza previa: `analysis_tools.utils.clear_directories()`.

---

## Tests y Verificación Rápida

### Self-check (sin pytest)
```bash
python scripts/self_check.py
```
Salida esperada: `Self-check passed: 1 episode executed.`

### Test de humo (pytest)
```bash
python -m pytest -k smoke -q
```
Valida que un episodio corre y produce un DataFrame no vacío.

---

## Extender el Framework

### Nuevo Agente
1. Crear archivo en `agents/` con sufijo `_agent.py`.
2. Decorar la clase con `@register_agent("nombre")`.
3. Implementar al menos: `update_power`. La recompensa se define vía YAML en `agents.<tipo>.reward` y se consume desde `core/rewards.py`. La Q-table se inicializa automáticamente desde `state_space`.
4. Añadir su bloque en `configs/default.yaml`.

### Personalizar Entorno
- Extender `MultiAgentEnv` para nuevos campos / transformaciones.
- (Futuro) Convertir a interfaz Gym / PettingZoo (`reset`, `step`).

### Espacio de Estado Declarativo
Cada entrada en `state_space` posee:
| Clave | Descripción |
|-------|-------------|
| var | nombre lógico de la variable |
| source | `local` / `env` / `global` / `self` / `external` |
| bins | Placeholder para validación futura |

---

## Recompensas

Las recompensas ahora se definen y consumen vía `core/rewards.py` según el YAML (`agents.<tipo>.reward`). No existe `calculate_reward` en los agentes.

---

## Refactor Reciente (Resumen)
- Docstrings en inglés añadidos a módulos núcleo y agentes.
- Seeding determinista (`core.utils.set_global_seed`).
- Logging estructurado (`core.utils.build_logger`).
- Entrada modular (`main.main`).
- Estructura preparada para tuning y futuros wrappers.

---

## Roadmap Sugerido
1. Argparse (`--config`, `--episodes`, `--epsilon-schedule`).
2. API estilo Gym / PettingZoo.
3. Callbacks (on_step, on_episode_end) y export incremental de métricas.
4. Persistencia / reanudación de Q‑tables.
5. Barridos automáticos (grid / random search) en `experiments/`.
6. Reporte de métricas en JSON para dashboards.

---

## Comandos Rápidos
```bash
# Instalar dependencias
python -m pip install -r requirements.txt

# Entrenamiento principal
python main.py

# Self-check
python scripts/self_check.py

# Test de humo
python -m pytest -k smoke -q

# Limpieza manual (opcional)
python -c "from analysis_tools.utils import clear_directories; clear_directories()"
```

---

## Notas
- Colocar datasets CSV en `assets/datasets/`.
- Paso temporal por defecto: 1 hora.
- Ajustar recompensas según objetivos energéticos reales.

---

## Licencia
Proyecto para fines de **investigación académica**. Por favor **citar apropiadamente** si se utiliza en publicaciones.

