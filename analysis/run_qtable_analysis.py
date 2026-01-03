"""Q-table analysis for MARL microgrid agents.

This script analyzes saved sparse Q-tables per agent from a checkpoint directory and
produces:
- Individual PNG plots (heatmaps, distributions, policy summaries)
- A consolidated HTML report with metrics, interpretations, and recommendations

Usage:
    1) Edita RUN_ID dentro de este archivo
    2) Ejecuta: python analysis/run_qtable_analysis.py

Opcionalmente, puedes sobreescribir por consola:
    python analysis/run_qtable_analysis.py --id <run_id>

It expects checkpoint folders like:
    results/checkpoints/<run_id>/qtable_<agent>#<i>.npz

Outputs are written to:
    results/checkpoints/<run_id>/analysis/

Notes:
- Q-tables are stored in sparse triplet form: (states, actions, values)
- This analysis is static (final checkpoint only) and does not use episode history.

Important:
- This script intentionally avoids importing pandas due to an environment-level
  Bus error observed when importing pandas in this workspace.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------
# Define aquí el checkpoint a analizar (carpeta results/checkpoints/<RUN_ID>/)
# Ejemplo: RUN_ID = "44"
RUN_ID = "44"


@dataclass(frozen=True)
class QTableSpec:
    agent: str
    instance: str
    path: Path


@dataclass
class QTableDense:
    states_unique: np.ndarray  # (S, dims)
    actions_unique: np.ndarray  # (A,)
    q_dense: np.ndarray  # (S, A) with NaN for missing entries


@dataclass
class QTableAnalysis:
    spec: QTableSpec
    dims: int
    n_entries: int
    n_unique_states: int
    n_actions: int
    coverage_states: Optional[float]
    completeness_states: float
    q_min: float
    q_max: float
    q_mean: float
    q_std: float
    greedy_entropy_mean: float
    greedy_gap_mean: float
    greedy_gap_p10: float
    greedy_gap_p50: float
    greedy_gap_p90: float
    greedy_action_counts: Dict[int, int]
    expected_match_rate: Optional[float]
    expected_match_note: str
    recommendations: List[str]
    dense: QTableDense


# -----------------------------
# Path & loading utilities
# -----------------------------

def _find_checkpoint_dir(run_id: str) -> Path:
    candidates = [
        PROJECT_ROOT / "results" / "checkpoints" / str(run_id),
        PROJECT_ROOT / "results" / "checkpoint" / str(run_id),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No se encontró el directorio del checkpoint para id={run_id}. "
        f"Probé: {', '.join(str(c) for c in candidates)}"
    )


def _load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _discover_qtables(checkpoint_dir: Path) -> List[QTableSpec]:
    qtables: List[QTableSpec] = []
    for path in sorted(checkpoint_dir.glob("qtable_*.npz")):
        name = path.name
        # Examples: qtable_battery#0.npz, qtable_grid#0.npz
        match = re.match(r"^qtable_(?P<agent>[a-zA-Z0-9_]+)(?P<instance>#\d+)?\.npz$", name)
        if not match:
            continue
        agent = match.group("agent")
        instance = (match.group("instance") or "#0").lstrip("#")
        qtables.append(QTableSpec(agent=agent, instance=instance, path=path))
    return qtables


def _load_sparse_qtable(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if not {"states", "actions", "values"}.issubset(set(data.keys())):
        raise ValueError(
            f"Formato inesperado en {path.name}. Keys={list(data.keys())}. "
            "Se esperaba: states, actions, values"
        )

    states = data["states"]
    actions = data["actions"]
    values = data["values"]

    if len(states) != len(actions) or len(actions) != len(values):
        raise ValueError(
            f"Dimensiones inconsistentes en {path.name}: "
            f"len(states)={len(states)}, len(actions)={len(actions)}, len(values)={len(values)}"
        )

    return states, actions, values


def _states_to_int_matrix(states: np.ndarray) -> np.ndarray:
    # states comes as object array of shape (N, dims), each entry is list/array/int
    if states.ndim != 2:
        raise ValueError(f"Se esperaba states con ndim=2, pero ndim={states.ndim}.")

    out = np.zeros((states.shape[0], states.shape[1]), dtype=int)
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            out[i, j] = int(states[i, j])
    return out


def _lexsort_rows(a: np.ndarray) -> np.ndarray:
    """Return indices that sort rows lexicographically."""
    if a.ndim != 2:
        raise ValueError("Se esperaba matriz 2D para ordenar filas")
    keys = [a[:, col] for col in range(a.shape[1] - 1, -1, -1)]
    return np.lexsort(keys)


def _unique_rows_sorted(a: np.ndarray) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("Se esperaba matriz 2D")
    if a.shape[0] == 0:
        return a.copy()
    idx = _lexsort_rows(a)
    sorted_a = a[idx]
    mask = np.ones(sorted_a.shape[0], dtype=bool)
    mask[1:] = np.any(sorted_a[1:] != sorted_a[:-1], axis=1)
    return sorted_a[mask]


def _build_dense_q(states_int: np.ndarray, actions: np.ndarray, values: np.ndarray) -> QTableDense:
    states_unique = _unique_rows_sorted(states_int)
    actions_unique = np.unique(actions.astype(int))
    actions_unique = np.sort(actions_unique)

    # map state tuple -> row index
    state_to_row: Dict[Tuple[int, ...], int] = {
        tuple(states_unique[i].tolist()): i for i in range(states_unique.shape[0])
    }
    action_to_col: Dict[int, int] = {int(a): i for i, a in enumerate(actions_unique.tolist())}

    q_dense = np.full((states_unique.shape[0], actions_unique.shape[0]), np.nan, dtype=float)

    for s, a, q in zip(states_int, actions.astype(int), values.astype(float)):
        r = state_to_row.get(tuple(s.tolist()))
        c = action_to_col.get(int(a))
        if r is None or c is None:
            continue
        # Keep max if duplicates exist
        if math.isnan(q_dense[r, c]) or q > q_dense[r, c]:
            q_dense[r, c] = float(q)

    return QTableDense(states_unique=states_unique, actions_unique=actions_unique, q_dense=q_dense)


def _state_labels(states_unique: np.ndarray) -> List[str]:
    labels: List[str] = []
    for row in states_unique:
        parts = [f"s{d}={int(v)}" for d, v in enumerate(row.tolist())]
        labels.append("(" + ", ".join(parts) + ")")
    return labels


# -----------------------------
# Domain expectations
# -----------------------------

def _agent_action_labels(agent: str, n_actions: int) -> Dict[int, str]:
    if agent in {"wind", "solar"}:
        # From core/rewards.py: action 1 rewarded when delta_ph>0, action 0 otherwise
        return {0: "curtail", 1: "inject"}
    if agent == "grid":
        return {0: "no_import", 1: "import"}
    if agent == "battery":
        # From core/rewards.py
        return {0: "idle", 1: "charge", 2: "discharge"}
    if agent == "load":
        return {0: "shed", 1: "on"}

    # Fallback
    return {i: f"a{i}" for i in range(n_actions)}


def _expected_action_for_state(agent: str, state: Tuple[int, ...], n_actions: int) -> Tuple[Optional[int], str]:
    """Heuristic expectation from reward definitions.

    Returns:
        (expected_action, note)

    Notes:
        - For battery/grid the rewards depend on real_balance (not delta_ph). The
          Q-table state uses delta_ph, so this expectation is an approximation.
    """

    if agent in {"wind", "solar"} and n_actions >= 2 and len(state) >= 2:
        delta_ph_idx, potential_idx = state[0], state[1]
        # Reward logic: for delta_ph>0, inject (action=1) is correct ONLY if potential_idx>0
        if delta_ph_idx > 0 and potential_idx > 0:
            return 1, "Basado en reward: excedente y potencial>0 → inyectar"
        if delta_ph_idx > 0 and potential_idx <= 0:
            return 0, "Basado en reward: excedente pero potencial<=0 → evitar inyectar"
        if delta_ph_idx < 0:
            return 0, "Basado en reward: déficit → curtail"
        return None, "Caso neutro en reward (delta_ph=0)"

    if agent == "grid" and n_actions >= 2 and len(state) >= 2:
        delta_ph_idx, soc_idx = state[0], state[1]
        if delta_ph_idx < 0 and soc_idx == 0:
            return 1, "Aprox.: déficit y SOC=0 → importar"
        return 0, "Aprox.: no déficit o SOC>0 → no importar"

    if agent == "battery" and n_actions >= 3 and len(state) >= 2:
        delta_ph_idx, soc_idx = state[0], state[1]
        if delta_ph_idx < 0 and soc_idx > 0:
            return 2, "Aprox.: déficit y SOC>0 → descargar"
        if delta_ph_idx > 0 and soc_idx < 2:
            return 1, "Aprox.: excedente y SOC no lleno → cargar"
        if delta_ph_idx == 0:
            return 0, "Aprox.: balanceado → idle"
        return None, "Caso mixto/ambiguo"

    return None, "Sin regla específica"


# -----------------------------
# Metrics
# -----------------------------

def _softmax_entropy(q_row: np.ndarray, temperature: float = 1.0) -> float:
    q = np.asarray(q_row, dtype=float)
    q = q[np.isfinite(q)]
    if q.size == 0:
        return float("nan")
    if q.size == 1:
        return 0.0

    t = max(1e-8, float(temperature))
    z = (q - np.max(q)) / t
    p = np.exp(z)
    p = p / np.sum(p)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent)


def _row_gap(q_row: np.ndarray) -> float:
    q = np.asarray(q_row, dtype=float)
    q = q[np.isfinite(q)]
    if q.size < 2:
        return float("nan")
    q_sorted = np.sort(q)
    return float(q_sorted[-1] - q_sorted[-2])


def _infer_total_states_from_config(config: dict, agent: str, dims: int) -> Optional[int]:
    agents_cfg = config.get("agents", {})
    agent_cfg = agents_cfg.get(agent, {})
    state_space = agent_cfg.get("state_space", [])

    if not state_space or len(state_space) != dims:
        return None

    disc = config.get("discretization", {})
    delta_bins = int(disc.get("delta_bins", 0) or 0)
    power_bins = int(disc.get("power_bins", 0) or 0)
    price_bins = int(disc.get("price_bins", 0) or 0)
    soc_bins = int(disc.get("soc_bins", 0) or 0)

    def bins_for_var(var_name: str) -> Optional[int]:
        if var_name in {"delta_ph"}:
            return delta_bins
        if var_name in {"soc"}:
            return soc_bins
        if var_name in {"price", "cm"}:
            return price_bins
        # Treat all power-like inputs as power_bins
        if var_name.endswith("potential") or var_name.endswith("power") or var_name in {"pu_power"}:
            return power_bins
        return None

    bins: List[int] = []
    for dim in state_space:
        var = dim.get("var")
        b = bins_for_var(str(var)) if var is not None else None
        if b is None or b <= 0:
            return None
        bins.append(int(b))

    total = 1
    for b in bins:
        total *= b

    return int(total)


def _analyze_single_qtable(config: dict, spec: QTableSpec, output_dir: Path) -> QTableAnalysis:
    states_raw, actions_raw, values_raw = _load_sparse_qtable(spec.path)
    states_int = _states_to_int_matrix(states_raw)

    dims = int(states_int.shape[1])
    n_entries = int(states_int.shape[0])

    dense = _build_dense_q(states_int, actions_raw, values_raw)
    n_unique_states = int(dense.states_unique.shape[0])
    n_actions = int(dense.actions_unique.shape[0])

    total_possible_states = _infer_total_states_from_config(config, spec.agent, dims)
    coverage_states = None
    if total_possible_states and total_possible_states > 0:
        coverage_states = float(n_unique_states) / float(total_possible_states)

    completeness_states = float(np.mean(np.isfinite(dense.q_dense).all(axis=1))) if n_unique_states > 0 else 0.0

    q_values = values_raw.astype(float)
    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    q_mean = float(np.mean(q_values))
    q_std = float(np.std(q_values))

    temp = max(1e-6, float(np.nanstd(dense.q_dense)))
    entropies = []
    gaps = []
    greedy_actions: List[int] = []
    greedy_action_idx: List[int] = []

    for i in range(n_unique_states):
        row = dense.q_dense[i, :]
        entropies.append(_softmax_entropy(row, temperature=temp))
        gaps.append(_row_gap(row))
        if np.isfinite(row).any():
            c = int(np.nanargmax(row))
            greedy_action_idx.append(c)
            greedy_actions.append(int(dense.actions_unique[c]))

    greedy_entropy_mean = float(np.nanmean(entropies)) if entropies else float("nan")
    gaps_arr = np.asarray(gaps, dtype=float)
    greedy_gap_mean = float(np.nanmean(gaps_arr)) if gaps else float("nan")
    greedy_gap_p10 = float(np.nanpercentile(gaps_arr, 10)) if gaps else float("nan")
    greedy_gap_p50 = float(np.nanpercentile(gaps_arr, 50)) if gaps else float("nan")
    greedy_gap_p90 = float(np.nanpercentile(gaps_arr, 90)) if gaps else float("nan")

    greedy_action_counts: Dict[int, int] = {}
    if greedy_actions:
        uniq, cnt = np.unique(np.asarray(greedy_actions, dtype=int), return_counts=True)
        greedy_action_counts = {int(u): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}

    # Expected action match rate (domain-specific/heuristic)
    expected_match: List[bool] = []
    for i in range(n_unique_states):
        state_tuple = tuple(int(x) for x in dense.states_unique[i, :].tolist())
        exp_action, _note = _expected_action_for_state(spec.agent, state_tuple, n_actions)
        if exp_action is None:
            continue
        # exp_action is expressed in "raw" action index (0/1/2). Map to our dense action space.
        if exp_action not in dense.actions_unique.tolist():
            continue
        greedy_raw = greedy_actions[i] if i < len(greedy_actions) else None
        if greedy_raw is None:
            continue
        expected_match.append(int(greedy_raw) == int(exp_action))

    expected_match_rate = None
    expected_match_note = ""
    if expected_match:
        expected_match_rate = float(np.mean(expected_match))
        if spec.agent in {"battery", "grid"}:
            expected_match_note = "Las recompensas usan real_balance (no delta_ph). Esta validación es aproximada."
        else:
            expected_match_note = "Validación directa basada en core/rewards.py"
    else:
        expected_match_note = "No aplica (no hubo estados con expectativa fuerte)"

    # Plots
    agent_tag = f"{spec.agent}#{spec.instance}"
    labels = _state_labels(dense.states_unique)
    _plot_q_heatmap(dense, labels, spec.agent, output_dir / f"q_heatmap_{agent_tag}.png")
    _plot_q_histogram(q_values, spec.agent, output_dir / f"q_hist_{agent_tag}.png")
    _plot_greedy_action_bar(
        greedy_action_counts,
        spec.agent,
        dense.actions_unique,
        output_dir / f"greedy_actions_{agent_tag}.png",
    )

    if expected_match_rate is not None:
        _plot_expected_match(expected_match_rate, spec.agent, output_dir / f"expected_match_{agent_tag}.png")

    recommendations = _recommendations_for_agent(
        agent=spec.agent,
        coverage_states=coverage_states,
        completeness_states=completeness_states,
        q_min=q_min,
        q_max=q_max,
        q_std=q_std,
        greedy_entropy_mean=greedy_entropy_mean,
        greedy_gap_p50=greedy_gap_p50,
        expected_match_rate=expected_match_rate,
    )

    return QTableAnalysis(
        spec=spec,
        dims=dims,
        n_entries=n_entries,
        n_unique_states=n_unique_states,
        n_actions=n_actions,
        coverage_states=coverage_states,
        completeness_states=float(completeness_states),
        q_min=q_min,
        q_max=q_max,
        q_mean=q_mean,
        q_std=q_std,
        greedy_entropy_mean=float(greedy_entropy_mean),
        greedy_gap_mean=float(greedy_gap_mean),
        greedy_gap_p10=float(greedy_gap_p10),
        greedy_gap_p50=float(greedy_gap_p50),
        greedy_gap_p90=float(greedy_gap_p90),
        greedy_action_counts=greedy_action_counts,
        expected_match_rate=expected_match_rate,
        expected_match_note=expected_match_note,
        recommendations=recommendations,
        dense=dense,
    )


# -----------------------------
# Plotting
# -----------------------------

def _plot_q_heatmap(dense: QTableDense, state_labels: List[str], agent: str, out_path: Path) -> None:
    plt.figure(figsize=(10, max(3.5, 0.35 * len(state_labels))))
    arr = dense.q_dense

    finite = arr[np.isfinite(arr)]
    fill = (float(np.min(finite)) - 1.0) if finite.size else 0.0
    plot_arr = np.where(np.isfinite(arr), arr, fill)

    plt.imshow(plot_arr, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Q(s,a)")
    plt.yticks(ticks=np.arange(len(state_labels)), labels=state_labels, fontsize=7)
    plt.xticks(
        ticks=np.arange(dense.actions_unique.shape[0]),
        labels=[f"a{int(a)}" for a in dense.actions_unique.tolist()],
    )
    plt.title(f"Heatmap Q(s,a) - {agent}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_q_histogram(q_values: np.ndarray, agent: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(q_values, bins=30, color="steelblue", alpha=0.85, edgecolor="black")
    plt.title(f"Distribución de Q-values - {agent}")
    plt.xlabel("Q")
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_greedy_action_bar(action_counts: Dict[int, int], agent: str, actions_unique: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    actions = [int(a) for a in actions_unique.tolist()]
    xs = list(range(len(actions)))
    ys = [action_counts.get(a, 0) for a in actions]

    labels = _agent_action_labels(agent, len(actions))
    xt = [labels.get(a, f"a{a}") for a in actions]

    plt.bar(xs, ys, color="darkorange", alpha=0.85)
    plt.xticks(xs, xt, rotation=0)
    plt.title(f"Acción greedy por estado - {agent}")
    plt.xlabel("Acción")
    plt.ylabel("# estados")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_expected_match(match_rate: float, agent: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.bar([0], [match_rate], color="seagreen" if match_rate >= 0.75 else "firebrick", alpha=0.9)
    plt.ylim(0, 1)
    plt.xticks([0], [agent])
    plt.ylabel("Tasa de coincidencia")
    plt.title("Coherencia con comportamiento esperado")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Recommendations
# -----------------------------

def _recommendations_for_agent(
    agent: str,
    coverage_states: Optional[float],
    completeness_states: float,
    q_min: float,
    q_max: float,
    q_std: float,
    greedy_entropy_mean: float,
    greedy_gap_p50: float,
    expected_match_rate: Optional[float],
) -> List[str]:
    recs: List[str] = []

    if coverage_states is not None:
        if coverage_states < 0.5:
            recs.append(
                "Cobertura baja de estados: sugiere exploración insuficiente. "
                "Recomendación: aumentar episodios, usar un epsilon mínimo > 0 (p.ej., 0.05), "
                "o desacelerar el decaimiento de epsilon."
            )
        elif coverage_states < 0.8:
            recs.append(
                "Cobertura moderada de estados. Recomendación: revisar si el dataset/ventana de episodio "
                "explora suficientes condiciones (demand/potenciales/precios) o aumentar episode_window_hours."
            )

    if completeness_states < 0.8:
        recs.append(
            "Muchos estados no tienen Q-values para todas las acciones (tabla incompleta). "
            "Recomendación: incentivar exploración de acciones raras (optimistic init ya ayuda; también sirve ε mínimo)."
        )

    q_range = q_max - q_min
    if q_range > 50 or q_std > 15:
        recs.append(
            "Rango/varianza alta de Q-values: posible inestabilidad por escala de recompensas. "
            "Recomendación: normalizar/recortar recompensas (reward clipping) o ajustar pesos en reward params." 
        )

    if not math.isnan(greedy_entropy_mean) and greedy_entropy_mean < 0.15 and (coverage_states is None or coverage_states < 0.8):
        recs.append(
            "Política muy determinista con cobertura incompleta: riesgo de convergencia prematura a una política miope. "
            "Recomendación: mantener exploración al final (epsilon_end > 0) o reducir alpha."
        )

    if expected_match_rate is not None:
        if expected_match_rate < 0.6:
            recs.append(
                "La política greedy no coincide con el comportamiento esperado por la recompensa. "
                "Recomendación: revisar (1) definición de estado vs reward (posible desalineación), "
                "(2) escala de recompensas, (3) gamma demasiado alto si induce acciones oportunistas." 
            )
        elif expected_match_rate < 0.8:
            recs.append(
                "Coherencia parcial con la recompensa. Recomendación: aumentar episodios o suavizar actualización (alpha)."
            )

    # Agent-specific prescriptive ideas
    if agent in {"wind", "solar"}:
        recs.append(
            "En MARL estigmérgico, revisa que 'delta_ph' se actualice correctamente entre agentes y que el orden de ejecución "
            "no genere sesgo sistemático (p.ej., un renovable siempre 'come' primero)."
        )

    if agent in {"battery", "grid"}:
        recs.append(
            "Nota: la recompensa usa real_balance, pero el estado de la Q-table usa delta_ph. Si observas incoherencias, "
            "una mejora fuerte es alinear el estado con real_balance (y/o incluirlo como variable adicional)."
        )

    if not recs:
        recs.append("No se detectaron señales fuertes de problemas; el comportamiento parece consistente con el checkpoint actual.")

    return recs


# -----------------------------
# Multi-agent analysis
# -----------------------------

def _multi_agent_delta_summary(analyses: List[QTableAnalysis], checkpoint_dir: Path) -> Tuple[List[dict], Optional[Path]]:
    """Aggregate greedy actions conditioned on delta_ph_idx (s0).

    This is heuristic but useful for coordination/conflict patterns.
    """

    rows: List[dict] = []

    for a in analyses:
        tag = f"{a.spec.agent}#{a.spec.instance}"
        dense = a.dense
        if dense.states_unique.shape[0] == 0:
            continue

        # Greedy action in raw action-space
        greedy_col = np.nanargmax(dense.q_dense, axis=1)
        greedy_raw = dense.actions_unique[greedy_col]
        delta = dense.states_unique[:, 0].astype(int)

        for d in sorted(set(delta.tolist())):
            mask = delta == d
            acts = greedy_raw[mask].astype(int)
            total = int(acts.size)
            if total == 0:
                continue
            uniq, cnt = np.unique(acts, return_counts=True)
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                rows.append(
                    {
                        "agent": tag,
                        "delta_ph_idx": int(d),
                        "action": int(u),
                        "count": int(c),
                        "share": float(c) / float(total),
                    }
                )

    if not rows:
        return rows, None

    out_path = checkpoint_dir / "analysis" / "multiagent_delta_actions.png"
    _plot_multiagent_delta_actions(rows, out_path)
    return rows, out_path


def _plot_multiagent_delta_actions(rows: List[dict], out_path: Path) -> None:
    agents = sorted(set(r["agent"] for r in rows))
    deltas = sorted(set(int(r["delta_ph_idx"]) for r in rows))
    actions = sorted(set(int(r["action"]) for r in rows))

    fig, axes = plt.subplots(nrows=len(agents), ncols=1, figsize=(10, 3.2 * len(agents)), sharex=True)
    if len(agents) == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        sub = [r for r in rows if r["agent"] == agent]
        bottom = np.zeros(len(deltas), dtype=float)
        for act in actions:
            shares = []
            for d in deltas:
                s = [r["share"] for r in sub if int(r["delta_ph_idx"]) == int(d) and int(r["action"]) == int(act)]
                shares.append(float(s[0]) if len(s) else 0.0)
            ax.bar([str(d) for d in deltas], shares, bottom=bottom, label=f"a{act}", alpha=0.9)
            bottom += np.asarray(shares)
        ax.set_title(f"{agent}: distribución de acción greedy por delta_ph_idx")
        ax.set_ylabel("Proporción")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("delta_ph_idx")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# HTML report
# -----------------------------

def _health_color(score: float) -> str:
    if score >= 0.8:
        return "good"
    if score >= 0.6:
        return "warn"
    return "bad"


def _build_executive_summary(analyses: List[QTableAnalysis]) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    findings: List[str] = []

    for a in analyses:
        score_parts = []
        if a.coverage_states is not None:
            score_parts.append(a.coverage_states)
        if a.expected_match_rate is not None:
            score_parts.append(a.expected_match_rate)
        score = float(np.mean(score_parts)) if score_parts else 0.7

        rows.append({
            "Agente": f"{a.spec.agent}#{a.spec.instance}",
            "Estados únicos": a.n_unique_states,
            "Acciones": a.n_actions,
            "Cobertura estados": ("{:.1%}".format(a.coverage_states) if a.coverage_states is not None else "N/A"),
            "Completitud (estados con todas las acciones)": "{:.1%}".format(a.completeness_states),
            "Entropía media (softmax(Q))": "{:.3f}".format(a.greedy_entropy_mean),
            "Gap greedy p50": "{:.3f}".format(a.greedy_gap_p50),
            "Coherencia con reward": ("{:.1%}".format(a.expected_match_rate) if a.expected_match_rate is not None else "N/A"),
            "Salud": _health_color(score),
        })

        if a.coverage_states is not None and a.coverage_states < 0.6:
            findings.append(f"{a.spec.agent}#{a.spec.instance}: cobertura de estados baja ({a.coverage_states:.1%}).")
        if a.expected_match_rate is not None and a.expected_match_rate < 0.7:
            findings.append(
                f"{a.spec.agent}#{a.spec.instance}: política no alinea bien con reward ({a.expected_match_rate:.1%})."
            )

    if not findings:
        findings.append("No se detectaron alertas fuertes en cobertura/coherencia para este checkpoint.")

    return rows, findings[:3]


def _write_html_report(
    run_id: str,
    checkpoint_dir: Path,
    output_dir: Path,
    analyses: List[QTableAnalysis],
    multiagent_rows: List[dict],
    multiagent_plot: Optional[Path],
) -> Path:
    summary_rows, top_findings = _build_executive_summary(analyses)

    # Save machine-readable summary
    summary_csv = output_dir / "qtable_metrics_summary.csv"
    _write_csv(summary_csv, summary_rows)

    # Build HTML
    def img(rel_path: str, alt: str) -> str:
        return f'<img src="{rel_path}" alt="{alt}" style="max-width: 100%; height: auto; border: 1px solid #ddd;" />'

    css = """
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2, h3 { margin-bottom: 8px; }
    .muted { color: #555; }
    .card { border: 1px solid #e5e5e5; border-radius: 8px; padding: 14px; margin: 12px 0; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }
    th { background: #f6f6f6; text-align: left; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .good { background: #e7f7ee; color: #0f5132; }
    .warn { background: #fff4e5; color: #7a4b00; }
    .bad { background: #fde8e8; color: #842029; }
    details summary { cursor: pointer; font-weight: bold; }
    .bullet { margin: 6px 0 6px 16px; }
    """

    # Render summary table with colored tags
    summary_html_rows: List[dict] = []
    for r in summary_rows:
        health = str(r.get("Salud", "warn"))
        tag = f'<span class="tag {health}">{health.upper()}</span>'
        rr = dict(r)
        rr["Salud"] = tag
        summary_html_rows.append(rr)

    summary_html = _html_table(summary_html_rows, escape=False)

    # Per-agent sections
    per_agent_html = []
    for a in analyses:
        tag = f"{a.spec.agent}#{a.spec.instance}"

        heatmap = f"q_heatmap_{tag}.png"
        hist = f"q_hist_{tag}.png"
        actions = f"greedy_actions_{tag}.png"
        exp = f"expected_match_{tag}.png"

        recs = "".join([f"<li>{r}</li>" for r in a.recommendations])

        per_agent_html.append(
            f"""
            <details class="card">
              <summary>{tag}</summary>
              <p class="muted">Archivo: <code>{a.spec.path.name}</code></p>
              <div class="grid">
                <div class="card">
                  <h3>Métricas</h3>
                  <ul class="bullet">
                    <li>Estados únicos: {a.n_unique_states}</li>
                    <li>Acciones: {a.n_actions}</li>
                    <li>Cobertura (según config): {('{:.1%}'.format(a.coverage_states) if a.coverage_states is not None else 'N/A')}</li>
                    <li>Completitud (estados con todas las acciones): {a.completeness_states:.1%}</li>
                    <li>Q min/max/mean/std: {a.q_min:.3f} / {a.q_max:.3f} / {a.q_mean:.3f} / {a.q_std:.3f}</li>
                    <li>Entropía media (softmax(Q)): {a.greedy_entropy_mean:.3f}</li>
                    <li>Gap greedy (p10/p50/p90): {a.greedy_gap_p10:.3f} / {a.greedy_gap_p50:.3f} / {a.greedy_gap_p90:.3f}</li>
                    <li>Coherencia con reward: {('{:.1%}'.format(a.expected_match_rate) if a.expected_match_rate is not None else 'N/A')}<br/><span class="muted">{a.expected_match_note}</span></li>
                  </ul>
                </div>

                <div class="card">
                  <h3>Gráficas</h3>
                  <p>{img(heatmap, f"Heatmap {tag}")}</p>
                  <p>{img(hist, f"Hist {tag}")}</p>
                  <p>{img(actions, f"Actions {tag}")}</p>
                  {f'<p>{img(exp, f"Expected {tag}")}</p>' if (output_dir / exp).exists() else ''}
                </div>

                <div class="card">
                  <h3>Recomendaciones prescriptivas (MARL)</h3>
                  <ul class="bullet">{recs}</ul>
                </div>
              </div>
            </details>
            """
        )

    multiagent_html = ""
    if multiagent_plot is not None and multiagent_plot.exists():
        multiagent_rel = multiagent_plot.name
        multiagent_html = f"""
        <div class="card">
          <h2>Análisis multi-agente</h2>
          <p class="muted">Distribución de acción greedy condicionada en <code>delta_ph_idx</code> (heurístico).</p>
          <p>{img(multiagent_rel, "Multi-agent delta actions")}</p>
        </div>
        """

    findings_html = "".join([f"<li>{f}</li>" for f in top_findings])

    html = f"""<!doctype html>
    <html lang="es">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Q-table Analysis - checkpoint {run_id}</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>Reporte de análisis de Q-tables</h1>
      <p class="muted">Checkpoint: <code>{checkpoint_dir}</code></p>
      <div class="card">
        <h2>Resumen ejecutivo</h2>
        <ul class="bullet">{findings_html}</ul>
        <p class="muted">Archivo de métricas: <code>{summary_csv.name}</code></p>
        {summary_html}
      </div>

      {multiagent_html}

      <div class="card">
        <h2>Detalle por agente</h2>
        {''.join(per_agent_html)}
      </div>

      <div class="card">
        <h2>Notas de interpretación</h2>
        <ul class="bullet">
          <li>Este análisis es estático (Q-tables finales). Para convergencia temporal se requiere historial por episodio.</li>
          <li>Para battery/grid, la recompensa usa <code>real_balance</code>, pero el estado actual usa <code>delta_ph</code>; por eso algunas validaciones son aproximadas.</li>
          <li>En MARL estigmérgico (delta_ph), el orden de ejecución y el consumo de potencial renovable pueden inducir sesgos en políticas aprendidas.</li>
        </ul>
      </div>

    </body>
    </html>"""

    out_file = output_dir / "qtable_report.html"
    out_file.write_text(html, encoding="utf-8")
    return out_file


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            s = str(v)
            # naive CSV escaping
            if "," in s or "\n" in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _html_table(rows: List[dict], escape: bool = True) -> str:
    if not rows:
        return "<p class=\"muted\">(sin datos)</p>"
    headers = list(rows[0].keys())
    th = "".join([f"<th>{html.escape(str(h))}</th>" for h in headers])
    body_rows = []
    for r in rows:
        tds = []
        for h in headers:
            v = r.get(h, "")
            if escape:
                tds.append(f"<td>{html.escape(str(v))}</td>")
            else:
                tds.append(f"<td>{str(v)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    return "<table><thead><tr>" + th + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Analiza Q-tables por agente desde un checkpoint")
    parser.add_argument(
        "--id",
        default=None,
        help="ID del checkpoint (results/checkpoints/<id>). Si se omite, usa RUN_ID del script.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Ruta al YAML de configuración (default: configs/default.yaml)",
    )

    args = parser.parse_args()

    run_id = str(args.id) if args.id is not None else str(RUN_ID)
    checkpoint_dir = _find_checkpoint_dir(run_id)

    config_path = Path(args.config)
    config = _load_config(config_path)

    output_dir = checkpoint_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    qtables = _discover_qtables(checkpoint_dir)
    if not qtables:
        raise FileNotFoundError(f"No se encontraron archivos qtable_*.npz en {checkpoint_dir}")

    analyses: List[QTableAnalysis] = []
    for spec in qtables:
        analyses.append(_analyze_single_qtable(config, spec, output_dir))

    multi_rows, multi_plot = _multi_agent_delta_summary(analyses, checkpoint_dir)

    report_path = _write_html_report(run_id, checkpoint_dir, output_dir, analyses, multi_rows, multi_plot)

    print(f"✅ Reporte generado: {report_path}")
    print(f"✅ Gráficas y métricas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
