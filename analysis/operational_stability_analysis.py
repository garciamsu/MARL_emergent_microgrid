"""Operational Stability & Convergence Analysis (Post-Training).

This module adds stability/convergence analyses that are theoretically valid
for tertiary-level (hourly) microgrid control under Independent Q-Learning (IQL),
without assuming strict convergence.

IMPORTANT CONSTRAINTS
---------------------
- Post-training only: reads existing CSV outputs from results/evolution.
- Does NOT modify training loops, agents, rewards, or environment code.
- Uses only NumPy, Pandas, Matplotlib (+ Python stdlib).
- Writes outputs to results/stability/ using core.csv_handler.write_result_csv.

Available signals (per episode CSV):
- env_energy_balance: e(t) = PH(t) + P_U(t) + P_bat(t) - PL(t)
- action_<agent>#<i>: discrete actions chosen by each agent copy

Analyses implemented here (missing methods 3–6):
- EnergyBoundednessAnalyzer
- VariabilityStabilityAnalyzer
- ResilienceAnalyzer
- PolicyStabilityAnalyzer

Notes on parameters (interpretation is embedded in each class docstring):
- epsilon_band_w: stability band threshold ε in power units (e.g., W).
- event_threshold_w: disturbance trigger threshold (recommended: 2*epsilon_band_w).
- persistence_window_steps: number of consecutive steps within band required
  to declare recovery (recommended: 6 for hourly dt).
- rolling_window_steps: optional rolling-window size for variability.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.csv_handler import read_result_csv, write_result_csv

# Reuse existing (already-implemented) learning stability analyzers (methods 1–2)
from analysis.collect_qtables_per_episode import load_qtables_history
from analysis.stability_analysis import (
    BellmanContractionStabilityAnalyzer as _BellmanContractionStabilityAnalyzer,
    ConsensusStabilityAnalyzer as _ConsensusStabilityAnalyzer,
)


RESULTS_DIR_DEFAULT = Path("results/stability")
EVOLUTION_DIR_DEFAULT = Path("results/evolution")


@dataclass(frozen=True)
class EpisodeData:
    """Container for one episode time series."""

    episode: int
    df: pd.DataFrame


def _iter_episode_files(evolution_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (episode_idx, path) for episode CSVs."""

    files: List[Tuple[int, Path]] = []
    for p in evolution_dir.glob("episode_*.csv"):
        name = p.name
        if name.startswith(".~lock."):
            continue
        if name == "episodes_consolidated.csv":
            continue
        # Parse episode_<k>.csv
        try:
            k_str = name.replace("episode_", "").replace(".csv", "")
            k = int(k_str)
        except Exception:
            continue
        files.append((k, p))

    files.sort(key=lambda x: x[0])
    return files


def load_episodes(
    evolution_dir: Path = EVOLUTION_DIR_DEFAULT,
    max_episodes: Optional[int] = None,
) -> List[EpisodeData]:
    """Load per-episode CSV outputs as EpisodeData.

    Args:
        evolution_dir: Directory containing episode CSVs.
        max_episodes: Optional cap for faster debug runs.

    Returns:
        List of EpisodeData sorted by episode index.
    """

    episode_files = _iter_episode_files(evolution_dir)
    if max_episodes is not None:
        episode_files = episode_files[: max_episodes]

    episodes: List[EpisodeData] = []
    for k, path in episode_files:
        df = read_result_csv(path)
        episodes.append(EpisodeData(episode=k, df=df))
    return episodes


def _clean_episode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-physical rows and ensure numeric series are usable.

    The episode CSVs contain an initialization row with step == -1.
    This row often has missing values and must be excluded from any
    time-series stability analysis.
    """

    out = df.copy()
    if "step" in out.columns:
        out = out[out["step"].astype(float) >= 0]

    # Ensure energy balance is numeric where available.
    if "env_energy_balance" in out.columns:
        out["env_energy_balance"] = pd.to_numeric(out["env_energy_balance"], errors="coerce")

    return out


def _detect_active_action_columns(df: pd.DataFrame) -> List[str]:
    """Detect action columns for active agents.

    Rationale:
    - User requested policy stability only for agents with count != 0.
    - In results/evolution, agents with count == 0 typically have no action columns.

    We detect columns that match: action_<name>#<idx>.
    """

    cols = [c for c in df.columns if c.startswith("action_")]
    # Keep only columns that have at least 2 non-NaN values (needed for switch rate).
    active: List[str] = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if int(s.notna().sum()) >= 2:
            active.append(c)
    return sorted(active)


class EnergyBoundednessAnalyzer:
    """Energy Balance Boundedness (Operational Stability).

    Given the microgrid power balance error:
        e(t) = PH(t) + P_U(t) + P_bat(t) - PL(t)

    In this codebase, `env_energy_balance` is the recorded e(t).

    Metrics (per episode):
    - max_abs_e: max_t |e(t)| (worst-case imbalance)
    - p95_abs_e: 95th percentile of |e(t)| (typical worst-case)
    - time_in_band_ratio: fraction of steps where |e(t)| <= ε

    Parameters:
    - epsilon_band_w (ε): power band threshold used to define acceptable
      operational balance. Interpretation: higher time_in_band_ratio implies
      tighter operational balance.

    Interpretation:
    - Bounded max_abs_e and high time-in-band indicate stable operation.
    - This does not require e(t) → 0, only practical boundedness.
    """

    def __init__(
        self,
        epsilon_band_w: float,
        results_dir: Path = RESULTS_DIR_DEFAULT,
    ) -> None:
        self.epsilon_band_w = float(epsilon_band_w)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, episodes: Sequence[EpisodeData]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for ep in episodes:
            df = _clean_episode_df(ep.df)
            if "env_energy_balance" not in df.columns:
                raise KeyError("Missing required column: env_energy_balance")

            e = df["env_energy_balance"].to_numpy(dtype=float)
            e = e[np.isfinite(e)]
            if e.size == 0:
                continue

            abs_e = np.abs(e)
            rows.append(
                {
                    "episode": ep.episode,
                    "max_abs_e": float(np.max(abs_e)),
                    "p95_abs_e": float(np.percentile(abs_e, 95)),
                    "time_in_band_ratio": float(np.mean(abs_e <= self.epsilon_band_w)),
                    "epsilon_band_w": float(self.epsilon_band_w),
                }
            )

        return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)

    def save_results(self, df: pd.DataFrame, filename: str = "energy_boundedness.csv") -> Path:
        out = self.results_dir / filename
        write_result_csv(df, out)
        return out


class VariabilityStabilityAnalyzer:
    """Variability-Based Stability (Operational Smoothness).

    This complements integral metrics (IAE/ISE) by directly measuring
    variability of the balance error e(t) within each episode.

    Metrics (per episode):
    - mean_e: μ_e
    - var_e:  σ_e^2 (population variance)
    - std_e:  σ_e
    - rms_e:  RMS_e = sqrt(mean(e(t)^2))

    Optional rolling-window metrics:
    - rolling_std_mean: mean of rolling std over the episode
    - rolling_std_max: max of rolling std over the episode

    Parameters:
    - rolling_window_steps: rolling window length in steps (hours).

    Interpretation:
    - Lower σ_e and RMS_e indicate smoother, more stable operation.
    - This is compatible with non-convergent MARL: we care about
      boundedness and variability reduction, not strict convergence.
    """

    def __init__(
        self,
        rolling_window_steps: Optional[int] = 12,
        results_dir: Path = RESULTS_DIR_DEFAULT,
    ) -> None:
        self.rolling_window_steps = None if rolling_window_steps is None else int(rolling_window_steps)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, episodes: Sequence[EpisodeData]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for ep in episodes:
            df = _clean_episode_df(ep.df)
            if "env_energy_balance" not in df.columns:
                raise KeyError("Missing required column: env_energy_balance")

            e = df["env_energy_balance"].to_numpy(dtype=float)
            e = e[np.isfinite(e)]
            if e.size == 0:
                continue

            mean_e = float(np.mean(e))
            var_e = float(np.var(e))
            std_e = float(np.std(e))
            rms_e = float(np.sqrt(np.mean(e ** 2)))

            row: Dict[str, float] = {
                "episode": ep.episode,
                "mean_e": mean_e,
                "var_e": var_e,
                "std_e": std_e,
                "rms_e": rms_e,
            }

            if self.rolling_window_steps is not None and self.rolling_window_steps >= 2:
                s = pd.Series(e)
                rolling_std = s.rolling(self.rolling_window_steps, min_periods=self.rolling_window_steps).std()
                rolling_std = rolling_std.to_numpy(dtype=float)
                rolling_std = rolling_std[np.isfinite(rolling_std)]
                if rolling_std.size > 0:
                    row["rolling_window_steps"] = float(self.rolling_window_steps)
                    row["rolling_std_mean"] = float(np.mean(rolling_std))
                    row["rolling_std_max"] = float(np.max(rolling_std))

            rows.append(row)

        return pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)

    def save_results(self, df: pd.DataFrame, filename: str = "variability_stability.csv") -> Path:
        out = self.results_dir / filename
        write_result_csv(df, out)
        return out

    def plot(self, df: pd.DataFrame, filename_prefix: str = "variability") -> List[Path]:
        """Create required plots: σ_e vs episodes and RMS_e vs episodes."""

        out_paths: List[Path] = []
        if df.empty:
            return out_paths

        # σ_e vs episode
        p1 = self.results_dir / f"{filename_prefix}_std_vs_episode.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["episode"], df["std_e"], linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(r"$\sigma_e$ (Std of energy balance error)")
        ax.set_title("Variability-Based Stability: Standard Deviation vs Episode")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p1)

        # RMS_e vs episode
        p2 = self.results_dir / f"{filename_prefix}_rms_vs_episode.png"
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["episode"], df["rms_e"], linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(r"$RMS_e$ (Root-mean-square error)")
        ax.set_title("Variability-Based Stability: RMS Error vs Episode")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p2)

        return out_paths


class ResilienceAnalyzer:
    """Resilience / Recovery Time after disturbances.

    Disturbance detection (default):
    - An event starts at time t when |e(t)| > event_threshold_w and
      the previous step was not in event.

    Recovery definition:
    - Recovery time t_r is the first time index >= t_event such that
      |e(t)| <= epsilon_band_w for `persistence_window_steps` consecutive steps.

    Parameters:
    - epsilon_band_w (ε): defines the acceptable operational band.
    - event_threshold_w: disturbance trigger threshold (recommended: 2*ε).
    - persistence_window_steps: consecutive in-band steps required to
      consider recovery achieved (recommended: 6 for hourly dt).
    - dt_h: hours per step. Used to convert recovery steps to hours.

    Interpretation:
    - Shorter recovery times imply higher operational resilience.
    - Systems may not always recover within an episode (reported as NaN).
    """

    def __init__(
        self,
        epsilon_band_w: float,
        event_threshold_w: float,
        persistence_window_steps: int = 6,
        dt_h: float = 1.0,
        results_dir: Path = RESULTS_DIR_DEFAULT,
    ) -> None:
        self.epsilon_band_w = float(epsilon_band_w)
        self.event_threshold_w = float(event_threshold_w)
        self.persistence_window_steps = int(persistence_window_steps)
        self.dt_h = float(dt_h)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _find_recovery_index(self, abs_e: np.ndarray, start_idx: int) -> Optional[int]:
        """Return first index where persistence window is fully within band."""

        w = self.persistence_window_steps
        if w <= 0:
            raise ValueError("persistence_window_steps must be positive")

        # Need at least w samples starting at candidate idx.
        for i in range(start_idx, max(0, abs_e.size - w + 1)):
            if np.all(abs_e[i : i + w] <= self.epsilon_band_w):
                return i
        return None

    def analyze(self, episodes: Sequence[EpisodeData]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (events_df, per_episode_summary_df)."""

        events_rows: List[Dict[str, float]] = []
        summary_rows: List[Dict[str, float]] = []

        for ep in episodes:
            df = _clean_episode_df(ep.df)
            if "env_energy_balance" not in df.columns:
                raise KeyError("Missing required column: env_energy_balance")

            e = df["env_energy_balance"].to_numpy(dtype=float)
            steps = df["step"].to_numpy(dtype=float) if "step" in df.columns else np.arange(len(e), dtype=float)

            mask = np.isfinite(e)
            e = e[mask]
            steps = steps[mask]
            if e.size == 0:
                continue

            abs_e = np.abs(e)
            in_event = False
            event_idx = 0
            recovered_times_steps: List[float] = []

            for t in range(abs_e.size):
                if (not in_event) and (abs_e[t] > self.event_threshold_w):
                    in_event = True
                    event_start_t = t
                    event_start_step = float(steps[t])

                    recovery_t = self._find_recovery_index(abs_e, start_idx=t)
                    if recovery_t is None:
                        recovery_step = np.nan
                        recovery_time_steps = np.nan
                        recovery_time_h = np.nan
                    else:
                        recovery_step = float(steps[recovery_t])
                        recovery_time_steps = float(recovery_t - event_start_t)
                        recovery_time_h = float(recovery_time_steps * self.dt_h)
                        recovered_times_steps.append(recovery_time_steps)

                    peak_abs_e = float(np.max(abs_e[event_start_t:]))

                    events_rows.append(
                        {
                            "episode": ep.episode,
                            "event_index": event_idx,
                            "event_start_step": event_start_step,
                            "event_start_t_idx": float(event_start_t),
                            "epsilon_band_w": float(self.epsilon_band_w),
                            "event_threshold_w": float(self.event_threshold_w),
                            "persistence_window_steps": float(self.persistence_window_steps),
                            "recovery_step": recovery_step,
                            "recovery_time_steps": recovery_time_steps,
                            "recovery_time_hours": recovery_time_h,
                            "peak_abs_e_after_event": peak_abs_e,
                        }
                    )
                    event_idx += 1

                # Exit event once back within event threshold (hysteresis-free)
                if in_event and (abs_e[t] <= self.event_threshold_w):
                    in_event = False

            num_events = float(event_idx)
            num_recovered = float(np.sum(np.isfinite(recovered_times_steps)))
            mean_recovery_steps = float(np.mean(recovered_times_steps)) if recovered_times_steps else np.nan
            median_recovery_steps = float(np.median(recovered_times_steps)) if recovered_times_steps else np.nan

            summary_rows.append(
                {
                    "episode": ep.episode,
                    "num_events": num_events,
                    "num_recovered": num_recovered,
                    "recovered_ratio": float(num_recovered / num_events) if num_events > 0 else np.nan,
                    "mean_recovery_time_steps": mean_recovery_steps,
                    "median_recovery_time_steps": median_recovery_steps,
                    "mean_recovery_time_hours": float(mean_recovery_steps * self.dt_h) if np.isfinite(mean_recovery_steps) else np.nan,
                    "median_recovery_time_hours": float(median_recovery_steps * self.dt_h) if np.isfinite(median_recovery_steps) else np.nan,
                }
            )

        events_df = pd.DataFrame(events_rows)
        summary_df = pd.DataFrame(summary_rows).sort_values("episode").reset_index(drop=True)
        return events_df, summary_df

    def save_results(
        self,
        events_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        events_filename: str = "resilience_recovery_events.csv",
        summary_filename: str = "resilience_recovery_summary.csv",
    ) -> Tuple[Path, Path]:
        p_events = self.results_dir / events_filename
        p_summary = self.results_dir / summary_filename
        write_result_csv(events_df, p_events)
        write_result_csv(summary_df, p_summary)
        return p_events, p_summary


class PolicyStabilityAnalyzer:
    """Policy Stability via action switch rate.

    For each active agent action series a(t), define:
        SR = (# of action changes) / (T - 1)

    Where T is the number of valid time steps.

    Parameters:
    - action_columns: optional explicit list of action columns to analyze.
      If None, action columns are auto-detected per episode.

    Interpretation:
    - High SR indicates oscillatory control (frequent switching).
    - Stable policies typically exhibit reduced SR over training.

    Notes:
    - User requested analysis only for agents with count != 0; in practice
      this corresponds to action columns present in the episode CSVs.
    """

    def __init__(self, results_dir: Path = RESULTS_DIR_DEFAULT) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _switch_rate(actions: np.ndarray) -> float:
        if actions.size < 2:
            return np.nan
        changes = np.sum(actions[1:] != actions[:-1])
        return float(changes / float(actions.size - 1))

    def analyze(
        self,
        episodes: Sequence[EpisodeData],
        action_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []

        for ep in episodes:
            df = _clean_episode_df(ep.df)

            cols = list(action_columns) if action_columns is not None else _detect_active_action_columns(df)
            if not cols:
                continue

            for c in cols:
                s = pd.to_numeric(df[c], errors="coerce")
                s = s[s.notna()]
                if s.size < 2:
                    continue
                a = s.to_numpy(dtype=int)
                sr = self._switch_rate(a)
                rows.append({"episode": ep.episode, "action_column": c, "switch_rate": sr})

        return pd.DataFrame(rows).sort_values(["action_column", "episode"]).reset_index(drop=True)

    def save_results(self, df: pd.DataFrame, filename: str = "policy_switch_rate.csv") -> Path:
        out = self.results_dir / filename
        write_result_csv(df, out)
        return out

    def plot(self, df: pd.DataFrame, filename: str = "policy_switch_rate_vs_episode.png") -> Optional[Path]:
        """Create required plot: switch rate vs episode (one line per action column)."""

        if df.empty:
            return None

        out = self.results_dir / filename
        fig, ax = plt.subplots(figsize=(12, 6))

        for action_col, g in df.groupby("action_column"):
            ax.plot(g["episode"], g["switch_rate"], linewidth=2, label=action_col)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Switch Rate SR")
        ax.set_title("Policy Stability: Action Switch Rate vs Episode")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out


class BellmanStabilityAnalyzer:
    """Bellman Update Stability (Learning Stability) - Method 1.

    This is a thin wrapper around the already-implemented
    `BellmanContractionStabilityAnalyzer` in analysis/stability_analysis.py.

    Metric:
        ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞

    Requirements satisfied:
    - Loads Q-table snapshots per agent per episode from an .npz file.
    - Flattens Q-tables before norm computation (handled by underlying analyzer).
    - Computes ΔV per episode, saves CSV and plot.

    Interpretation (high-level):
    - Bounded or decreasing ΔV indicates stable learning dynamics.
    - Strict convergence to 0 is not required for practical MARL stability.
    """

    def __init__(self, results_dir: Path = RESULTS_DIR_DEFAULT) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_from_npz(
        self,
        qtables_npz_path: Path = Path("results/stability/qtables_per_episode.npz"),
    ) -> Tuple[Path, Path]:
        qtables = load_qtables_history(qtables_npz_path)
        analyzer = _BellmanContractionStabilityAnalyzer(results_dir=str(self.results_dir))
        analyzer.q_tables_per_episode = qtables
        return analyzer.run_analysis()


class ConsensusStabilityAnalyzer:
    """Consensus Stability (Mean-Square Disagreement) - Method 2.

    This is a thin wrapper around the already-implemented
    `ConsensusStabilityAnalyzer` in analysis/stability_analysis.py.

    Metric family:
        V_avg(k) = (1/M) Σ_i V_i(k)
        D(k)     = (1/M) Σ_i || V_i(k) - V_avg(k) ||_2  (implementation)

    Note:
    The project’s existing implementation uses the average L2 distance to
    the mean vector. This is consistent with a disagreement metric; squared
    variants are common, but we keep the existing implementation unchanged.

    Interpretation:
    - Bounded disagreement indicates coordinated emergent behavior.
    - Full consensus is not required.
    """

    def __init__(self, results_dir: Path = RESULTS_DIR_DEFAULT) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_from_npz(
        self,
        qtables_npz_path: Path = Path("results/stability/qtables_per_episode.npz"),
    ) -> Tuple[Path, Path]:
        qtables = load_qtables_history(qtables_npz_path)
        analyzer = _ConsensusStabilityAnalyzer(results_dir=str(self.results_dir))
        analyzer.q_tables_per_episode = qtables
        return analyzer.run_analysis()
