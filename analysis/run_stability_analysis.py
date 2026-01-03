"""Unified stability analysis runner.

This script integrates:
- Learning stability (methods 1–2): Bellman contraction + consensus on Q-tables.
- Operational stability (methods 3–6): boundedness/variability/resilience/policy.

Default behavior runs the full suite (methods 1–6) if inputs exist.

Prerequisites:
    - Methods 1–2 require: results/stability/qtables_per_episode.npz
    - Methods 3–6 require: results/evolution/episode_<k>.csv

Usage:
    python analysis/run_stability_analysis.py
    python analysis/run_stability_analysis.py --mode learning
"""

import sys
from pathlib import Path
import argparse


def _summarize_numeric(series, prefix: str) -> dict:
    """Return robust summary stats for a numeric series."""

    import numpy as np
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_median": float(s.median()),
        f"{prefix}_p10": float(s.quantile(0.10)),
        f"{prefix}_p90": float(s.quantile(0.90)),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_max": float(s.max()),
    }


def _tail_slice(df, frac: float = 0.10):
    if df.empty:
        return df
    n = max(1, int(round(len(df) * frac)))
    return df.iloc[-n:].copy()


def _episode_mean(df, value_col: str):
    import pandas as pd

    if df.empty or "episode" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["episode", value_col])
    g = df.groupby("episode", as_index=False)[value_col].mean()
    return g.sort_values("episode").reset_index(drop=True)


def _print_methods_comparison(results_dir: Path) -> None:
    import pandas as pd

    rows = []

    def add(method: str, metric: str, value, units: str = "") -> None:
        rows.append({"method": method, "metric": metric, "value": value, "units": units})

    # Method 1: Bellman contraction
    p_bell = results_dir / "bellman_contraction_stability.csv"
    if p_bell.exists():
        df = pd.read_csv(p_bell)
        if "delta_v" in df.columns and not df["delta_v"].empty:
            add("M1", "ΔV_final", float(df["delta_v"].iloc[-1]), "")
            add("M1", "ΔV_p90", float(pd.to_numeric(df["delta_v"], errors="coerce").quantile(0.90)), "")
    else:
        add("M1", "ΔV_final", "(missing qtables)")

    # Method 2: Consensus deviation
    p_cons = results_dir / "consensus_stability.csv"
    if p_cons.exists():
        df = pd.read_csv(p_cons)
        if "consensus_deviation" in df.columns and not df["consensus_deviation"].empty:
            add("M2", "D_final", float(df["consensus_deviation"].iloc[-1]), "")
            add("M2", "D_p90", float(pd.to_numeric(df["consensus_deviation"], errors="coerce").quantile(0.90)), "")
    else:
        add("M2", "D_final", "(missing qtables)")

    # Method 3: Energy boundedness
    p_bound = results_dir / "energy_boundedness.csv"
    if p_bound.exists():
        df = pd.read_csv(p_bound)
        if "max_abs_e" in df.columns:
            tail = _tail_slice(df, 0.10)
            add("M3", "last10_max|e|_mean", float(pd.to_numeric(tail["max_abs_e"], errors="coerce").mean()), "W")
        if "time_in_band_ratio" in df.columns:
            tail = _tail_slice(df, 0.10)
            add("M3", "last10_in_band_mean", float(pd.to_numeric(tail["time_in_band_ratio"], errors="coerce").mean()), "")

    # Method 4: Variability
    p_var = results_dir / "variability_stability.csv"
    if p_var.exists():
        df = pd.read_csv(p_var)
        tail = _tail_slice(df, 0.10)
        if "std_e" in tail.columns:
            add("M4", "last10_std(e)_mean", float(pd.to_numeric(tail["std_e"], errors="coerce").mean()), "W")
        if "rms_e" in tail.columns:
            add("M4", "last10_rms(e)_mean", float(pd.to_numeric(tail["rms_e"], errors="coerce").mean()), "W")

    # Method 5: Resilience
    p_res = results_dir / "resilience_recovery_summary.csv"
    if p_res.exists():
        df = pd.read_csv(p_res)
        tail = _tail_slice(df, 0.10)
        if "recovered_ratio" in tail.columns:
            add("M5", "last10_recovered_ratio_mean", float(pd.to_numeric(tail["recovered_ratio"], errors="coerce").mean()), "")
        if "median_recovery_time_hours" in tail.columns:
            add(
                "M5",
                "last10_median_recovery_h_mean",
                float(pd.to_numeric(tail["median_recovery_time_hours"], errors="coerce").mean()),
                "h",
            )

    # Method 6: Policy stability (switch rate)
    p_pol = results_dir / "policy_switch_rate.csv"
    if p_pol.exists():
        df = pd.read_csv(p_pol)
        per_ep = _episode_mean(df, "switch_rate")
        tail = _tail_slice(per_ep, 0.10)
        if not tail.empty and "switch_rate" in tail.columns:
            add("M6", "last10_switch_rate_mean", float(pd.to_numeric(tail["switch_rate"], errors="coerce").mean()), "")

    if not rows:
        return

    table = pd.DataFrame(rows)
    print("\n" + "-" * 80)
    print("METHODS 1–6: QUICK COMPARISON (late-training = last 10% episodes when applicable)")
    print("-" * 80)
    print(table.to_string(index=False))

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_learning_stability(results_dir: str = "results/stability") -> None:
    """Run methods 1–2 (learning stability) from Q-table history."""

    from analysis.stability.stability_analysis import run_both_stability_analyses
    from analysis.collect_qtables_per_episode import load_qtables_history

    print("\n" + "=" * 80)
    print("LEARNING STABILITY ANALYSIS (METHODS 1–2)")
    print("=" * 80)

    input_path = Path(results_dir) / "qtables_per_episode.npz"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Q-table history not found at {input_path}. "
            "Run: python analysis/collect_qtables_per_episode.py"
        )

    print(f"Loading Q-table history from {input_path}...")
    qtables_per_episode = load_qtables_history(input_path)
    if not qtables_per_episode:
        raise ValueError(f"No Q-table data found in {input_path}")

    print(f"[OK] Loaded {len(qtables_per_episode)} episodes")
    if qtables_per_episode[0]:
        agent_names = list(qtables_per_episode[0].keys())
        print(f"[OK] Agents: {', '.join(agent_names)}")

    run_both_stability_analyses(qtables_per_episode, results_dir=results_dir)


def run_full_suite() -> None:
    """Run methods 1–6 (learning + operational stability).

    This runs operational analyses (methods 3–6) and, when available,
    includes learning stability (methods 1–2) derived from Q-table history.
    """

    run_operational_stability_suite()


def run_operational_stability_suite() -> None:
    """Run methods 1–6 in one pass (operational runner behavior)."""

    import numpy as np
    import pandas as pd

    from core.csv_handler import write_result_csv
    from analysis.stability.operational_stability_analysis import (
        EVOLUTION_DIR_DEFAULT,
        RESULTS_DIR_DEFAULT,
        BellmanStabilityAnalyzer,
        ConsensusStabilityAnalyzer,
        EnergyBoundednessAnalyzer,
        VariabilityStabilityAnalyzer,
        ResilienceAnalyzer,
        PolicyStabilityAnalyzer,
        load_episodes,
    )

    # =========================
    # PARAMETERS (EDITABLE)
    # =========================
    epsilon_band_w = 3000.0
    event_threshold_w = 2.0 * epsilon_band_w
    persistence_window_steps = 6
    dt_h = 1.0
    rolling_window_steps = 12

    evolution_dir = EVOLUTION_DIR_DEFAULT
    results_dir = RESULTS_DIR_DEFAULT

    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS (METHODS 1–6)")
    print("=" * 80)
    print(f"Input:  {evolution_dir}")
    print(f"Output: {results_dir}")

    episodes = load_episodes(evolution_dir=evolution_dir)
    if not episodes:
        raise FileNotFoundError(f"No episode CSVs found under {evolution_dir}")
    print(f"Loaded {len(episodes)} episodes")

    # Methods 1–2 (learning stability) - requires Q-table history
    qtables_path = results_dir / "qtables_per_episode.npz"
    if qtables_path.exists():
        bellman = BellmanStabilityAnalyzer(results_dir=results_dir)
        bellman_csv, bellman_plot = bellman.run_from_npz(qtables_path)
        print(f"[OK] Saved Bellman stability: {bellman_csv}")
        print(f"[OK] Saved plot: {bellman_plot}")

        consensus = ConsensusStabilityAnalyzer(results_dir=results_dir)
        consensus_csv, consensus_plot = consensus.run_from_npz(qtables_path)
        print(f"[OK] Saved consensus stability: {consensus_csv}")
        print(f"[OK] Saved plot: {consensus_plot}")
    else:
        print(f"[SKIP] Missing {qtables_path} (run collect_qtables_per_episode.py to generate it)")

    # 3) Energy Balance Boundedness
    boundedness = EnergyBoundednessAnalyzer(epsilon_band_w=epsilon_band_w, results_dir=results_dir)
    df_bounded = boundedness.analyze(episodes)
    p_bounded = boundedness.save_results(df_bounded)
    print(f"[OK] Saved boundedness metrics: {p_bounded}")

    # 4) Variability-Based Stability
    variability = VariabilityStabilityAnalyzer(
        rolling_window_steps=rolling_window_steps,
        results_dir=results_dir,
    )
    df_var = variability.analyze(episodes)
    p_var = variability.save_results(df_var)
    plot_paths = variability.plot(df_var)
    print(f"[OK] Saved variability metrics: {p_var}")
    for p in plot_paths:
        print(f"[OK] Saved plot: {p}")

    # 5) Resilience / Recovery Time
    resilience = ResilienceAnalyzer(
        epsilon_band_w=epsilon_band_w,
        event_threshold_w=event_threshold_w,
        persistence_window_steps=persistence_window_steps,
        dt_h=dt_h,
        results_dir=results_dir,
    )
    events_df, summary_df = resilience.analyze(episodes)
    p_events, p_summary = resilience.save_results(events_df, summary_df)
    print(f"[OK] Saved resilience events: {p_events}")
    print(f"[OK] Saved resilience summary: {p_summary}")

    # 6) Policy Stability (Switch Rate)
    policy = PolicyStabilityAnalyzer(results_dir=results_dir)
    df_sr = policy.analyze(episodes)
    p_sr = policy.save_results(df_sr)
    p_sr_plot = policy.plot(df_sr)
    print(f"[OK] Saved switch-rate metrics: {p_sr}")
    if p_sr_plot is not None:
        print(f"[OK] Saved plot: {p_sr_plot}")

    # One-row summary
    summary: dict = {
        "num_episodes": float(len(episodes)),
        "epsilon_band_w": float(epsilon_band_w),
        "event_threshold_w": float(event_threshold_w),
        "persistence_window_steps": float(persistence_window_steps),
        "dt_h": float(dt_h),
        "rolling_window_steps": float(rolling_window_steps) if rolling_window_steps is not None else np.nan,
    }

    if not df_bounded.empty:
        summary.update(_summarize_numeric(df_bounded["max_abs_e"], "max_abs_e"))
        summary.update(_summarize_numeric(df_bounded["p95_abs_e"], "p95_abs_e"))
        summary.update(_summarize_numeric(df_bounded["time_in_band_ratio"], "time_in_band_ratio"))

        tail_b = _tail_slice(df_bounded, 0.10)
        summary.update(_summarize_numeric(tail_b["max_abs_e"], "last10_max_abs_e"))
        summary.update(_summarize_numeric(tail_b["time_in_band_ratio"], "last10_time_in_band_ratio"))

    if not df_var.empty:
        summary.update(_summarize_numeric(df_var["std_e"], "std_e"))
        summary.update(_summarize_numeric(df_var["rms_e"], "rms_e"))

        tail_v = _tail_slice(df_var, 0.10)
        summary.update(_summarize_numeric(tail_v["std_e"], "last10_std_e"))
        summary.update(_summarize_numeric(tail_v["rms_e"], "last10_rms_e"))

    if not summary_df.empty:
        summary.update(_summarize_numeric(summary_df["num_events"], "events_per_episode"))
        summary.update(_summarize_numeric(summary_df["recovered_ratio"], "recovered_ratio"))
        summary.update(_summarize_numeric(summary_df["median_recovery_time_hours"], "median_recovery_time_hours"))

        tail_r = _tail_slice(summary_df, 0.10)
        summary.update(_summarize_numeric(tail_r["recovered_ratio"], "last10_recovered_ratio"))

    if not df_sr.empty:
        summary.update(_summarize_numeric(df_sr["switch_rate"], "switch_rate"))
        tail_sr = _tail_slice(df_sr, 0.10)
        summary.update(_summarize_numeric(tail_sr["switch_rate"], "last10_switch_rate"))

    bellman_csv_path = results_dir / "bellman_contraction_stability.csv"
    if bellman_csv_path.exists():
        bellman_df = pd.read_csv(bellman_csv_path)
        if "delta_v" in bellman_df.columns:
            summary.update(_summarize_numeric(bellman_df["delta_v"], "delta_v"))

    consensus_csv_path = results_dir / "consensus_stability.csv"
    if consensus_csv_path.exists():
        cons_df = pd.read_csv(consensus_csv_path)
        if "consensus_deviation" in cons_df.columns:
            summary.update(_summarize_numeric(cons_df["consensus_deviation"], "consensus_deviation"))

    summary_df_one_row = pd.DataFrame([summary])
    p_summary = results_dir / "operational_stability_summary.csv"
    write_result_csv(summary_df_one_row, p_summary)
    print(f"[OK] Saved one-row summary: {p_summary}")

    _print_methods_comparison(results_dir)

    print("\n" + "=" * 80)
    print("[OK] STABILITY ANALYSIS COMPLETE")
    print("=" * 80)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified stability analysis runner")
    parser.add_argument(
        "--mode",
        choices=["all", "learning"],
        default="all",
        help="Run full suite (all) or learning-only stability (learning).",
    )
    args = parser.parse_args(argv)

    if args.mode == "learning":
        run_learning_stability(results_dir="results/stability")
        return

    run_full_suite()
    

if __name__ == "__main__":
    main()
