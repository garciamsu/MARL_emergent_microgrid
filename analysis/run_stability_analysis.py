"""Standalone stability analysis runner.

Executes both Bellman contraction and consensus stability analyses on
pre-collected Q-table snapshots.

This script loads Q-table history from a .npz file and runs both stability
analyses, generating metrics, plots, and interpretations.

Prerequisites:
    Run collect_qtables_per_episode.py first to generate Q-table history.

Usage:
    python analysis/run_stability_analysis.py

Output:
    - results/stability/bellman_contraction_stability.csv
    - results/stability/bellman_contraction_stability.png
    - results/stability/consensus_stability.csv
    - results/stability/consensus_stability.png
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.stability_analysis import (
    BellmanContractionStabilityAnalyzer,
    ConsensusStabilityAnalyzer,
    run_both_stability_analyses
)
from analysis.collect_qtables_per_episode import load_qtables_history


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("DISTRIBUTED MARL STABILITY ANALYSIS")
    print("="*80)
    
    # Load Q-table history
    input_path = Path("results/stability/qtables_per_episode.npz")
    
    if not input_path.exists():
        print(f"\n[ERROR] Q-table history not found at {input_path}")
        print(f"\nFirst run: python analysis/collect_qtables_per_episode.py")
        sys.exit(1)
    
    print(f"\nLoading Q-table history from {input_path}...")
    qtables_per_episode = load_qtables_history(input_path)
    
    if not qtables_per_episode:
        print(f"\n[ERROR] No Q-table data found in {input_path}")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(qtables_per_episode)} episodes")
    
    # Get agent names from first episode
    if qtables_per_episode[0]:
        agent_names = list(qtables_per_episode[0].keys())
        print(f"[OK] Agents: {', '.join(agent_names)}")
    
    # Run both stability analyses
    run_both_stability_analyses(qtables_per_episode, results_dir="results/stability")
    
    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: results/stability/")
    print(f"   - bellman_contraction_stability.csv")
    print(f"   - bellman_contraction_stability.png")
    print(f"   - consensus_stability.csv")
    print(f"   - consensus_stability.png")
    

if __name__ == "__main__":
    main()
