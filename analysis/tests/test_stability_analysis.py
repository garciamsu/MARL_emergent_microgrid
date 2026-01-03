"""Validation script for stability analysis implementation.

Tests the stability analysis modules with synthetic Q-table data to verify:
1. Correct metric computation
2. File I/O operations
3. Plot generation
4. Expected convergence detection

This is a smoke test - does NOT require running full training.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
# This module lives under analysis/tests/, so the repo root is two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.stability.stability_analysis import (
    BellmanContractionStabilityAnalyzer,
    ConsensusStabilityAnalyzer
)


def create_synthetic_qtables(num_episodes=50, num_agents=5, 
                             convergence_rate=0.9, noise_level=0.1):
    """Create synthetic Q-table snapshots simulating convergence.
    
    Args:
        num_episodes: Number of episodes to simulate.
        num_agents: Number of agents.
        convergence_rate: Exponential decay rate for value changes.
        noise_level: Random noise amplitude.
    
    Returns:
        List of Q-table snapshots (one per episode).
    """
    qtables_per_episode = []
    
    # Initialize random Q-tables for each agent
    base_qtables = {}
    for i in range(num_agents):
        agent_name = f"agent_{i}"
        # Create 10 states with 3 actions each
        qtable = {}
        for state_idx in range(10):
            state = (state_idx, 0, 0)  # Simple state tuple
            qtable[state] = {
                0: np.random.randn(),
                1: np.random.randn(),
                2: np.random.randn()
            }
        base_qtables[agent_name] = qtable
    
    # Simulate convergence over episodes
    for episode in range(num_episodes):
        episode_qtables = {}
        
        for agent_name, base_qtable in base_qtables.items():
            qtable_copy = {}
            
            for state, actions in base_qtable.items():
                qtable_copy[state] = {}
                for action, value in actions.items():
                    # Add decreasing noise to simulate convergence
                    decay_factor = convergence_rate ** episode
                    noise = noise_level * decay_factor * np.random.randn()
                    qtable_copy[state][action] = value + noise
            
            episode_qtables[agent_name] = qtable_copy
        
        qtables_per_episode.append(episode_qtables)
    
    return qtables_per_episode


def test_bellman_contraction():
    """Test Bellman contraction analyzer."""
    print("\n" + "="*80)
    print("TEST 1: BELLMAN CONTRACTION STABILITY ANALYZER")
    print("="*80)
    
    # Create synthetic data
    print("\n📦 Creating synthetic Q-table data (50 episodes, 5 agents)...")
    qtables = create_synthetic_qtables(num_episodes=50, num_agents=5)
    
    # Initialize analyzer
    analyzer = BellmanContractionStabilityAnalyzer("results/stability/test")
    analyzer.q_tables_per_episode = qtables
    
    # Run analysis
    print("\n🔬 Running Bellman contraction analysis...")
    csv_path, plot_path = analyzer.run_analysis()
    
    # Validate outputs
    assert csv_path.exists(), "CSV file not created"
    assert plot_path.exists(), "Plot file not created"
    
    # Check convergence
    final_delta = analyzer.delta_v_per_episode[-1]
    initial_delta = analyzer.delta_v_per_episode[0]
    
    print(f"\n✅ Test passed!")
    print(f"   Initial ΔV: {initial_delta:.6f}")
    print(f"   Final ΔV:   {final_delta:.6f}")
    print(f"   Reduction:  {(1 - final_delta/initial_delta)*100:.1f}%")
    
    assert final_delta < initial_delta, "Expected convergence (ΔV should decrease)"
    
    return True


def test_consensus_stability():
    """Test consensus analyzer."""
    print("\n" + "="*80)
    print("TEST 2: CONSENSUS STABILITY ANALYZER")
    print("="*80)
    
    # Create synthetic data
    print("\n📦 Creating synthetic Q-table data (50 episodes, 5 agents)...")
    qtables = create_synthetic_qtables(num_episodes=50, num_agents=5)
    
    # Initialize analyzer
    analyzer = ConsensusStabilityAnalyzer("results/stability/test")
    analyzer.q_tables_per_episode = qtables
    
    # Run analysis
    print("\n🔬 Running consensus stability analysis...")
    csv_path, plot_path = analyzer.run_analysis()
    
    # Validate outputs
    assert csv_path.exists(), "CSV file not created"
    assert plot_path.exists(), "Plot file not created"
    
    # Check metrics
    final_D = analyzer.consensus_deviation_per_episode[-1]
    initial_D = analyzer.consensus_deviation_per_episode[0]
    
    print(f"\n✅ Test passed!")
    print(f"   Initial D: {initial_D:.6f}")
    print(f"   Final D:   {final_D:.6f}")
    print(f"   Reduction: {(1 - final_D/initial_D)*100:.1f}%")
    
    # Note: Consensus may not always decrease with independent convergence
    # This is expected behavior - just validate computation
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 3: EDGE CASES")
    print("="*80)
    
    # Test with single episode (should warn)
    print("\n📦 Testing with single episode (should warn)...")
    qtables_single = create_synthetic_qtables(num_episodes=1, num_agents=3)
    
    analyzer = BellmanContractionStabilityAnalyzer("results/stability/test")
    analyzer.q_tables_per_episode = qtables_single
    analyzer.compute_stability_metrics()
    
    assert len(analyzer.delta_v_per_episode) == 0, "Should have no metrics for single episode"
    print("✅ Correctly handled single episode case")
    
    # Test with heterogeneous Q-tables (different sizes)
    print("\n📦 Testing with heterogeneous Q-table sizes...")
    qtables_hetero = []
    for episode in range(10):
        episode_qtables = {}
        for i in range(3):
            agent_name = f"agent_{i}"
            qtable = {}
            # Each agent has different number of states
            num_states = 5 + i + episode  # Growing state space
            for state_idx in range(num_states):
                state = (state_idx, 0)
                qtable[state] = {0: np.random.randn(), 1: np.random.randn()}
            episode_qtables[agent_name] = qtable
        qtables_hetero.append(episode_qtables)
    
    analyzer = BellmanContractionStabilityAnalyzer("results/stability/test")
    analyzer.q_tables_per_episode = qtables_hetero
    analyzer.compute_stability_metrics()
    
    assert len(analyzer.delta_v_per_episode) == 9, "Should compute 9 deltas for 10 episodes"
    print("✅ Correctly handled heterogeneous Q-table sizes")
    
    # Test with empty Q-tables
    print("\n📦 Testing with empty Q-tables...")
    qtables_empty = [{"agent_0": {}, "agent_1": {}} for _ in range(5)]
    
    analyzer = ConsensusStabilityAnalyzer("results/stability/test")
    analyzer.q_tables_per_episode = qtables_empty
    analyzer.compute_consensus_metrics()
    
    assert all(d == 0.0 for d in analyzer.consensus_deviation_per_episode), \
        "Empty Q-tables should yield zero deviation"
    print("✅ Correctly handled empty Q-tables")
    
    print("\n✅ All edge cases passed!")
    
    return True


def test_flattening_consistency():
    """Test that Q-table flattening is consistent."""
    print("\n" + "="*80)
    print("TEST 4: FLATTENING CONSISTENCY")
    print("="*80)
    
    # Create Q-table
    qtable = {
        (0, 0): {0: 1.0, 1: 2.0, 2: 3.0},
        (0, 1): {0: 4.0, 1: 5.0, 2: 6.0},
        (1, 0): {0: 7.0, 1: 8.0, 2: 9.0},
    }
    
    analyzer = BellmanContractionStabilityAnalyzer("results/stability/test")
    
    # Flatten multiple times
    flat1 = analyzer.flatten_qtable(qtable)
    flat2 = analyzer.flatten_qtable(qtable)
    flat3 = analyzer.flatten_qtable(qtable)
    
    # Check consistency
    assert np.array_equal(flat1, flat2), "Flattening should be deterministic"
    assert np.array_equal(flat2, flat3), "Flattening should be deterministic"
    
    # Check values
    expected_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert np.array_equal(flat1, expected_values), "Flattened values should match sorted order"
    
    print("✅ Q-table flattening is consistent and correct")
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("STABILITY ANALYSIS VALIDATION SUITE")
    print("="*80)
    
    try:
        # Run tests
        test_flattening_consistency()
        test_bellman_contraction()
        test_consensus_stability()
        test_edge_cases()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\n💡 The stability analysis implementation is validated.")
        print("   Ready for use with real training data.")
        print("\n📝 Next steps:")
        print("   1. python analysis/collect_qtables_per_episode.py")
        print("   2. python analysis/run_stability_analysis.py")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
