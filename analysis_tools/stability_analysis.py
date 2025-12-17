"""Stability Analysis for Distributed Multi-Agent Reinforcement Learning.

This module implements two independent stability studies aligned with
distributed dynamic programming and emergent systems theory:

1. Bellman Contraction Stability: Measures convergence of value updates
   by tracking the infinity norm of value function differences between
   consecutive episodes.

2. Consensus Stability: Measures distributed consensus among agents by
   computing the average deviation from the mean value function.

Both analyzers are designed to work with Q-table snapshots saved during
or after training, providing post-hoc analysis of learning dynamics.

Theoretical Background:
-----------------------
In distributed Q-learning, stability requires that:
- Individual value functions converge (Bellman contraction property)
- Agents reach consensus on state valuations (distributed coordination)

These metrics provide empirical validation of theoretical convergence
guarantees in multi-agent settings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Anyimport sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.csv_handler import write_result_csvimport os


class BellmanContractionStabilityAnalyzer:
    """Analyzes stability via distributed Bellman contraction.
    
    Computes the maximum infinity norm of value function differences
    between consecutive episodes across all agents:
    
        ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞
    
    where V_i(k) is the flattened Q-table (value function) of agent i
    at episode k.
    
    Interpretation:
    - ΔV(k) → 0 indicates stable learning dynamics and convergence
    - Persistent oscillations or divergence indicate instability
    - Theoretical foundation: Bellman operator is a contraction mapping
      with rate γ (discount factor), so differences should decrease
    
    Attributes:
        q_tables_per_episode (List[Dict[str, Dict]]): Q-tables indexed by
            episode, then agent name.
        delta_v_per_episode (List[float]): Computed ΔV values per episode.
        results_dir (Path): Directory for output files.
    """
    
    def __init__(self, results_dir: str = "results/stability"):
        """Initialize analyzer.
        
        Args:
            results_dir: Directory path for output CSV and plots.
        """
        self.q_tables_per_episode: List[Dict[str, Dict]] = []
        self.delta_v_per_episode: List[float] = []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_episode_qtables(self, episode: int, agents: Dict[str, Any]) -> None:
        """Extract and store Q-tables from agents at a specific episode.
        
        Args:
            episode: Episode index (0-based).
            agents: Dictionary mapping agent names to agent objects.
                    Each agent must have a 'q_table' attribute.
        """
        episode_qtables = {}
        for name, agent in agents.items():
            # Deep copy Q-table to avoid references to mutable state
            q_table = getattr(agent, "q_table", {})
            if q_table:
                # Store as nested dict: {state: {action: value}}
                episode_qtables[name] = {
                    state: dict(actions) 
                    for state, actions in q_table.items()
                }
        
        self.q_tables_per_episode.append(episode_qtables)
    
    def flatten_qtable(self, q_table: Dict) -> np.ndarray:
        """Flatten Q-table into a 1D array of Q-values.
        
        The flattened representation enables direct computation of norms.
        Sorts states and actions for consistent ordering across episodes.
        
        Args:
            q_table: Nested dictionary {state: {action: value}}.
        
        Returns:
            1D numpy array of Q-values in sorted order.
        """
        values = []
        # Sort states for consistent ordering
        sorted_states = sorted(q_table.keys())
        for state in sorted_states:
            # Sort actions within each state
            sorted_actions = sorted(q_table[state].keys())
            for action in sorted_actions:
                values.append(q_table[state][action])
        
        return np.array(values) if values else np.array([0.0])
    
    def compute_stability_metrics(self) -> None:
        """Compute ΔV(k) for all consecutive episode pairs.
        
        For each episode k and k+1, computes:
            ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞
        
        Handles Q-tables with different state-action space sizes by
        computing per-agent norms and taking the maximum.
        """
        self.delta_v_per_episode = []
        
        if len(self.q_tables_per_episode) < 2:
            print("Warning: Need at least 2 episodes to compute stability metrics.")
            return
        
        for k in range(len(self.q_tables_per_episode) - 1):
            qtables_k = self.q_tables_per_episode[k]
            qtables_k1 = self.q_tables_per_episode[k + 1]
            
            # Compute max infinity norm across all agents
            max_norm = 0.0
            
            # Iterate over agents present in both episodes
            common_agents = set(qtables_k.keys()) & set(qtables_k1.keys())
            
            for agent_name in common_agents:
                v_k = self.flatten_qtable(qtables_k[agent_name])
                v_k1 = self.flatten_qtable(qtables_k1[agent_name])
                
                # Handle dimension mismatch (new states discovered)
                if len(v_k) != len(v_k1):
                    # Pad shorter array with zeros
                    max_len = max(len(v_k), len(v_k1))
                    v_k_padded = np.pad(v_k, (0, max_len - len(v_k)))
                    v_k1_padded = np.pad(v_k1, (0, max_len - len(v_k1)))
                    diff = v_k1_padded - v_k_padded
                else:
                    diff = v_k1 - v_k
                
                # Compute infinity norm (maximum absolute element)
                inf_norm = np.max(np.abs(diff)) if len(diff) > 0 else 0.0
                max_norm = max(max_norm, inf_norm)
            
            self.delta_v_per_episode.append(max_norm)
    
    def save_results(self, filename: str = "bellman_contraction_stability.csv") -> Path:
        """Save ΔV metrics to CSV file.
        
        Args:
            filename: Output CSV filename.
        
        Returns:
            Path to saved CSV file.
        """
        output_path = self.results_dir / filename
        
        df = pd.DataFrame({
            "episode": range(len(self.delta_v_per_episode)),
            "delta_v": self.delta_v_per_episode
        })
        
        write_result_csv(df, output_path)
        print(f"✅ Bellman contraction stability metrics saved to {output_path}")
        
        return output_path
    
    def plot_stability(self, filename: str = "bellman_contraction_stability.png") -> Path:
        """Generate and save stability plot.
        
        Creates a line plot of ΔV(k) over episodes with:
        - Logarithmic y-axis (better visualization of convergence)
        - Horizontal reference line at ΔV = 0
        - Grid for readability
        
        Args:
            filename: Output figure filename.
        
        Returns:
            Path to saved figure.
        """
        output_path = self.results_dir / filename
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(self.delta_v_per_episode))
        ax.plot(episodes, self.delta_v_per_episode, 
                linewidth=2, color='#2E86AB', marker='o', 
                markersize=4, label='ΔV(k)')
        
        ax.set_xlabel("Episode k", fontsize=12, fontweight='bold')
        ax.set_ylabel("ΔV(k) = max_i ||V_i(k+1) - V_i(k)||_∞", 
                     fontsize=12, fontweight='bold')
        ax.set_title("Bellman Contraction Stability Analysis\n"
                    "Convergence: ΔV → 0 indicates stable learning",
                    fontsize=14, fontweight='bold', pad=20)
        
        # Logarithmic scale for better visualization of convergence
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # Add interpretation text
        textstr = (
            "Interpretation:\n"
            "• ΔV → 0: Convergence (stable)\n"
            "• ΔV oscillates: Instability\n"
            "• ΔV diverges: Non-convergent"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Bellman contraction stability plot saved to {output_path}")
        
        return output_path
    
    def run_analysis(self) -> Tuple[Path, Path]:
        """Execute complete stability analysis pipeline.
        
        Returns:
            Tuple of (csv_path, plot_path).
        """
        print("\n" + "="*80)
        print("BELLMAN CONTRACTION STABILITY ANALYSIS")
        print("="*80)
        
        self.compute_stability_metrics()
        csv_path = self.save_results()
        plot_path = self.plot_stability()
        
        # Print summary statistics
        if self.delta_v_per_episode:
            final_delta = self.delta_v_per_episode[-1]
            max_delta = max(self.delta_v_per_episode)
            min_delta = min(self.delta_v_per_episode)
            
            print(f"\n📊 Summary Statistics:")
            print(f"   Final ΔV:    {final_delta:.6f}")
            print(f"   Max ΔV:      {max_delta:.6f}")
            print(f"   Min ΔV:      {min_delta:.6f}")
            
            # Convergence assessment
            if final_delta < 0.01:
                print(f"   Status: ✅ CONVERGED (ΔV < 0.01)")
            elif final_delta < 0.1:
                print(f"   Status: ⚠️  NEAR CONVERGENCE (ΔV < 0.1)")
            else:
                print(f"   Status: ❌ NOT CONVERGED (ΔV ≥ 0.1)")
        
        return csv_path, plot_path


class ConsensusStabilityAnalyzer:
    """Analyzes stability via distributed consensus among agents.
    
    Computes the average deviation from mean value function across agents:
    
        V_avg(k) = (1/N) * Σ_i V_i(k)
        D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2
    
    where N is the number of agents and || · ||_2 is the Euclidean norm.
    
    Interpretation:
    - D(k) → 0 indicates consensus and distributed stability
    - Persistent D(k) > 0 implies lack of coordination
    - Theoretical foundation: Distributed consensus requires agents to
      converge toward a common value representation
    
    Attributes:
        q_tables_per_episode (List[Dict[str, Dict]]): Q-tables indexed by
            episode, then agent name.
        consensus_deviation_per_episode (List[float]): Computed D(k) values.
        results_dir (Path): Directory for output files.
    """
    
    def __init__(self, results_dir: str = "results/stability"):
        """Initialize analyzer.
        
        Args:
            results_dir: Directory path for output CSV and plots.
        """
        self.q_tables_per_episode: List[Dict[str, Dict]] = []
        self.consensus_deviation_per_episode: List[float] = []
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def add_episode_qtables(self, episode: int, agents: Dict[str, Any]) -> None:
        """Extract and store Q-tables from agents at a specific episode.
        
        Args:
            episode: Episode index (0-based).
            agents: Dictionary mapping agent names to agent objects.
        """
        episode_qtables = {}
        for name, agent in agents.items():
            q_table = getattr(agent, "q_table", {})
            if q_table:
                episode_qtables[name] = {
                    state: dict(actions) 
                    for state, actions in q_table.items()
                }
        
        self.q_tables_per_episode.append(episode_qtables)
    
    def flatten_qtable(self, q_table: Dict) -> np.ndarray:
        """Flatten Q-table into a 1D array of Q-values.
        
        Args:
            q_table: Nested dictionary {state: {action: value}}.
        
        Returns:
            1D numpy array of Q-values in sorted order.
        """
        values = []
        sorted_states = sorted(q_table.keys())
        for state in sorted_states:
            sorted_actions = sorted(q_table[state].keys())
            for action in sorted_actions:
                values.append(q_table[state][action])
        
        return np.array(values) if values else np.array([0.0])
    
    def compute_consensus_metrics(self) -> None:
        """Compute D(k) for each episode.
        
        For each episode k:
        1. Flatten all agent Q-tables to vectors
        2. Compute mean vector V_avg(k) across agents
        3. Compute Euclidean distance from each agent to mean
        4. Average distances to get D(k)
        
        Handles heterogeneous Q-table sizes by padding to common dimension.
        """
        self.consensus_deviation_per_episode = []
        
        if not self.q_tables_per_episode:
            print("Warning: No Q-tables available for consensus analysis.")
            return
        
        for k, qtables_k in enumerate(self.q_tables_per_episode):
            if not qtables_k:
                self.consensus_deviation_per_episode.append(0.0)
                continue
            
            # Flatten all agent Q-tables
            agent_vectors = {}
            max_dim = 0
            
            for agent_name, qtable in qtables_k.items():
                v_i = self.flatten_qtable(qtable)
                agent_vectors[agent_name] = v_i
                max_dim = max(max_dim, len(v_i))
            
            # Pad all vectors to common dimension
            padded_vectors = []
            agent_names = []
            
            for agent_name, v_i in agent_vectors.items():
                v_i_padded = np.pad(v_i, (0, max_dim - len(v_i)))
                padded_vectors.append(v_i_padded)
                agent_names.append(agent_name)
            
            if not padded_vectors:
                self.consensus_deviation_per_episode.append(0.0)
                continue
            
            # Compute mean vector V_avg(k)
            v_avg = np.mean(padded_vectors, axis=0)
            
            # Compute average Euclidean distance from mean
            N = len(padded_vectors)
            total_distance = 0.0
            
            for v_i in padded_vectors:
                diff = v_i - v_avg
                l2_norm = np.linalg.norm(diff)  # Euclidean norm
                total_distance += l2_norm
            
            D_k = total_distance / N if N > 0 else 0.0
            self.consensus_deviation_per_episode.append(D_k)
    
    def save_results(self, filename: str = "consensus_stability.csv") -> Path:
        """Save consensus metrics to CSV file.
        
        Args:
            filename: Output CSV filename.
        
        Returns:
            Path to saved CSV file.
        """
        output_path = self.results_dir / filename
        
        df = pd.DataFrame({
            "episode": range(len(self.consensus_deviation_per_episode)),
            "consensus_deviation": self.consensus_deviation_per_episode
        })
        
        write_result_csv(df, output_path)
        print(f"✅ Consensus stability metrics saved to {output_path}")
        
        return output_path
    
    def plot_stability(self, filename: str = "consensus_stability.png") -> Path:
        """Generate and save consensus stability plot.
        
        Creates a line plot of D(k) over episodes with:
        - Logarithmic y-axis (better visualization)
        - Reference line at D = 0
        - Grid for readability
        
        Args:
            filename: Output figure filename.
        
        Returns:
            Path to saved figure.
        """
        output_path = self.results_dir / filename
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(len(self.consensus_deviation_per_episode))
        ax.plot(episodes, self.consensus_deviation_per_episode,
                linewidth=2, color='#A23B72', marker='s',
                markersize=4, label='D(k)')
        
        ax.set_xlabel("Episode k", fontsize=12, fontweight='bold')
        ax.set_ylabel("D(k) = (1/N) Σ_i ||V_i(k) - V_avg(k)||_2",
                     fontsize=12, fontweight='bold')
        ax.set_title("Consensus Stability Analysis\n"
                    "Convergence: D → 0 indicates distributed consensus",
                    fontsize=14, fontweight='bold', pad=20)
        
        # Logarithmic scale for better visualization
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # Add interpretation text
        textstr = (
            "Interpretation:\n"
            "• D → 0: Consensus achieved\n"
            "• D > 0: Lack of coordination\n"
            "• D diverges: System instability"
        )
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Consensus stability plot saved to {output_path}")
        
        return output_path
    
    def run_analysis(self) -> Tuple[Path, Path]:
        """Execute complete consensus analysis pipeline.
        
        Returns:
            Tuple of (csv_path, plot_path).
        """
        print("\n" + "="*80)
        print("CONSENSUS STABILITY ANALYSIS")
        print("="*80)
        
        self.compute_consensus_metrics()
        csv_path = self.save_results()
        plot_path = self.plot_stability()
        
        # Print summary statistics
        if self.consensus_deviation_per_episode:
            final_D = self.consensus_deviation_per_episode[-1]
            max_D = max(self.consensus_deviation_per_episode)
            min_D = min(self.consensus_deviation_per_episode)
            
            print(f"\n📊 Summary Statistics:")
            print(f"   Final D:     {final_D:.6f}")
            print(f"   Max D:       {max_D:.6f}")
            print(f"   Min D:       {min_D:.6f}")
            
            # Consensus assessment
            if final_D < 0.1:
                print(f"   Status: ✅ CONSENSUS ACHIEVED (D < 0.1)")
            elif final_D < 1.0:
                print(f"   Status: ⚠️  PARTIAL CONSENSUS (D < 1.0)")
            else:
                print(f"   Status: ❌ NO CONSENSUS (D ≥ 1.0)")
        
        return csv_path, plot_path


def run_both_stability_analyses(q_tables_per_episode: List[Dict[str, Dict]],
                                results_dir: str = "results/stability") -> None:
    """Run both stability analyses with shared Q-table data.
    
    Convenience function to execute both Bellman contraction and consensus
    analyses in sequence.
    
    Args:
        q_tables_per_episode: List of Q-table snapshots, one per episode.
            Each element is a dictionary mapping agent names to Q-tables.
        results_dir: Directory for output files.
    """
    print("\n" + "="*80)
    print("DISTRIBUTED MARL STABILITY ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(q_tables_per_episode)} episodes...")
    
    # Initialize analyzers
    bellman_analyzer = BellmanContractionStabilityAnalyzer(results_dir)
    consensus_analyzer = ConsensusStabilityAnalyzer(results_dir)
    
    # Populate data
    bellman_analyzer.q_tables_per_episode = q_tables_per_episode
    consensus_analyzer.q_tables_per_episode = q_tables_per_episode
    
    # Run analyses
    bellman_analyzer.run_analysis()
    consensus_analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("✅ STABILITY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    """Standalone execution example.
    
    This requires Q-tables to be saved per episode during training.
    See accompanying data collection script for implementation.
    """
    print("This module is designed to be imported and used programmatically.")
    print("See 'collect_qtables_per_episode.py' for data collection.")
    print("See 'run_stability_analysis.py' for standalone execution.")
