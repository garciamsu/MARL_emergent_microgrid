#!/usr/bin/env python3
"""
hyperparameter_search.py

Standalone experimental script for online Q-learning hyperparameter search.
Searches for suitable (alpha, gamma, epsilon_min) combinations under strictly
online learning conditions.

This script:
1. Wraps the existing simulator as a black box
2. Injects parameter values only through configuration
3. Executes single-episode experiments with long windows (96, 120, 144+ hours)
4. Evaluates whether the system achieves measurable reduction in energy imbalance
5. Logs imbalance metrics over time
6. Reports which parameter sets achieve consistent imbalance reduction

IMPORTANT: This script does NOT modify any existing training logic, agent classes,
environment code, or reward functions. It only tests different configurations.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import product
import tempfile
import shutil
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.simulation import run_training
from configs.loader import load_config
from core.utils import set_global_seed
from core.csv_handler import read_result_csv, write_result_csv


class HyperparameterSearcher:
    """Orchestrates hyperparameter search experiments for online Q-learning.
    
    This class wraps the existing training system and injects different
    hyperparameter configurations to evaluate their performance under
    online learning conditions with long episode windows.
    """
    
    def __init__(self, base_config_path: str = "configs/default.yaml"):
        """Initialize the searcher with base configuration.
        
        Args:
            base_config_path: Path to the base YAML configuration file
        """
        self.base_config_path = base_config_path
        self.base_config = load_config(base_config_path)
        
        # Create dedicated results directory for experiments
        self.experiment_dir = Path("results/experiments/hyperparameter_search")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this search run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.experiment_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🔬 Hyperparameter Search Initialized")
        print(f"{'='*80}")
        print(f"Base config: {base_config_path}")
        print(f"Results dir: {self.run_dir}")
        print(f"{'='*80}\n")
    
    def define_search_space(self) -> Dict[str, List[float]]:
        """Define the hyperparameter search space for online Q-learning.
        
        Returns:
            Dictionary with parameter names and their candidate values
        """
        search_space = {
            # Learning rate: expanded range for better exploration
            # Range: 0.20-0.50, testing lower and higher values
            'alpha': [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            
            # Discount factor: expanded range including lower values
            # Range: 0.85-0.95, testing more reactive (lower) values
            'gamma': [0.85, 0.88, 0.90, 0.92, 0.95],
            
            # Minimum exploration: never disable exploration in online learning
            # Range: 0.10-0.25, extended for more exploration
            'epsilon_min': [0.10, 0.15, 0.20, 0.25, 0.30],
        }
        
        print(f"📊 Search space defined:")
        for param, values in search_space.items():
            print(f"   {param}: {values}")
        print(f"\nTotal combinations: {np.prod([len(v) for v in search_space.values()])}\n")
        
        return search_space
    
    def create_experiment_config(
        self, 
        alpha: float, 
        gamma: float, 
        epsilon_min: float,
        episode_window_hours: int,
        seed: int
    ) -> Dict[str, Any]:
        """Create a configuration for a single experiment.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon_min: Minimum exploration rate
            episode_window_hours: Length of episode window in hours
            seed: Random seed for reproducibility
            
        Returns:
            Complete configuration dictionary for the experiment
        """
        # Deep copy base config
        import copy
        config = copy.deepcopy(self.base_config)
        
        # Single episode with long window for online learning evaluation
        config['simulation']['episodes'] = 1
        config['simulation']['episode_window_hours'] = episode_window_hours
        config['simulation']['seed'] = seed
        
        # Configure epsilon for online learning with high initial exploration
        config['simulation']['epsilon'] = {
            'schedule': 'exponential',
            'start': 1.0,
            'end': epsilon_min,
            'min': epsilon_min,
            'decay': 0.9985,  # Very slow decay
            'values': []
        }
        
        # Apply hyperparameters to all agents
        agent_types = ['solar', 'wind', 'battery', 'grid', 'load']
        for agent_type in agent_types:
            if agent_type in config.get('agents', {}):
                config['agents'][agent_type]['policy']['alpha'] = alpha
                config['agents'][agent_type]['policy']['gamma'] = gamma
        
        return config
    
    def compute_imbalance_metrics(self, evolution_file: Path) -> Dict[str, float]:
        """Compute energy imbalance metrics from episode evolution data.
        
        Args:
            evolution_file: Path to the episode evolution CSV file
            
        Returns:
            Dictionary with imbalance metrics
        """
        try:
            df = read_result_csv(evolution_file)
            
            if 'env_energy_balance' not in df.columns:
                return {
                    'mean_abs_imbalance': np.inf,
                    'max_abs_imbalance': np.inf,
                    'imbalance_std': np.inf,
                    'imbalance_trend': 0.0,
                    'valid': False
                }
            
            energy_balance = df['env_energy_balance'].values
            
            # Compute metrics
            mean_abs_imbalance = np.mean(np.abs(energy_balance))
            max_abs_imbalance = np.max(np.abs(energy_balance))
            imbalance_std = np.std(energy_balance)
            
            # Compute trend: negative means improvement over time
            # Use simple linear regression slope
            time_steps = np.arange(len(energy_balance))
            abs_energy_balance = np.abs(energy_balance)
            if len(time_steps) > 1:
                coeffs = np.polyfit(time_steps, abs_energy_balance, 1)
                imbalance_trend = coeffs[0]  # Slope
            else:
                imbalance_trend = 0.0
            
            # Check for improvement: trend should be negative (reducing imbalance)
            # and final imbalance should be lower than initial
            window_size = min(10, len(energy_balance) // 4)
            if window_size > 0:
                initial_imbalance = np.mean(np.abs(energy_balance[:window_size]))
                final_imbalance = np.mean(np.abs(energy_balance[-window_size:]))
                improvement_ratio = (initial_imbalance - final_imbalance) / (initial_imbalance + 1e-6)
            else:
                improvement_ratio = 0.0
            
            return {
                'mean_abs_imbalance': float(mean_abs_imbalance),
                'max_abs_imbalance': float(max_abs_imbalance),
                'imbalance_std': float(imbalance_std),
                'imbalance_trend': float(imbalance_trend),
                'improvement_ratio': float(improvement_ratio),
                'valid': True
            }
            
        except Exception as e:
            print(f"   ⚠️  Error computing metrics: {e}")
            return {
                'mean_abs_imbalance': np.inf,
                'max_abs_imbalance': np.inf,
                'imbalance_std': np.inf,
                'imbalance_trend': 0.0,
                'improvement_ratio': 0.0,
                'valid': False
            }
    
    def run_single_experiment(
        self,
        alpha: float,
        gamma: float,
        epsilon_min: float,
        episode_window_hours: int,
        seed: int,
        experiment_id: int
    ) -> Dict[str, Any]:
        """Run a single experiment with given hyperparameters.
        
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon_min: Minimum exploration rate
            episode_window_hours: Length of episode window
            seed: Random seed
            experiment_id: Unique identifier for this experiment
            
        Returns:
            Dictionary with experiment results and metrics
        """
        print(f"\n{'─'*80}")
        print(f"🧪 Experiment {experiment_id}")
        print(f"   α={alpha:.3f}, γ={gamma:.3f}, ε_min={epsilon_min:.3f}, window={episode_window_hours}h, seed={seed}")
        print(f"{'─'*80}")
        
        # Create experiment-specific temporary config
        config = self.create_experiment_config(
            alpha, gamma, epsilon_min, episode_window_hours, seed
        )
        
        # Create temporary results directory for this experiment
        temp_results = self.run_dir / f"exp_{experiment_id:04d}"
        temp_results.mkdir(parents=True, exist_ok=True)
        
        # Temporarily redirect results directory
        original_results = config['io']['results_dir']
        config['io']['results_dir'] = str(temp_results)
        
        try:
            # Run training (single episode)
            set_global_seed(seed)
            agents, results = run_training(config)
            
            # Find evolution file for the single episode
            evolution_dir = temp_results / "evolution"
            evolution_files = list(evolution_dir.glob("episode_*.csv"))
            
            if not evolution_files:
                print(f"   ⚠️  No evolution file found")
                metrics = {
                    'mean_abs_imbalance': np.inf,
                    'max_abs_imbalance': np.inf,
                    'imbalance_std': np.inf,
                    'imbalance_trend': 0.0,
                    'improvement_ratio': 0.0,
                    'valid': False
                }
            else:
                evolution_file = evolution_files[0]
                metrics = self.compute_imbalance_metrics(evolution_file)
                
                # Copy evolution file to experiment directory for later analysis
                dest_file = self.run_dir / f"exp_{experiment_id:04d}_evolution.csv"
                shutil.copy2(evolution_file, dest_file)
            
            print(f"   ✅ Completed")
            print(f"      Mean |EB|: {metrics['mean_abs_imbalance']:.2f} W (target: 0)")
            print(f"      Max |EB|:  {metrics['max_abs_imbalance']:.2f} W")
            print(f"      Trend:     {metrics['imbalance_trend']:.4f} W/step")
            print(f"      Improve:   {metrics['improvement_ratio']:.2%}")
            
            result = {
                'experiment_id': experiment_id,
                'alpha': alpha,
                'gamma': gamma,
                'epsilon_min': epsilon_min,
                'episode_window_hours': episode_window_hours,
                'seed': seed,
                **metrics,
                'success': metrics['valid']
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            return {
                'experiment_id': experiment_id,
                'alpha': alpha,
                'gamma': gamma,
                'epsilon_min': epsilon_min,
                'episode_window_hours': episode_window_hours,
                'seed': seed,
                'mean_abs_imbalance': np.inf,
                'max_abs_imbalance': np.inf,
                'imbalance_std': np.inf,
                'imbalance_trend': 0.0,
                'improvement_ratio': 0.0,
                'valid': False,
                'success': False,
                'error': str(e)
            }
        
        finally:
            # Restore original results directory
            config['io']['results_dir'] = original_results
    
    def run_search(
        self,
        episode_windows: List[int] = [96, 120, 144],
        seeds: List[int] = [42, 43, 44],
        max_experiments: int = None
    ) -> pd.DataFrame:
        """Run complete hyperparameter search.
        
        Args:
            episode_windows: List of episode window sizes to test (hours)
            seeds: List of random seeds for reproducibility
            max_experiments: Maximum number of experiments to run (None = all)
            
        Returns:
            DataFrame with all experiment results
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Hyperparameter Search")
        print(f"{'='*80}")
        
        search_space = self.define_search_space()
        
        # Generate all parameter combinations
        param_combinations = list(product(
            search_space['alpha'],
            search_space['gamma'],
            search_space['epsilon_min'],
            episode_windows,
            seeds
        ))
        
        total_experiments = len(param_combinations)
        if max_experiments:
            param_combinations = param_combinations[:max_experiments]
            total_experiments = len(param_combinations)
        
        print(f"\n📋 Experiment plan:")
        print(f"   Episode windows: {episode_windows}")
        print(f"   Seeds: {seeds}")
        print(f"   Total experiments: {total_experiments}")
        print(f"\n{'='*80}\n")
        
        # Run all experiments
        results = []
        for idx, (alpha, gamma, epsilon_min, window, seed) in enumerate(param_combinations, 1):
            result = self.run_single_experiment(
                alpha=alpha,
                gamma=gamma,
                epsilon_min=epsilon_min,
                episode_window_hours=window,
                seed=seed,
                experiment_id=idx
            )
            results.append(result)
            
            # Save intermediate results
            if idx % 10 == 0 or idx == total_experiments:
                df_results = pd.DataFrame(results)
                interim_file = self.run_dir / "results_interim.csv"
                write_result_csv(df_results, interim_file)
                print(f"\n💾 Saved interim results ({idx}/{total_experiments} experiments)")
        
        # Create final results DataFrame
        df_results = pd.DataFrame(results)
        
        # Save final results
        final_file = self.run_dir / "results_final.csv"
        write_result_csv(df_results, final_file)
        
        print(f"\n{'='*80}")
        print(f"✅ Search Completed")
        print(f"{'='*80}")
        print(f"Results saved to: {final_file}\n")
        
        return df_results
    
    def analyze_results(self, df_results: pd.DataFrame) -> None:
        """Analyze and report search results.
        
        Args:
            df_results: DataFrame with experiment results
        """
        print(f"\n{'='*80}")
        print(f"📊 RESULTS ANALYSIS")
        print(f"{'='*80}\n")
        
        # Filter valid experiments
        valid_results = df_results[df_results['valid'] == True].copy()
        
        if len(valid_results) == 0:
            print("❌ No valid experiments found")
            return
        
        print(f"Valid experiments: {len(valid_results)}/{len(df_results)}\n")
        
        # Find experiments that best minimize energy_balance (tendency to zero)
        # Primary criterion: lowest mean_abs_imbalance (closest to zero)
        # Secondary criterion: negative trend (improving over time)
        # Tertiary criterion: low standard deviation (stable)
        valid_results['score'] = -valid_results['mean_abs_imbalance'] + valid_results['imbalance_trend'] * 10000 - valid_results['imbalance_std'] / 100
        best_experiments = valid_results.nlargest(10, 'score')
        
        print("🏆 Top 10 Parameter Combinations (closest to zero energy_balance):")
        print("─" * 80)
        for rank, (idx, row) in enumerate(best_experiments.iterrows(), 1):
            print(f"\nRank #{rank}:")
            print(f"   α={row['alpha']:.3f}, γ={row['gamma']:.3f}, ε_min={row['epsilon_min']:.3f}")
            print(f"   Window: {row['episode_window_hours']}h, Seed: {row['seed']}")
            print(f"   Mean |EB|: {row['mean_abs_imbalance']:.2f} W (closer to 0 is better)")
            print(f"   Max |EB|:  {row['max_abs_imbalance']:.2f} W")
            print(f"   Std Dev:   {row['imbalance_std']:.2f} W (lower is more stable)")
            print(f"   Trend:     {row['imbalance_trend']:.4f} W/step {'✅ (improving toward 0)' if row['imbalance_trend'] < 0 else '⚠️ (diverging from 0)'}")
            print(f"   Improve:   {row['improvement_ratio']:.2%}")
        
        # Statistical analysis by parameter
        print(f"\n\n📈 Parameter Impact Analysis:")
        print("─" * 80)
        
        for param in ['alpha', 'gamma', 'epsilon_min', 'episode_window_hours']:
            print(f"\n{param}:")
            grouped = valid_results.groupby(param).agg({
                'improvement_ratio': ['mean', 'std'],
                'mean_abs_imbalance': ['mean', 'std']
            }).round(4)
            print(grouped)
        
        # Recommendations
        print(f"\n\n💡 Recommendations:")
        print("─" * 80)
        
        best_alpha = best_experiments['alpha'].mode().values[0] if len(best_experiments) > 0 else 0.35
        best_gamma = best_experiments['gamma'].mode().values[0] if len(best_experiments) > 0 else 0.90
        best_epsilon = best_experiments['epsilon_min'].mode().values[0] if len(best_experiments) > 0 else 0.15
        
        print(f"\nBest performing hyperparameters:")
        print(f"   alpha (learning rate):        {best_alpha:.3f}")
        print(f"   gamma (discount factor):      {best_gamma:.3f}")
        print(f"   epsilon_min (min exploration): {best_epsilon:.3f}")
        
        # Save analysis report
        report_file = self.run_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER SEARCH ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total experiments: {len(df_results)}\n")
            f.write(f"Valid experiments: {len(valid_results)}\n\n")
            f.write("Best parameters:\n")
            f.write(f"  alpha: {best_alpha:.3f}\n")
            f.write(f"  gamma: {best_gamma:.3f}\n")
            f.write(f"  epsilon_min: {best_epsilon:.3f}\n")
        
        print(f"\n📄 Analysis report saved to: {report_file}")
        print(f"\n{'='*80}\n")


def main():
    """Main entry point for hyperparameter search."""
    print(f"\n{'='*80}")
    print(f"🔬 MARL Microgrid - Online Q-Learning Hyperparameter Search")
    print(f"{'='*80}\n")
    
    # Initialize searcher
    searcher = HyperparameterSearcher(base_config_path="configs/default.yaml")
    
    # Define search parameters
    # Start with shorter windows for faster initial search, then expand if needed
    episode_windows = [96, 168, 240]  # Test 4 days, 7 days, 10 days
    seeds = [42, 43, 46]  # Three seeds for robustness
    
    # Run search
    # For quick testing: max_experiments=50
    # For comprehensive search: max_experiments=None (all combinations)
    df_results = searcher.run_search(
        episode_windows=episode_windows,
        seeds=seeds,
        max_experiments=1000  # Start with 100 experiments for initial exploration
    )
    
    # Analyze results
    searcher.analyze_results(df_results)
    
    print(f"\n✅ Hyperparameter search completed successfully!")
    print(f"📁 Results available in: {searcher.run_dir}\n")


if __name__ == "__main__":
    main()
