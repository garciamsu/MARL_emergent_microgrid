#!/usr/bin/env python3
"""
run_full_pipeline.py

Executes the complete analysis pipeline in sequence:
1. A_data_check.py - Verify dataset integrity
2. B_run_training.py - Run training simulation
3. C_collect_episodes.py - Collect episode data
4. D_compute_metrics.py - Compute performance metrics
5. E_accumulated_reward.py - Comprehensive reward analysis & visualization
6. E_graph_episode.py - Generate episode graphs

Each script is executed in order, and the pipeline stops if any script fails.
"""
"""

import os
import sys
import subprocess
from pathlib import Path


def run_script(script_name: str, script_path: Path) -> bool:
    """Execute a script and return True if successful, False otherwise.
    
    Args:
        script_name: Display name of the script
        script_path: Full path to the script file
        
    Returns:
        bool: True if script executed successfully, False otherwise
    """
    print("\n" + "="*80)
    print(f"▶️  Running: {script_name}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent.parent),
            check=True,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {script_name} completed successfully")
            return True
        else:
            print(f"\n⚠️  {script_name} finished with code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR executing {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: {script_name} not found at {script_path}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR in {script_name}: {e}")
        return False


def main():
    """Execute the complete analysis pipeline."""
    print("="*80)
    print("🚀 MARL Microgrid - Full Analysis Pipeline")
    print("="*80)
    print("\nThis pipeline will execute the following scripts in sequence:")
    print("  1. A_data_check.py           - Dataset verification")
    print("  2. B_run_training.py         - Training execution")
    print("  3. C_collect_episodes.py     - Episode collection")
    print("  4. D_compute_metrics.py      - Metrics computation")
    print("  5. E_accumulated_reward.py   - Reward analysis & visualization")
    print("  6. E_graph_episode.py        - Episode graphs")
    print("\n" + "="*80)
    
    # Define scripts directory
    scripts_dir = Path(__file__).parent
    
    # Define pipeline scripts in execution order
    pipeline = [
        ("A_data_check.py", "Data Check"),
        ("B_run_training.py", "Training"),
        ("C_collect_episodes.py", "Collect Episodes"),
        ("D_compute_metrics.py", "Compute Metrics"),
        ("E_accumulated_reward.py", "Reward Analysis & Visualization"),
        ("E_graph_episode.py", "Episode Graphs")
    ]
    
    # Track execution results
    results = []
    
    # Execute each script in sequence
    for script_file, script_name in pipeline:
        script_path = scripts_dir / script_file
        
        success = run_script(script_name, script_path)
        results.append((script_name, success))
        
        if not success:
            print("\n" + "="*80)
            print(f"❌ Pipeline FAILED at: {script_name}")
            print("   Stopping execution.")
            print("="*80)
            
            # Show summary of completed steps
            print("\n📊 Execution Summary:")
            for step_name, step_success in results:
                status = "✅" if step_success else "❌"
                print(f"   {status} {step_name}")
            
            sys.exit(1)
    
    # All scripts completed successfully
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\n📊 All steps executed:")
    for step_name, step_success in results:
        print(f"   ✅ {step_name}")
    
    print("\n📁 Results available in:")
    print("   - results/evolution/     (episode CSVs)")
    print("   - results/logs/          (training logs + episode_metadata.xlsx + episode_rewards.csv)")
    print("   - results/metrics/       (metrics Excel)")
    print("   - results/plots/         (visualizations + episode reward plots)")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
