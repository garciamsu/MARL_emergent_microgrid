#!/usr/bin/env python3
"""
Test script to validate initial SOC in final episode.
Runs a minimal training session and checks the last episode CSV.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from configs.loader import load_config
from core.simulation import run_training


def main():
    print("="*80)
    print("🧪 Testing Initial SOC Configuration")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Get expected initial SOC for final episode
    expected_soc = config.get("final_episode", {}).get("initial_soc", 0.5)
    print(f"\n📋 Configuration:")
    print(f"   Expected final episode initial SOC: {expected_soc}")
    
    # Reduce episodes for quick test
    original_episodes = config["simulation"]["episodes"]
    config["simulation"]["episodes"] = 5  # Only 5 episodes for quick test
    print(f"   Running {config['simulation']['episodes']} episodes (reduced from {original_episodes})")
    
    # Run training
    print("\n🚀 Starting training...\n")
    agents, results = run_training(config)
    
    # Check last episode CSV
    last_episode_num = config["simulation"]["episodes"] - 1
    csv_path = Path(f"results/evolution/episode_{last_episode_num}.csv")
    
    if not csv_path.exists():
        print(f"\n❌ ERROR: Last episode CSV not found at {csv_path}")
        sys.exit(1)
    
    # Read CSV and check initial SOC
    df = pd.read_csv(csv_path)
    
    # Find battery SOC column
    soc_columns = [col for col in df.columns if 'soc_battery' in col and not 'idx' in col]
    
    if not soc_columns:
        print("\n❌ ERROR: No battery SOC column found in CSV")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    soc_col = soc_columns[0]
    
    # Get initial SOC (step -1 or step 0)
    initial_row = df[df['step'] == -1] if -1 in df['step'].values else df[df['step'] == 0]
    
    if initial_row.empty:
        print("\n❌ ERROR: No initial step found in CSV")
        sys.exit(1)
    
    actual_soc = initial_row[soc_col].iloc[0]
    
    print(f"\n📊 Final Episode (episode {last_episode_num}) Results:")
    print(f"   CSV Path: {csv_path}")
    print(f"   Initial SOC column: {soc_col}")
    print(f"   Expected SOC: {expected_soc:.6f}")
    print(f"   Actual SOC:   {actual_soc:.6f}")
    
    # Check tolerance
    tolerance = 0.001
    difference = abs(actual_soc - expected_soc)
    
    if difference < tolerance:
        print(f"\n✅ SUCCESS: SOC matches (difference: {difference:.6f})")
        print("="*80)
        return 0
    else:
        print(f"\n❌ FAILURE: SOC mismatch (difference: {difference:.6f})")
        print(f"   Tolerance: {tolerance}")
        print("="*80)
        
        # Print first few rows for debugging
        print("\n🔍 First 5 rows of CSV:")
        print(df[['episode', 'step', soc_col, 'epsilon']].head())
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
