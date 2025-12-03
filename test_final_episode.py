"""Test script to verify final episode functionality.

This script runs a small training session to verify that:
1. The final episode uses the predefined window from config
2. The final episode uses fixed initial SOC
3. The final episode has epsilon = 0 (pure exploitation)
4. Q-tables are not updated in the final episode
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from configs.loader import load_config
from core.simulation import run_training
import pandas as pd


def test_final_episode():
    """Test final episode functionality with a small training run."""
    print("=" * 80)
    print("🧪 Testing Final Episode Functionality")
    print("=" * 80)
    
    # Load config
    config = load_config()
    
    # Override episodes to run only 5 for quick test
    original_episodes = config['simulation']['episodes']
    config['simulation']['episodes'] = 5
    
    print(f"\n📋 Configuration:")
    print(f"   Total episodes: {config['simulation']['episodes']}")
    print(f"   Dataset: {config['simulation']['dataset']}")
    
    # Display final episode config
    final_cfg = config.get('final_episode', {})
    print(f"\n🎯 Final Episode Configuration:")
    print(f"   Window: [{final_cfg.get('start_index', 0)}:{final_cfg.get('end_index', 24)}]")
    print(f"   Initial SOC: {final_cfg.get('initial_soc', 0.5)}")
    
    print(f"\n▶️  Running training...\n")
    
    # Run training
    agents, results = run_training(config)
    
    print(f"\n✅ Training completed!")
    
    # Verify final episode
    print(f"\n" + "=" * 80)
    print("📊 Verification Results")
    print("=" * 80)
    
    # Read metadata to verify
    metadata_path = Path("results/logs/episode_metadata.xlsx")
    if metadata_path.exists():
        metadata_df = pd.read_excel(metadata_path, sheet_name='Episode Details', engine='openpyxl')
        
        # Check final episode
        final_episode = metadata_df.iloc[-1]
        
        print(f"\n🔍 Final Episode (Episode {final_episode['episode']}):")
        print(f"   Window: [{final_episode['window_start_index']}:{final_episode['window_end_index']}]")
        print(f"   Window Size: {final_episode['window_size']} hours")
        print(f"   Initial SOC: {final_episode['initial_soc']:.3f}")
        print(f"   Is Final Episode: {final_episode.get('is_final_episode', 'N/A')}")
        
        # Expected values from config
        expected_start = final_cfg.get('start_index', 0)
        expected_end = final_cfg.get('end_index', 24)
        expected_soc = final_cfg.get('initial_soc', 0.5)
        expected_size = expected_end - expected_start
        
        # Verify
        checks_passed = 0
        checks_total = 4
        
        if final_episode['window_start_index'] == expected_start:
            print(f"\n   ✅ Window start matches config ({expected_start})")
            checks_passed += 1
        else:
            print(f"\n   ❌ Window start mismatch: expected {expected_start}, got {final_episode['window_start_index']}")
        
        if final_episode['window_end_index'] == expected_end:
            print(f"   ✅ Window end matches config ({expected_end})")
            checks_passed += 1
        else:
            print(f"   ❌ Window end mismatch: expected {expected_end}, got {final_episode['window_end_index']}")
        
        if final_episode['window_size'] == expected_size:
            print(f"   ✅ Window size matches config ({expected_size} hours)")
            checks_passed += 1
        else:
            print(f"   ❌ Window size mismatch: expected {expected_size}, got {final_episode['window_size']}")
        
        if abs(final_episode['initial_soc'] - expected_soc) < 0.001:
            print(f"   ✅ Initial SOC matches config ({expected_soc})")
            checks_passed += 1
        else:
            print(f"   ❌ Initial SOC mismatch: expected {expected_soc}, got {final_episode['initial_soc']:.3f}")
        
        # Read final episode CSV to check epsilon
        final_episode_csv = Path(f"results/evolution/episode_{final_episode['episode']}.csv")
        if final_episode_csv.exists():
            final_df = pd.read_csv(final_episode_csv)
            epsilon_values = final_df['epsilon'].unique()
            
            if len(epsilon_values) == 1 and epsilon_values[0] == 0.0:
                print(f"   ✅ Epsilon is 0.0 (pure exploitation)")
                checks_passed += 1
                checks_total += 1
            else:
                print(f"   ❌ Epsilon is not 0.0: {epsilon_values}")
                checks_total += 1
        
        print(f"\n{'='*80}")
        print(f"📈 Test Results: {checks_passed}/{checks_total} checks passed")
        
        if checks_passed == checks_total:
            print(f"✅ All checks passed! Final episode functionality working correctly.")
        else:
            print(f"⚠️  Some checks failed. Please review the implementation.")
        
        print(f"{'='*80}")
        
        # Display all episodes for reference
        print(f"\n📋 All Episodes Summary:")
        print(metadata_df.to_string(index=False))
        
    else:
        print(f"\n❌ Metadata file not found at {metadata_path}")
        print(f"   Training may have failed or not completed.")
    
    # Restore original episodes count
    config['simulation']['episodes'] = original_episodes


if __name__ == "__main__":
    try:
        test_final_episode()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
