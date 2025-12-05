"""
Verification script for episode reward calculation.

This script validates that episode rewards are computed correctly by:
1. Checking that episode_rewards.csv matches manual sum from episode CSVs
2. Verifying data format and structure
3. Ensuring no accumulation across episodes
"""

import pandas as pd
import os
import glob
import numpy as np


def verify_episode_rewards():
    """
    Verify that episode rewards are calculated correctly.
    """
    print("\n" + "="*70)
    print("🔍 EPISODE REWARD VERIFICATION")
    print("="*70)
    
    # Check if episode_rewards.csv exists
    rewards_file = "results/logs/episode_rewards.csv"
    if not os.path.exists(rewards_file):
        print(f"\n❌ Episode rewards file not found: {rewards_file}")
        print("Please run training first to generate the file.")
        return False
    
    # Load episode rewards
    print(f"\n✅ Found episode rewards file: {rewards_file}")
    episode_rewards = pd.read_csv(rewards_file)
    
    print(f"   Shape: {episode_rewards.shape}")
    print(f"   Columns: {list(episode_rewards.columns)}")
    
    # Verify structure
    if 'episode' not in episode_rewards.columns:
        print("\n❌ ERROR: Missing 'episode' column")
        return False
    
    agent_columns = [col for col in episode_rewards.columns if col != 'episode']
    print(f"   Agents: {agent_columns}")
    
    # Check episode numbering
    expected_episodes = set(range(len(episode_rewards)))
    actual_episodes = set(episode_rewards['episode'].values)
    
    if expected_episodes != actual_episodes:
        print(f"\n❌ ERROR: Episode numbering mismatch")
        print(f"   Expected: {sorted(expected_episodes)[:10]}...")
        print(f"   Actual: {sorted(actual_episodes)[:10]}...")
        return False
    
    print(f"\n✅ Episode numbering correct: 0 to {len(episode_rewards) - 1}")
    
    # Verify against individual episode files
    print("\n" + "="*70)
    print("🔍 VERIFYING AGAINST INDIVIDUAL EPISODE FILES")
    print("="*70)
    
    evolution_dir = "results/evolution"
    if not os.path.exists(evolution_dir):
        print(f"\n⚠️  Evolution directory not found: {evolution_dir}")
        print("Skipping detailed verification.")
        return True
    
    # Check a sample of episodes
    episode_files = sorted(glob.glob(f"{evolution_dir}/episode_*.csv"))
    num_files = len(episode_files)
    
    if num_files == 0:
        print(f"\n⚠️  No episode files found in {evolution_dir}")
        return True
    
    print(f"\n✅ Found {num_files} episode files")
    
    # Sample episodes to verify (first, middle, last, and a few random)
    sample_indices = [0]
    if num_files > 1:
        sample_indices.append(num_files - 1)
    if num_files > 2:
        sample_indices.append(num_files // 2)
    if num_files > 10:
        # Add a few random samples
        np.random.seed(42)
        extra_samples = np.random.choice(range(1, num_files - 1), 
                                         size=min(5, num_files - 2), 
                                         replace=False)
        sample_indices.extend(extra_samples)
    
    sample_indices = sorted(set(sample_indices))
    
    print(f"\n🔍 Verifying {len(sample_indices)} sample episodes: {sample_indices}")
    
    all_valid = True
    tolerance = 1e-6
    
    for ep_num in sample_indices:
        ep_file = f"{evolution_dir}/episode_{ep_num}.csv"
        
        if not os.path.exists(ep_file):
            print(f"  ⚠️  Episode {ep_num}: File not found")
            continue
        
        # Load episode evolution data
        ep_df = pd.read_csv(ep_file)
        
        # Find reward columns in episode file
        reward_cols = [col for col in ep_df.columns if col.startswith('reward_')]
        
        if not reward_cols:
            print(f"  ⚠️  Episode {ep_num}: No reward columns found")
            continue
        
        # Verify each agent's reward
        episode_valid = True
        for reward_col in reward_cols:
            agent_name = reward_col.replace('reward_', '')
            
            # Skip if this agent not in episode_rewards file
            if agent_name not in agent_columns:
                continue
            
            # Calculate manual sum (excluding initial state at step -1 if present)
            episode_data = ep_df[ep_df['step'] >= 0]
            manual_sum = episode_data[reward_col].sum()
            
            # Get stored value
            stored_value = episode_rewards.loc[
                episode_rewards['episode'] == ep_num, 
                agent_name
            ].values
            
            if len(stored_value) == 0:
                print(f"  ❌ Episode {ep_num}, {agent_name}: Not found in episode_rewards.csv")
                episode_valid = False
                all_valid = False
                continue
            
            stored_value = stored_value[0]
            
            # Compare
            diff = abs(manual_sum - stored_value)
            
            if diff > tolerance:
                print(f"  ❌ Episode {ep_num}, {agent_name}: MISMATCH")
                print(f"     Manual sum: {manual_sum:.6f}")
                print(f"     Stored:     {stored_value:.6f}")
                print(f"     Difference: {diff:.6f}")
                episode_valid = False
                all_valid = False
            else:
                print(f"  ✅ Episode {ep_num}, {agent_name}: Match ({stored_value:.3f})")
        
        if not episode_valid:
            print(f"  ❌ Episode {ep_num}: Validation FAILED")
    
    # Final result
    print("\n" + "="*70)
    if all_valid:
        print("✅ VERIFICATION PASSED")
        print("="*70)
        print("\nAll checked episodes match:")
        print("  - Episode rewards = sum of timestep rewards within episode")
        print("  - No accumulation across episodes detected")
        print("  - Data structure is correct")
        return True
    else:
        print("❌ VERIFICATION FAILED")
        print("="*70)
        print("\nSome episodes have mismatched rewards.")
        print("Please check the implementation in core/simulation.py")
        return False


def print_statistics():
    """
    Print basic statistics about episode rewards.
    """
    rewards_file = "results/logs/episode_rewards.csv"
    
    if not os.path.exists(rewards_file):
        return
    
    df = pd.read_csv(rewards_file)
    agent_columns = [col for col in df.columns if col != 'episode']
    
    print("\n" + "="*70)
    print("📊 EPISODE REWARD STATISTICS")
    print("="*70)
    
    stats = df[agent_columns].describe()
    print("\n" + stats.to_string())
    
    # Check for monotonic increase (would indicate wrong implementation)
    print("\n" + "="*70)
    print("🔍 CHECKING FOR INCORRECT ACCUMULATION")
    print("="*70)
    
    for agent in agent_columns:
        values = df[agent].values
        
        # Check if strictly increasing (sign of accumulation bug)
        is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
        
        if is_monotonic and len(values) > 5:
            print(f"  ⚠️  {agent}: Values are monotonically increasing!")
            print(f"     This suggests rewards may be accumulating across episodes.")
        else:
            print(f"  ✅ {agent}: Values fluctuate (correct behavior)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    success = verify_episode_rewards()
    print_statistics()
    
    if not success:
        exit(1)
    else:
        print("\n✅ All verifications passed!")
        exit(0)
