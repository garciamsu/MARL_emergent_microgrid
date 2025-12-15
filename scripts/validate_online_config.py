#!/usr/bin/env python3
"""
validate_online_config.py

Quick validation script to verify online Q-learning configuration is correct.
Checks that:
- Alpha is in the correct range (0.25-0.45)
- Gamma is in the correct range (0.88-0.95)
- Epsilon starts at 1.0 and min is in range (0.10-0.20)
- Episode window is >= 96 hours
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_config


def validate_config():
    """Validate configuration for online Q-learning."""
    print("\n" + "="*80)
    print("🔍 Online Q-Learning Configuration Validation")
    print("="*80 + "\n")
    
    config = load_config("configs/default.yaml")
    
    errors = []
    warnings = []
    
    # Check epsilon configuration
    epsilon_cfg = config.get("simulation", {}).get("epsilon", {})
    eps_start = epsilon_cfg.get("start", 1.0)
    eps_min = epsilon_cfg.get("min", 0.01)
    eps_schedule = epsilon_cfg.get("schedule", "linear")
    
    print("📊 Epsilon Configuration:")
    print(f"   Schedule: {eps_schedule}")
    print(f"   Start: {eps_start}")
    print(f"   Min: {eps_min}")
    
    if eps_start != 1.0:
        warnings.append(f"Epsilon start is {eps_start}, should be 1.0 for full initial exploration")
    
    if eps_min < 0.10 or eps_min > 0.20:
        warnings.append(f"Epsilon min is {eps_min}, recommended range is 0.10-0.20")
    
    if eps_schedule == "constant" and eps_start < 1.0:
        warnings.append(f"Constant epsilon schedule without initial exploration phase")
    
    # Check episode window
    episode_window = config.get("simulation", {}).get("episode_window_hours", 48)
    print(f"\n📏 Episode Window: {episode_window} hours")
    
    if episode_window < 96:
        warnings.append(f"Episode window is {episode_window}h, recommended >= 96h for online learning")
    
    # Check agent configurations
    agents_cfg = config.get("agents", {})
    agent_types = ["solar", "wind", "battery", "grid", "load"]
    
    print(f"\n👥 Agent Configurations:")
    
    all_alphas = []
    all_gammas = []
    
    for agent_type in agent_types:
        if agent_type not in agents_cfg:
            warnings.append(f"Agent '{agent_type}' not found in configuration")
            continue
        
        policy_cfg = agents_cfg[agent_type].get("policy", {})
        alpha = policy_cfg.get("alpha", 0.1)
        alpha_min = policy_cfg.get("alpha_min", None)
        gamma = policy_cfg.get("gamma", 0.9)
        
        all_alphas.append(alpha)
        all_gammas.append(gamma)
        
        print(f"\n   {agent_type}:")
        print(f"      α (alpha): {alpha}")
        if alpha_min is not None:
            print(f"      α_min:     {alpha_min}")
        print(f"      γ (gamma): {gamma}")
        
        # Validate alpha
        if alpha < 0.25 or alpha > 0.45:
            warnings.append(f"{agent_type}: alpha={alpha} outside recommended range [0.25, 0.45]")
        
        # Validate gamma
        if gamma < 0.88 or gamma > 0.95:
            warnings.append(f"{agent_type}: gamma={gamma} outside recommended range [0.88, 0.95]")
        
        # Check alpha_min if present
        if alpha_min is not None:
            if alpha_min < 0.10 or alpha_min > 0.20:
                warnings.append(f"{agent_type}: alpha_min={alpha_min} outside recommended range [0.10, 0.20]")
    
    # Summary statistics
    if all_alphas:
        avg_alpha = sum(all_alphas) / len(all_alphas)
        avg_gamma = sum(all_gammas) / len(all_gammas)
        
        print(f"\n📈 Average across agents:")
        print(f"   α (alpha): {avg_alpha:.3f}")
        print(f"   γ (gamma): {avg_gamma:.3f}")
    
    # Print results
    print(f"\n{'='*80}")
    print("📋 Validation Results")
    print("="*80 + "\n")
    
    if not errors and not warnings:
        print("✅ Configuration is VALID for online Q-learning")
        print("\nRecommended ranges:")
        print("   ✓ Alpha:       0.25-0.45 (default: 0.35)")
        print("   ✓ Gamma:       0.88-0.95 (default: 0.90-0.92)")
        print("   ✓ Epsilon min: 0.10-0.20 (default: 0.15)")
        print("   ✓ Window:      >= 96 hours")
        status = 0
    else:
        if errors:
            print("❌ ERRORS found:")
            for error in errors:
                print(f"   • {error}")
            print()
            status = 1
        
        if warnings:
            print("⚠️  WARNINGS:")
            for warning in warnings:
                print(f"   • {warning}")
            print()
            status = 1 if errors else 0
        
        print("\nRecommended actions:")
        print("   1. Review docs/ONLINE_QLEARNING_CONFIG.md")
        print("   2. Adjust configs/default.yaml as needed")
        print("   3. Run this script again to validate")
    
    print("="*80 + "\n")
    
    return status


if __name__ == "__main__":
    sys.exit(validate_config())
