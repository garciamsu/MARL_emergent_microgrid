"""
Validation script for Load Agent configuration and behavior.

This script analyzes:
1. Coherence of p_load with dataset demand range
2. Impact of different comfort_threshold values on reward function
3. Simulation of agent behavior under various scenarios

Usage:
    python scripts/validate_load_agent.py
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_config():
    """Load configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(config):
    """Load and scale dataset."""
    dataset_name = config['simulation']['dataset']
    scale_factor = config['simulation'].get('power_scale_factor', 1000.0)
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'datasets', dataset_name)
    df = read_dataset_csv(dataset_path)
    
    # Scale power columns
    for col in df.columns:
        if col not in ["price", "Datetime", "datetime"]:
            if "power" in col.lower() or col.lower() == "demand":
                df[col] = df[col] * scale_factor
    
    return df


def analyze_demand_coherence(df, p_load):
    """Analyze if p_load is coherent with dataset demand."""
    print("=" * 80)
    print("1. DEMAND COHERENCE ANALYSIS")
    print("=" * 80)
    
    demand_min = df['demand'].min()
    demand_max = df['demand'].max()
    demand_mean = df['demand'].mean()
    
    print(f"\nDataset demand statistics (after scaling):")
    print(f"  Minimum: {demand_min:8.0f} W")
    print(f"  Maximum: {demand_max:8.0f} W")
    print(f"  Average: {demand_mean:8.0f} W")
    
    print(f"\nConfigured p_load: {p_load} W")
    print(f"  As % of average demand: {(p_load/demand_mean)*100:6.2f}%")
    print(f"  As % of minimum demand: {(p_load/demand_min)*100:6.2f}%")
    print(f"  As % of maximum demand: {(p_load/demand_max)*100:6.2f}%")
    
    # Check for potential negative demand
    negative_demand_hours = (df['demand'] - p_load < 0).sum()
    
    if negative_demand_hours > 0:
        print(f"\n⚠️  WARNING: p_load > demand in {negative_demand_hours} hours ({negative_demand_hours/len(df)*100:.1f}%)")
        print(f"    When action=0 (OFF), effective demand would be NEGATIVE")
        print(f"    This is physically inconsistent!")
        print(f"\n✅  RECOMMENDATION: Set p_load < {demand_min:.0f} W (current minimum demand)")
        print(f"    Suggested safe value: {int(demand_min * 0.9)} W (90% of minimum)")
    else:
        print(f"\n✅  OK: p_load is always less than demand (no negative demand risk)")
    
    return demand_min, demand_mean, demand_max


def analyze_price_threshold(df, comfort_threshold):
    """Analyze comfort_threshold vs dataset price distribution."""
    print("\n" + "=" * 80)
    print("2. COMFORT THRESHOLD ANALYSIS")
    print("=" * 80)
    
    price_min = df['price'].min()
    price_max = df['price'].max()
    price_mean = df['price'].mean()
    price_median = df['price'].median()
    
    percentiles = [25, 50, 75, 90, 95]
    
    print(f"\nDataset price statistics (EUR/MWh):")
    print(f"  Minimum:  {price_min:6.2f}")
    print(f"  Maximum:  {price_max:6.2f}")
    print(f"  Average:  {price_mean:6.2f}")
    print(f"  Median:   {price_median:6.2f}")
    
    print(f"\nPrice percentiles:")
    for p in percentiles:
        val = df['price'].quantile(p/100)
        print(f"  P{p:2d}: {val:6.2f} EUR/MWh")
    
    print(f"\nConfigured comfort_threshold: {comfort_threshold} EUR/MWh")
    
    # Calculate percentage of time price is below threshold
    pct_below = (df['price'] < comfort_threshold).sum() / len(df) * 100
    pct_above = 100 - pct_below
    
    print(f"  Price < threshold: {pct_below:5.1f}% of time")
    print(f"  Price > threshold: {pct_above:5.1f}% of time")
    
    if pct_above < 1:
        print(f"\n⚠️  WARNING: Price is ALMOST NEVER above threshold")
        print(f"    The 'expensive' condition in reward function rarely/never activates")
        print(f"    Agent cannot learn price-based decision making!")
        print(f"\n✅  RECOMMENDATION: Set comfort_threshold within dataset price range")
        print(f"    Suggested values:")
        print(f"      - {df['price'].quantile(0.67):.1f} EUR/MWh (P67, ~33% expensive)")
        print(f"      - {df['price'].quantile(0.75):.1f} EUR/MWh (P75, ~25% expensive)")
        print(f"      - {df['price'].quantile(0.80):.1f} EUR/MWh (P80, ~20% expensive)")
    elif pct_above > 50:
        print(f"\n⚠️  WARNING: Price is above threshold more than 50% of time")
        print(f"    Load will be frequently penalized for turning ON")
        print(f"    Consider increasing threshold")
    else:
        print(f"\n✅  OK: Threshold creates meaningful price-based learning signal")
    
    return price_mean, price_max


def simulate_reward_scenarios(comfort_threshold):
    """Simulate reward function behavior under different scenarios."""
    print("\n" + "=" * 80)
    print("3. REWARD FUNCTION SIMULATION")
    print("=" * 80)
    
    # Reward parameters from config
    sigma = 1.0
    psi = 1.0
    nu = 1.0
    beta = 0.0
    
    scenarios = [
        # (action, soc_idx, renewable_idx, demand_idx, price, description)
        (1, 3, 8, 5, 15.0, "ON with surplus renewable, cheap price"),
        (1, 3, 5, 8, 15.0, "ON with high SOC but deficit, cheap price"),
        (1, 0, 5, 8, 15.0, "ON with deficit and low SOC, cheap price"),
        (1, 0, 5, 8, 35.0, "ON with deficit and low SOC, EXPENSIVE price"),
        (0, 3, 8, 5, 15.0, "OFF with surplus renewable, cheap price"),
        (0, 0, 5, 8, 35.0, "OFF with deficit and low SOC, EXPENSIVE price"),
        (0, 0, 5, 8, 15.0, "OFF with deficit and low SOC, cheap price"),
    ]
    
    print(f"\nReward parameters: sigma={sigma}, psi={psi}, nu={nu}, beta={beta}")
    print(f"Comfort threshold: {comfort_threshold} EUR/MWh")
    print(f"\nAction: 1=ON, 0=OFF")
    print("\n{:<50} {:>10} {:>12}".format("Scenario", "Reward", "Decision"))
    print("-" * 80)
    
    for action, soc_idx, renewable_idx, demand_idx, price, desc in scenarios:
        surplus = (renewable_idx > demand_idx)
        expensive = (price > comfort_threshold)
        internal = (soc_idx > 1 or surplus)
        
        # Compute reward (replica of DefaultLoadReward logic)
        if action == 1 and internal:
            reward = sigma
            decision = "✅ Correct"
        elif action == 1 and expensive:
            reward = -psi
            decision = "❌ Penalized"
        elif action == 0 and internal:
            reward = -nu
            decision = "⚠️  Missed opp"
        elif action == 0 and expensive:
            reward = sigma
            decision = "✅ Correct"
        else:
            reward = beta
            decision = "➖ Neutral"
        
        # Format scenario description
        state_desc = f"soc={soc_idx}, ren={renewable_idx}, dem={demand_idx}, p={price:.0f}"
        full_desc = f"[{action}] {desc} ({state_desc})"
        
        print(f"{full_desc:<50} {reward:>10.2f} {decision:>12}")


def recommend_optimal_values(demand_min, demand_mean, price_data):
    """Provide final recommendations."""
    print("\n" + "=" * 80)
    print("4. FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    # Recommended p_load
    recommended_p_load = int(demand_min * 0.9)
    safe_p_load = int(demand_min * 0.8)
    
    print(f"\n📊 p_load recommendations:")
    print(f"  Current:     3000 W")
    print(f"  Recommended: {recommended_p_load} W (90% of min demand, safe margin)")
    print(f"  Conservative: {safe_p_load} W (80% of min demand, extra safety)")
    print(f"  Alternative: 2000 W (24% of avg demand, good balance)")
    
    # Recommended comfort_threshold
    p67 = price_data.quantile(0.67)
    p75 = price_data.quantile(0.75)
    p80 = price_data.quantile(0.80)
    
    print(f"\n💰 comfort_threshold recommendations:")
    print(f"  Current:     100 EUR/MWh (ineffective - never triggers)")
    print(f"  Recommended: {p67:.1f} EUR/MWh (P67, balanced learning)")
    print(f"  Alternative: {p75:.1f} EUR/MWh (P75, conservative)")
    print(f"  Alternative: {p80:.1f} EUR/MWh (P80, permissive)")
    
    print(f"\n🎯 Implementation steps:")
    print(f"  1. Edit configs/default.yaml")
    print(f"  2. Update agents.load.limits.p_load to {recommended_p_load}")
    print(f"  3. Update agents.load.limits.comfort_threshold to {p67:.1f}")
    print(f"  4. Re-train agents to learn new behavior")
    print(f"  5. Validate results using analysis scripts")


def main():
    """Main validation routine."""
    print("\n" + "=" * 80)
    print("LOAD AGENT CONFIGURATION VALIDATION")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    p_load = config['agents']['load']['limits']['p_load']
    comfort_threshold = config['agents']['load']['limits']['comfort_threshold']
    
    print(f"\nLoaded configuration:")
    print(f"  Dataset: {config['simulation']['dataset']}")
    print(f"  Power scale factor: {config['simulation']['power_scale_factor']}")
    print(f"  p_load: {p_load} W")
    print(f"  comfort_threshold: {comfort_threshold} EUR/MWh")
    
    # Load dataset
    df = load_dataset(config)
    print(f"\nDataset loaded: {len(df)} rows")
    
    # Run analyses
    demand_min, demand_mean, demand_max = analyze_demand_coherence(df, p_load)
    price_mean, price_max = analyze_price_threshold(df, comfort_threshold)
    simulate_reward_scenarios(comfort_threshold)
    recommend_optimal_values(demand_min, demand_mean, df['price'])
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
