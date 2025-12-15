# Online Q-Learning Configuration Guide

## Overview

This document describes the configuration for **online, non-stationary Q-learning** without offline pretraining. The system is configured to remain reactive and adaptive to changing environment conditions through high, mostly constant learning rates and persistent exploration.

## Configuration Philosophy

The configuration is designed for:
- **Real-time adaptation** to non-stationary environments
- **No offline pretraining** phase
- **Continuous learning** throughout operation
- **Persistent exploration** that never completely disables

## Hyperparameters

### Learning Rate (Alpha)

**Range**: 0.25 – 0.45  
**Default**: 0.35  
**Policy**: Moderately high and mostly constant

The learning rate determines how quickly agents update their Q-values based on new experiences.

- **High values** (0.35-0.45): Enable quick adaptation to changing conditions
- **Constant policy**: Does not anneal to zero, maintaining reactivity
- **Minimum floor** (`alpha_min ≈ 0.15`): If scheduled, never drops below this threshold

**Rationale**: In non-stationary environments, recent experiences are more valuable than historical averages. A high, constant learning rate ensures the system can quickly adjust to new patterns without getting stuck on outdated policies.

**Configuration** (in `configs/default.yaml`):
```yaml
agents:
  solar:
    policy:
      alpha: 0.35        # High constant learning rate for online adaptation
      alpha_min: 0.15    # Minimum alpha if scheduled (not annealed to zero)
      gamma: 0.90
```

### Discount Factor (Gamma)

**Range**: 0.88 – 0.95  
**Default**: 0.90 – 0.92  
**Policy**: Mid-to-high, avoiding over-trusting long horizons

The discount factor balances immediate vs. future rewards.

- **Mid-high values** (0.90-0.92): Consider future consequences without over-weighting distant predictions
- **Not too high**: In changing environments, distant future estimates are unreliable

**Rationale**: While agents should consider future impacts of current actions, over-trusting long-horizon estimates in a changing environment can lead to poor decisions based on outdated patterns.

**Configuration**:
```yaml
agents:
  solar:
    policy:
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90        # Mid-high discount for non-stationary env
```

### Exploration Rate (Epsilon)

**Initial**: 1.0 (100% exploration)  
**Minimum**: 0.10 – 0.20  
**Default Minimum**: 0.15  
**Decay**: Very slow (exponential with factor ≈ 0.9985)

The exploration rate controls the balance between exploration (trying new actions) and exploitation (using learned knowledge).

- **High initial exploration** (ε=1.0): Ensures thorough initial learning
- **Very slow decay**: Maintains high exploration for extended periods
- **Never disabled**: Minimum exploration (0.10-0.20) remains constant during online operation
- **No complete annealing**: Epsilon never reaches 0

**Rationale**: In online learning without pretraining, the system must continuously explore to:
1. Learn from scratch during initial operation
2. Detect and adapt to changing conditions
3. Avoid getting trapped in locally optimal but globally suboptimal policies

**Configuration**:
```yaml
simulation:
  epsilon:
    schedule: exponential     # Exponential decay (very slow)
    start: 1.0                # Full exploration initially
    end: 0.15                 # Target minimum (10-20% range)
    min: 0.15                 # Enforce minimum (never below this)
    decay: 0.9985             # Very slow decay factor
    values: []                # Not used for exponential schedule
```

## Episode Configuration for Online Learning

### Episode Window

**Default**: 96 hours (4 days)  
**Recommended for experiments**: 96, 120, 144+ hours

Longer episode windows are crucial for online learning:
- More data points for learning patterns
- Better evaluation of long-term policy impact
- Sufficient time to observe imbalance reduction

**Configuration**:
```yaml
simulation:
  episode_window_hours: 96    # Long episodes for online learning evaluation
```

### Number of Episodes

For online learning evaluation:
- **Training**: Multiple episodes to accumulate experience
- **Hyperparameter search**: Single episode per configuration (faster evaluation)

**Configuration**:
```yaml
simulation:
  episodes: 1500              # Training mode
  # or
  episodes: 1                 # Single-episode evaluation
```

## All Agents Configuration

All agents (solar, wind, battery, grid, load) use the same Q-learning hyperparameters:

```yaml
agents:
  solar:
    policy:
      type: tabular_ql
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90

  wind:
    policy:
      type: tabular_ql
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90

  battery:
    policy:
      type: tabular_ql
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90

  grid:
    policy:
      type: tabular_ql
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90

  load:
    policy:
      type: tabular_ql
      alpha: 0.35
      alpha_min: 0.15
      gamma: 0.90
```

## Hyperparameter Search

### Purpose

The `scripts/hyperparameter_search.py` script enables systematic search for optimal (alpha, gamma, epsilon_min) combinations under strictly online learning conditions.

### Key Features

1. **Black-box wrapper**: Does not modify internal training logic
2. **Configuration injection**: Parameters set only through config interface
3. **Single-episode experiments**: Quick evaluation per configuration
4. **Long episode windows**: Tests with 96h, 120h, 144h+ windows
5. **Imbalance metrics**: Evaluates energy balance improvement over time

### Usage

```bash
# Run full hyperparameter search
python scripts/hyperparameter_search.py

# Results will be saved to:
# results/experiments/hyperparameter_search/run_YYYYMMDD_HHMMSS/
```

### Search Space

Default search space:
- **alpha**: [0.25, 0.30, 0.35, 0.40, 0.45]
- **gamma**: [0.88, 0.90, 0.92, 0.95]
- **epsilon_min**: [0.10, 0.12, 0.15, 0.18, 0.20]
- **episode_windows**: [96, 120]
- **seeds**: [42, 43, 44]

Total: 5 × 4 × 5 × 2 × 3 = **600 experiments** (by default)

### Evaluation Metrics

Each experiment is evaluated on:
- **Mean absolute imbalance**: Average |Δ_ph| over episode
- **Maximum absolute imbalance**: Peak |Δ_ph|
- **Imbalance standard deviation**: Stability measure
- **Imbalance trend**: Slope of |Δ_ph| over time (negative = improving)
- **Improvement ratio**: (initial - final) / initial imbalance

### Output Files

```
results/experiments/hyperparameter_search/run_YYYYMMDD_HHMMSS/
├── results_final.csv          # All experiment results
├── results_interim.csv        # Intermediate results (saved every 10 exp)
├── analysis_report.txt        # Summary report with recommendations
├── exp_0001_evolution.csv     # Episode data for experiment 1
├── exp_0002_evolution.csv     # Episode data for experiment 2
└── ...
```

### Interpreting Results

The script automatically identifies best-performing configurations based on:
1. **Improvement ratio** (higher is better): Measures imbalance reduction
2. **Mean absolute imbalance** (lower is better): Overall balance quality
3. **Negative trend** (required): Confirms learning is occurring

Example output:
```
🏆 Top 10 Parameter Combinations (by imbalance reduction):
────────────────────────────────────────────────────────────────────────────────

Rank #1:
   α=0.350, γ=0.900, ε_min=0.150
   Window: 96h, Seed: 42
   Mean |Δ|: 12543.21 W
   Max |Δ|:  45231.87 W
   Trend:    -23.4567 W/step ✅ (improving)
   Improve:  18.45%
```

## Migration from Previous Configuration

### Changes Made

1. **Alpha**: 0.3 → 0.35 (with alpha_min=0.15)
2. **Gamma**: 0.85 → 0.90
3. **Epsilon schedule**: 
   - start: 0.3 → 1.0
   - end: 0.3 → 0.15
   - decay: null → 0.9985
   - min: (new) 0.15

### Rationale

Previous configuration (alpha=0.3, gamma=0.85, epsilon=0.3 constant):
- **Too conservative** for learning from scratch
- **No initial exploration phase**: Started at ε=0.3 instead of ε=1.0
- **Constant epsilon too high**: 30% exploration throughout (should start high, decay slowly)
- **Lower learning rate**: Less reactive to changes

New configuration:
- **Higher reactivity**: alpha=0.35 enables faster adaptation
- **Better future consideration**: gamma=0.90 balances immediate vs. long-term
- **Proper exploration schedule**: Starts at 100%, decays very slowly to 15%
- **Persistent exploration**: Never drops below 15% during online operation

## Validation

To validate the configuration:

1. **Run self-check**:
   ```bash
   python scripts/self_check.py
   ```

2. **Run short training**:
   ```bash
   # Modify configs/default.yaml: simulation.episodes = 50
   python main.py
   ```

3. **Check for imbalance reduction**:
   - Monitor `results/logs/episode_rewards.csv`
   - Check if mean absolute delta_ph decreases over episodes
   - Verify epsilon starts at 1.0 and decays slowly

4. **Run hyperparameter search** (optional):
   ```bash
   python scripts/hyperparameter_search.py
   ```

## Best Practices

### For Training

1. **Start with default values**: alpha=0.35, gamma=0.90, epsilon_min=0.15
2. **Use long episodes**: >= 96 hours for meaningful evaluation
3. **Monitor exploration**: Check epsilon values in logs
4. **Track imbalance**: Watch for decreasing trend in |Δ_ph|

### For Hyperparameter Tuning

1. **Run systematic search**: Use `hyperparameter_search.py`
2. **Test multiple seeds**: Ensure robustness (3-5 seeds minimum)
3. **Use long windows**: 96h or longer for reliable evaluation
4. **Prioritize improvement**: Focus on experiments showing negative trend
5. **Consider stability**: Balance mean imbalance vs. variance

### For Deployment

1. **Never disable exploration**: Keep epsilon_min >= 0.10
2. **Maintain high alpha**: Don't anneal below alpha_min
3. **Monitor performance**: Track imbalance metrics continuously
4. **Adapt if needed**: Re-run hyperparameter search if conditions change significantly

## Troubleshooting

### High imbalance persists
- Check if epsilon is decaying too fast
- Verify alpha is high enough for adaptation
- Ensure episode windows are long enough
- Run hyperparameter search to find better values

### Unstable learning
- Reduce alpha slightly (try 0.30-0.35)
- Increase gamma slightly (try 0.92-0.95)
- Ensure epsilon_min is not too low (<0.10)

### No improvement over time
- Verify epsilon starts at 1.0 (full exploration)
- Check that alpha is not too low
- Ensure reward functions are working correctly
- Try longer episode windows

## References

For more information, see:
- [configs/default.yaml](../configs/default.yaml) - Full configuration
- [scripts/hyperparameter_search.py](../scripts/hyperparameter_search.py) - Search implementation
- [core/simulation.py](../core/simulation.py) - Training loop and epsilon scheduler
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Project guidelines
