# Hyperparameter Search for Online Q-Learning

This script performs systematic search for optimal Q-learning hyperparameters under strictly online learning conditions (no offline pretraining).

## Purpose

Find suitable combinations of:
- **alpha** (learning rate): 0.25-0.45
- **gamma** (discount factor): 0.88-0.95
- **epsilon_min** (minimum exploration): 0.10-0.20

That achieve measurable reduction in energy imbalance over single-episode runs with long windows (96h, 120h, 144h+).

## Key Features

- **Black-box wrapper**: Does NOT modify existing training logic, agents, environment, or rewards
- **Configuration injection**: Parameters set only through config interface
- **Single-episode experiments**: Fast evaluation per configuration
- **Long episode windows**: Realistic online learning evaluation
- **Imbalance metrics**: Tracks energy balance improvement over time

## Usage

```bash
# Run full search (default: 600 experiments)
python scripts/hyperparameter_search.py
```

## Search Space

Default configuration:
- **alpha**: [0.25, 0.30, 0.35, 0.40, 0.45]
- **gamma**: [0.88, 0.90, 0.92, 0.95]
- **epsilon_min**: [0.10, 0.12, 0.15, 0.18, 0.20]
- **episode_windows**: [96, 120] hours
- **seeds**: [42, 43, 44]

Total: 5 × 4 × 5 × 2 × 3 = **600 experiments**

## Output

Results saved to: `results/experiments/hyperparameter_search/run_YYYYMMDD_HHMMSS/`

Files:
- `results_final.csv` - All experiment results
- `results_interim.csv` - Intermediate saves (every 10 experiments)
- `analysis_report.txt` - Summary with recommendations
- `exp_####_evolution.csv` - Episode data for each experiment

## Evaluation Metrics

Each experiment evaluated on:
- **Mean absolute imbalance**: Avg |Δ_ph| over episode
- **Max absolute imbalance**: Peak |Δ_ph|
- **Imbalance std**: Stability measure
- **Imbalance trend**: Slope (negative = improving)
- **Improvement ratio**: (initial - final) / initial

## Interpreting Results

The script identifies top configurations based on:
1. **Improvement ratio** (higher better): % imbalance reduction
2. **Mean imbalance** (lower better): Overall balance quality
3. **Negative trend** (required): Confirms learning occurs

Example output:
```
🏆 Top 10 Parameter Combinations (by imbalance reduction):

Rank #1:
   α=0.350, γ=0.900, ε_min=0.150
   Window: 96h, Seed: 42
   Mean |Δ|: 12543.21 W
   Trend:    -23.45 W/step ✅ (improving)
   Improve:  18.45%
```

## Customization

Edit `main()` function in script to modify:
- `episode_windows`: List of window sizes to test
- `seeds`: Random seeds for robustness
- `max_experiments`: Limit number of experiments (None = all)

## Quick Test

For faster testing (30 experiments instead of 600):

```python
# In hyperparameter_search.py, modify main():
df_results = searcher.run_search(
    episode_windows=[96],      # Single window
    seeds=[42],                # Single seed
    max_experiments=30         # Limit experiments
)
```

## Requirements

- Same environment as main training
- Sufficient disk space for evolution CSVs (≈1-2 MB per experiment)
- Execution time: ≈1-2 minutes per experiment (10-20 hours for full search)

## See Also

- [ONLINE_QLEARNING_CONFIG.md](../docs/ONLINE_QLEARNING_CONFIG.md) - Full configuration guide
- [configs/default.yaml](../configs/default.yaml) - Current configuration
- [copilot-instructions.md](../.github/copilot-instructions.md) - Project guidelines
