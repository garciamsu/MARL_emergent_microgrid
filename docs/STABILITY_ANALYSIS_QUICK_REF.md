# Stability Analysis Quick Reference

## Quick Start

```bash
# Step 1: Collect Q-table history
python analysis/collect_qtables_per_episode.py

# Step 2: Run stability analysis
python analysis/run_stability_analysis.py
```

---

## Output Files

All outputs saved to `results/stability/`:

| File | Description |
|------|-------------|
| `qtables_per_episode.npz` | Q-table history (Step 1) |
| `bellman_contraction_stability.csv` | ΔV metrics |
| `bellman_contraction_stability.png` | ΔV plot |
| `consensus_stability.csv` | D(k) metrics |
| `consensus_stability.png` | D(k) plot |

---

## Metrics

### Bellman Contraction: ΔV(k)

```
ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞
```

**Thresholds:**
- ✅ `< 0.01` : Converged
- ⚠️ `< 0.1`  : Near convergence
- ❌ `≥ 0.1`  : Not converged

### Consensus: D(k)

```
D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2
```

**Thresholds:**
- ✅ `< 0.1` : Consensus achieved
- ⚠️ `< 1.0` : Partial consensus
- ❌ `≥ 1.0` : No consensus

---

## Interpretation Guide

### Stable System

```
ΔV → 0  and  D → 0
```
✅ Learning converged, agents coordinated

### Unstable System

```
ΔV > 0.1  or  D > 1.0
```
❌ Adjust learning rate, epsilon schedule, or training duration

### Partial Convergence

```
ΔV declining  but  D persistent
```
⚠️ Value functions converging but agents not coordinated

---

## Configuration

Edit `configs/default.yaml`:

```yaml
simulation:
  episodes: 150          # More episodes → better convergence
  seed: 42              # For reproducibility
  
agents:
  <type>.policy:
    alpha: 0.1          # Learning rate (lower → slower but stable)
    gamma: 0.9          # Discount factor
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Q-table history not found" | Run Step 1 first |
| High final ΔV | Increase episodes or decrease alpha |
| High final D | Check reward functions for conflicts |
| Memory error | Reduce episodes or state space |

---

## Programmatic Usage

```python
from analysis.stability.stability_analysis import run_both_stability_analyses
from analysis.collect_qtables_per_episode import load_qtables_history

# Load data
qtables = load_qtables_history("results/stability/qtables_per_episode.npz")

# Run analysis
run_both_stability_analyses(qtables, results_dir="results/stability")
```

---

## Theory Summary

| Metric | Measures | Indicates |
|--------|----------|-----------|
| ΔV(k) | Value function changes | Bellman contraction |
| D(k) | Agent coordination | Distributed consensus |

Both → 0 implies distributed stability and emergent coordination.

---

For detailed documentation, see [STABILITY_ANALYSIS.md](STABILITY_ANALYSIS.md)
