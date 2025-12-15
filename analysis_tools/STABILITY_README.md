# Stability Analysis - Quick Navigation

## 🎯 What is this?

Two stability analysis tools for distributed MARL systems:
1. **Bellman Contraction**: Measures value function convergence
2. **Consensus**: Measures agent coordination

## 🚀 Quick Start

```bash
# Step 1: Collect Q-table snapshots during training
python analysis_tools/collect_qtables_per_episode.py

# Step 2: Run stability analysis
python analysis_tools/run_stability_analysis.py
```

## 📂 Files

| File | Purpose |
|------|---------|
| `stability_analysis.py` | Core analyzer classes |
| `collect_qtables_per_episode.py` | Data collection |
| `run_stability_analysis.py` | Analysis runner |
| `test_stability_analysis.py` | Validation suite |

## 📊 Outputs

Located in `results/stability/`:
- `bellman_contraction_stability.csv` & `.png`
- `consensus_stability.csv` & `.png`

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [STABILITY_ANALYSIS.md](../docs/STABILITY_ANALYSIS.md) | Comprehensive guide (theory + usage) |
| [STABILITY_ANALYSIS_QUICK_REF.md](../docs/STABILITY_ANALYSIS_QUICK_REF.md) | Quick reference (commands + thresholds) |
| [STABILITY_IMPLEMENTATION_SUMMARY.md](../docs/STABILITY_IMPLEMENTATION_SUMMARY.md) | Implementation details |

## 🔬 Metrics

### ΔV(k) - Bellman Contraction
```
ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞
```
- ✅ `< 0.01`: Converged
- ⚠️ `< 0.1`: Near convergence
- ❌ `≥ 0.1`: Not converged

### D(k) - Consensus
```
D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2
```
- ✅ `< 0.1`: Consensus achieved
- ⚠️ `< 1.0`: Partial consensus
- ❌ `≥ 1.0`: No consensus

## ✅ Validation

Run the test suite:
```bash
python analysis_tools/test_stability_analysis.py
```

Expected: All tests pass (validated on synthetic data)

## 🔧 Configuration

Uses settings from `configs/default.yaml`:
- `simulation.episodes`: Training episodes
- `simulation.seed`: Random seed
- `agents.<type>.policy`: Learning parameters

## 💡 Interpretation

| Scenario | ΔV | D | Meaning |
|----------|----|----|---------|
| Stable | → 0 | → 0 | ✅ Converged & coordinated |
| Unstable | > 0.1 | > 1.0 | ❌ Not converging |
| Partial | → 0 | > 0 | ⚠️ Converged but no consensus |

## 🎓 Theory

Based on:
- Bellman operator contraction mapping
- Distributed consensus theory
- Emergent coordination in MARL

## 📧 Questions?

See comprehensive documentation in `docs/STABILITY_ANALYSIS.md`

---

**Note:** These tools do NOT modify existing training code. They are self-contained analysis scripts that operate on Q-table snapshots.
