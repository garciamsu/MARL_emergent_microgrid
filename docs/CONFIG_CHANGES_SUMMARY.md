# Configuration Changes Summary

## Before → After

### Learning Rate (Alpha)
```yaml
# BEFORE
alpha: 0.3

# AFTER
alpha: 0.35        # High constant learning rate for online adaptation (range: 0.25-0.45)
alpha_min: 0.15    # Minimum alpha if scheduled (not annealed to zero)
```

**Rationale**: Higher learning rate (0.35) enables faster adaptation to non-stationary conditions. alpha_min ensures learning never anneals to zero.

---

### Discount Factor (Gamma)
```yaml
# BEFORE
gamma: 0.85

# AFTER
gamma: 0.90        # Mid-high discount for non-stationary env (range: 0.88-0.95)
```

**Rationale**: Higher discount (0.90) balances immediate vs. future rewards without over-trusting distant predictions in changing environments.

---

### Exploration Schedule (Epsilon)
```yaml
# BEFORE
epsilon:
  schedule: constant
  start: 0.3
  end: 0.3
  decay: null
  values: []

# AFTER
epsilon:
  schedule: exponential    # Exponential decay (very slow)
  start: 1.0               # Full exploration initially
  end: 0.15                # Target minimum (10-20% range)
  min: 0.15                # Enforce minimum (never below this)
  decay: 0.9985            # Very slow decay factor
  values: []
```

**Rationale**: 
- Starts at 100% exploration (ε=1.0) for thorough initial learning
- Decays very slowly to maintain high exploration
- Never drops below 15% to ensure continuous adaptation
- Previous config had constant 30% exploration (no initial learning phase)

---

### Episode Window
```yaml
# DEFAULT
episode_window_hours: 96    # 4 days for meaningful online learning evaluation
```

**Rationale**: Longer episodes (96h+) provide sufficient data for evaluating imbalance reduction in online learning.

---

## Key Principles

1. **No Offline Pretraining**: System learns from scratch during operation
2. **High & Constant Alpha**: Maintains reactivity (0.35, never below 0.15)
3. **Persistent Exploration**: epsilon ≥ 0.15 always, never disabled
4. **Moderate Gamma**: Balances future consideration (0.90) without over-trusting
5. **Long Episodes**: 96h+ windows for meaningful evaluation

---

## Validation

Check configuration is working:

```bash
# 1. Run self-check
python scripts/self_check.py

# 2. Run short training
python main.py  # (with simulation.episodes: 50)

# 3. Check logs
# - Verify epsilon starts at 1.0 in results/logs/
# - Check for decreasing imbalance trend

# 4. Run hyperparameter search (optional)
python scripts/hyperparameter_search.py
```

---

## Files Modified

1. ✅ [configs/default.yaml](../configs/default.yaml)
   - Updated all agents: alpha=0.35, gamma=0.90
   - Added alpha_min=0.15
   - Reconfigured epsilon schedule

2. ✅ [docs/ONLINE_QLEARNING_CONFIG.md](../docs/ONLINE_QLEARNING_CONFIG.md)
   - Comprehensive configuration guide
   - Parameter rationales
   - Troubleshooting tips

3. ✅ [scripts/hyperparameter_search.py](../scripts/hyperparameter_search.py)
   - Systematic hyperparameter search
   - Single-episode evaluation
   - Imbalance reduction metrics

4. ✅ [scripts/README_hyperparameter_search.md](../scripts/README_hyperparameter_search.md)
   - Quick usage guide
   - Search space description
   - Output interpretation

5. ✅ [.github/copilot-instructions.md](../.github/copilot-instructions.md)
   - Updated configuration conventions
   - Added hyperparameter search to workflow
   - Added reference to online learning config

---

## Next Steps

1. **Validate**: Run `python scripts/self_check.py`
2. **Test**: Run short training (50 episodes)
3. **Tune**: Execute `python scripts/hyperparameter_search.py` (optional)
4. **Deploy**: Use best-performing parameters from search

---

## See Also

- [ONLINE_QLEARNING_CONFIG.md](../docs/ONLINE_QLEARNING_CONFIG.md) - Detailed guide
- [README_hyperparameter_search.md](../scripts/README_hyperparameter_search.md) - Search tool usage
- [configs/default.yaml](../configs/default.yaml) - Full configuration
