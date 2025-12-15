# Stability Analysis for Distributed MARL Systems

This documentation describes the implementation of two stability analysis methods for the distributed multi-agent reinforcement learning (MARL) microgrid system.

## Overview

Two independent stability studies are implemented:

1. **Bellman Contraction Stability**: Measures convergence of value updates between consecutive episodes
2. **Consensus Stability**: Measures distributed consensus among agents

Both analyses provide empirical validation of theoretical convergence guarantees in distributed Q-learning systems.

---

## Theoretical Background

### Bellman Contraction Stability

**Metric:**
```
ΔV(k) = max_i || V_i(k+1) - V_i(k) ||_∞
```

Where:
- `V_i(k)` is the value function (flattened Q-table) of agent `i` at episode `k`
- `|| · ||_∞` is the infinity norm (maximum absolute element)
- `i` iterates over all agents

**Theoretical Foundation:**

The Bellman operator in Q-learning is a contraction mapping with contraction factor γ (discount factor). This means that successive applications of the Bellman update should bring value functions closer together. Formally:

```
|| T(V) - T(V') ||_∞ ≤ γ || V - V' ||_∞
```

where `T` is the Bellman operator and `γ < 1`.

**Interpretation:**

- **ΔV(k) → 0**: Indicates stable learning dynamics and convergence
  - Value functions are stabilizing
  - Learning has converged to a fixed point
  - System exhibits contraction property

- **ΔV(k) oscillates**: Indicates instability
  - Learning rate may be too high
  - Environment may be non-stationary
  - Agents may be interfering with each other

- **ΔV(k) diverges**: Indicates non-convergence
  - System parameters (α, γ) may be inappropriate
  - Exploration-exploitation balance may be off
  - Fundamental convergence conditions may not be met

---

### Consensus Stability

**Metrics:**
```
V_avg(k) = (1/N) * Σ_i V_i(k)
D(k) = (1/N) * Σ_i || V_i(k) - V_avg(k) ||_2
```

Where:
- `N` is the number of agents
- `|| · ||_2` is the Euclidean norm
- `V_avg(k)` is the mean value function across all agents

**Theoretical Foundation:**

In distributed systems, consensus requires that individual agents converge toward a common representation. This is critical for:
- Coordinated decision-making
- Emergent system-level behavior
- Distributed stability

The consensus metric measures how closely agents' value representations align with the collective average.

**Interpretation:**

- **D(k) → 0**: Consensus achieved
  - Agents have converged to similar value representations
  - Distributed coordination is successful
  - System exhibits emergent collective behavior

- **D(k) > 0 (persistent)**: Lack of coordination
  - Agents maintain distinct value representations
  - May indicate different local optima
  - Could reflect agent heterogeneity or information asymmetry

- **D(k) diverges**: System instability
  - Agents are diverging rather than converging
  - Learning dynamics are unstable
  - May indicate conflicting objectives or rewards

---

## Implementation

### Module Structure

```
analysis_tools/
├── stability_analysis.py              # Core analyzer classes
├── collect_qtables_per_episode.py     # Data collection script
└── run_stability_analysis.py          # Standalone runner
```

### Classes

#### `BellmanContractionStabilityAnalyzer`

Analyzes stability via distributed Bellman contraction.

**Key Methods:**
- `add_episode_qtables(episode, agents)`: Add Q-table snapshot for an episode
- `flatten_qtable(q_table)`: Convert nested dict Q-table to 1D array
- `compute_stability_metrics()`: Compute ΔV(k) for all episodes
- `save_results(filename)`: Save metrics to CSV
- `plot_stability(filename)`: Generate stability plot
- `run_analysis()`: Execute complete analysis pipeline

#### `ConsensusStabilityAnalyzer`

Analyzes stability via distributed consensus among agents.

**Key Methods:**
- `add_episode_qtables(episode, agents)`: Add Q-table snapshot for an episode
- `flatten_qtable(q_table)`: Convert nested dict Q-table to 1D array
- `compute_consensus_metrics()`: Compute D(k) for all episodes
- `save_results(filename)`: Save metrics to CSV
- `plot_stability(filename)`: Generate consensus plot
- `run_analysis()`: Execute complete analysis pipeline

---

## Usage

### Step 1: Collect Q-table History

Run a training session and collect Q-table snapshots per episode:

```bash
python analysis_tools/collect_qtables_per_episode.py
```

This script:
- Loads configuration from `configs/default.yaml`
- Runs training for the specified number of episodes
- Collects Q-table snapshots after each episode
- Saves history to `results/stability/qtables_per_episode.npz`

**Configuration:**

The script uses standard configuration from `configs/default.yaml`:
- `simulation.episodes`: Number of training episodes
- `simulation.seed`: Random seed for reproducibility
- `simulation.dataset`: Training dataset
- All agent and environment settings

### Step 2: Run Stability Analysis

Execute both stability analyses on collected data:

```bash
python analysis_tools/run_stability_analysis.py
```

This script:
- Loads Q-table history from `results/stability/qtables_per_episode.npz`
- Runs Bellman contraction analysis
- Runs consensus analysis
- Generates CSV files and plots

---

## Output Files

### CSV Files

#### `bellman_contraction_stability.csv`

Contains episode-wise ΔV metrics:

| Column | Description |
|--------|-------------|
| `episode` | Episode index (0-based) |
| `delta_v` | Maximum infinity norm across agents |

#### `consensus_stability.csv`

Contains episode-wise consensus metrics:

| Column | Description |
|--------|-------------|
| `episode` | Episode index (0-based) |
| `consensus_deviation` | Average deviation from mean |

### Plots

#### `bellman_contraction_stability.png`

Line plot showing ΔV(k) evolution over episodes:
- X-axis: Episode index
- Y-axis: ΔV(k) (logarithmic scale)
- Includes interpretation guide

#### `consensus_stability.png`

Line plot showing D(k) evolution over episodes:
- X-axis: Episode index
- Y-axis: D(k) (logarithmic scale)
- Includes interpretation guide

---

## Interpreting Results

### Example: Stable Convergence

```
Final ΔV:    0.000234
Max ΔV:      15.234567
Min ΔV:      0.000234
Status: ✅ CONVERGED (ΔV < 0.01)

Final D:     0.045678
Max D:       8.234567
Min D:       0.045678
Status: ✅ CONSENSUS ACHIEVED (D < 0.1)
```

**Interpretation:**
- Both metrics approach zero
- Learning has converged
- Agents have reached consensus
- System is stable

### Example: Near Convergence

```
Final ΔV:    0.034567
Max ΔV:      12.345678
Min ΔV:      0.012345
Status: ⚠️  NEAR CONVERGENCE (ΔV < 0.1)

Final D:     0.234567
Max D:       5.678901
Min D:       0.123456
Status: ⚠️  PARTIAL CONSENSUS (D < 1.0)
```

**Interpretation:**
- Metrics are decreasing but not fully converged
- May need more training episodes
- Consider adjusting learning rate or exploration schedule

### Example: Instability

```
Final ΔV:    2.345678
Max ΔV:      15.678901
Min ΔV:      0.234567
Status: ❌ NOT CONVERGED (ΔV ≥ 0.1)

Final D:     3.456789
Max D:       8.901234
Min D:       1.234567
Status: ❌ NO CONSENSUS (D ≥ 1.0)
```

**Interpretation:**
- High final values indicate instability
- Learning dynamics are not converging
- Possible causes:
  - Learning rate too high
  - Insufficient exploration
  - Conflicting agent objectives
  - Non-stationary environment

---

## Technical Details

### Handling Q-table Heterogeneity

Agents may discover different state-action spaces over time, leading to Q-tables of varying sizes. The implementation handles this by:

1. **Padding**: When comparing Q-tables of different sizes, shorter arrays are zero-padded to match the longer one
2. **Sorted Keys**: States and actions are sorted consistently to ensure comparable orderings
3. **Common Dimension**: For consensus analysis, all agent vectors are padded to a common maximum dimension

### Computational Complexity

For `N` agents, `E` episodes, and average Q-table size `S`:

- **Bellman Contraction**: O(E × N × S)
- **Consensus**: O(E × N × S)

Both analyses scale linearly with the number of episodes and agents.

### Memory Requirements

Q-table history is stored in compressed `.npz` format using:
- Object arrays for variable-length state tuples
- Int32 arrays for actions
- Float64 arrays for Q-values

Typical size: ~10-50 MB per 1000 episodes (depends on state-action space).

---

## Integration with Existing Workflow

### Compatibility

- **Does NOT modify**: Core training loop, agents, rewards, environment
- **Does NOT change**: Hyperparameters, learning behavior
- **Self-contained**: Separate scripts that don't affect existing code
- **Independent**: Can be run after training or in parallel workflows

### Relationship to Other Analysis Tools

This stability analysis complements existing tools in `analysis_tools/`:

- `B_run_training.py`: Standard training (can use this instead)
- `D_compute_metrics.py`: Episode-level metrics (complements stability)
- `E_accumulated_reward.py`: Reward analysis (different perspective)

---

## Advanced Usage

### Programmatic Usage

```python
from analysis_tools.stability_analysis import (
    BellmanContractionStabilityAnalyzer,
    ConsensusStabilityAnalyzer
)
from analysis_tools.collect_qtables_per_episode import load_qtables_history

# Load data
qtables = load_qtables_history("results/stability/qtables_per_episode.npz")

# Run Bellman analysis
bellman = BellmanContractionStabilityAnalyzer("results/stability")
bellman.q_tables_per_episode = qtables
bellman.run_analysis()

# Run Consensus analysis
consensus = ConsensusStabilityAnalyzer("results/stability")
consensus.q_tables_per_episode = qtables
consensus.run_analysis()
```

### Custom Analysis

You can extend the analyzers for custom metrics:

```python
class CustomStabilityAnalyzer(BellmanContractionStabilityAnalyzer):
    def compute_custom_metric(self):
        """Add your custom stability metric here."""
        # Access self.q_tables_per_episode
        # Compute custom analysis
        pass
```

---

## Troubleshooting

### Issue: "Q-table history not found"

**Solution:** Run data collection first:
```bash
python analysis_tools/collect_qtables_per_episode.py
```

### Issue: "No Q-table data found"

**Possible causes:**
- Agents not initialized properly
- Q-tables empty (no learning occurred)
- Configuration error

**Solution:** Check agent initialization and training configuration.

### Issue: Large memory usage

**Possible causes:**
- Very large state-action spaces
- Many episodes
- Many agents

**Solutions:**
- Reduce number of episodes collected
- Use sampling (collect every N episodes)
- Increase compression level

### Issue: Plots not displaying convergence

**Possible causes:**
- Insufficient training episodes
- Learning rate too high/low
- Exploration schedule inappropriate

**Solutions:**
- Increase training episodes
- Adjust `alpha` (learning rate)
- Modify epsilon schedule

---

## References

### Theoretical Foundations

1. **Bellman Contraction Mapping**:
   - Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic Programming*. Athena Scientific.

2. **Distributed Q-learning**:
   - Lauer, M., & Riedmiller, M. (2000). *An Algorithm for Distributed Reinforcement Learning in Cooperative Multi-Agent Systems*. ICML.

3. **Consensus in Multi-Agent Systems**:
   - Olfati-Saber, R., Fax, J. A., & Murray, R. M. (2007). *Consensus and Cooperation in Networked Multi-Agent Systems*. Proceedings of the IEEE.

4. **Emergent Coordination**:
   - Busoniu, L., Babuska, R., & De Schutter, B. (2008). *A Comprehensive Survey of Multiagent Reinforcement Learning*. IEEE Transactions on Systems, Man, and Cybernetics.

---

## Future Extensions

Potential enhancements to this analysis framework:

1. **Per-agent stability metrics**: Track convergence of individual agents
2. **State-space coverage**: Analyze exploration vs. exploitation dynamics
3. **Temporal stability windows**: Sliding window analysis for non-stationary detection
4. **Cross-correlation analysis**: Measure agent interaction effects
5. **Policy stability**: Analyze policy changes in addition to value functions

---

## Contact and Support

For questions or issues related to stability analysis:
- Check existing documentation in `docs/`
- Review configuration in `configs/default.yaml`
- Ensure all prerequisites are installed (see `requirements.txt`)
