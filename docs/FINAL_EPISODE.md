# Final Episode Evaluation Feature

## Overview

This feature implements a **final episode evaluation** mechanism where the last training episode uses a predefined configuration instead of random parameters. This allows for consistent evaluation of the trained agents on a specific scenario.

## Configuration

The final episode is configured in `configs/default.yaml` under the `final_episode` section:

```yaml
final_episode:
  # Predefined time window (indices in the dataset)
  # The last training episode will use this specific window
  # Window size can be any length (not restricted to 24 hours)
  start_index: 0
  end_index: 24
  
  # Fixed initial SOC for battery in final episode
  initial_soc: 0.5
```

### Parameters

- **`start_index`**: Starting index in the dataset for the evaluation window (default: 0)
- **`end_index`**: Ending index in the dataset for the evaluation window (default: 24)
  - Can be any size (not restricted to 24 hours)
  - Must be greater than `start_index`
- **`initial_soc`**: Fixed initial State of Charge for the battery (default: 0.5)
  - Must be between 0.0 and 1.0

## Behavior

### Training Episodes (Episodes 0 to N-2)

- **Window Selection**: Random 24-hour contiguous window from dataset
- **Initial SOC**: Random value between `initial_soc_min` and `initial_soc_max`
- **Exploration**: Epsilon-greedy policy with scheduled epsilon
- **Learning**: Q-tables are updated after each step

### Final Episode (Episode N-1)

- **Window Selection**: Uses predefined `start_index` and `end_index` from config
- **Initial SOC**: Uses fixed `initial_soc` from config
- **Exploration**: Pure exploitation (epsilon = 0.0)
- **Learning**: Q-tables are **NOT** updated (evaluation only)

## Implementation Details

### Modified Files

1. **`configs/default.yaml`**
   - Added `final_episode` configuration section

2. **`core/simulation.py`**
   - Added detection of final episode: `is_final_episode = (episode == num_episodes - 1)`
   - Conditional window selection based on episode type
   - Conditional SOC initialization based on episode type
   - Conditional epsilon setting (0.0 for final episode)
   - Conditional Q-table updates (skipped for final episode)
   - Added `is_final_episode` flag to episode metadata

### Episode Metadata

The episode metadata Excel file (`results/logs/episode_metadata.xlsx`) now includes an additional column:

- **`is_final_episode`**: Boolean flag indicating if the episode was the final evaluation episode

## Usage Example

To configure a specific evaluation scenario, edit `configs/default.yaml`:

```yaml
final_episode:
  # Evaluate on a 72-hour window (e.g., hours 3000-3072 in annual dataset)
  start_index: 3000
  end_index: 3072
  
  # Start with battery at 80% charge
  initial_soc: 0.8
```

## Testing

A test script is provided to verify the functionality:

```bash
python test_final_episode.py
```

This script:
1. Runs a short training session (5 episodes)
2. Verifies that the final episode uses the configured window and SOC
3. Checks that epsilon is 0.0 in the final episode
4. Displays a summary of all episodes

## Benefits

1. **Consistent Evaluation**: Always evaluate on the same scenario for reproducibility
2. **Performance Benchmarking**: Compare different training configurations on the same test case
3. **Domain-Specific Testing**: Test on critical scenarios (e.g., peak demand hours, low renewable generation)
4. **Pure Exploitation**: See how well agents perform without exploration noise
5. **No Learning Interference**: Q-tables remain unchanged during evaluation

## Notes

- This feature is **always active** and cannot be disabled
- The final episode is counted as part of the total episode count specified in `simulation.episodes`
- If you want only training (no evaluation), you can ignore the final episode results
- The window validation ensures the indices are valid but does **not** restrict the window size
- The episode will run for exactly `end_index - start_index` steps (hours)
