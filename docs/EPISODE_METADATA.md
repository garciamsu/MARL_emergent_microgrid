# Episode Metadata Documentation

## Overview

During training, the system generates a metadata file that tracks the exact data window and initial battery State of Charge (SOC) used for each episode, along with frequency analysis of window usage.

## File Location

```
results/logs/episode_metadata.xlsx
```

## Purpose

This file allows full reproducibility of training episodes and provides insights into data usage patterns by recording:
- Which 24-hour window from the dataset was used
- The initial SOC assigned to the battery agent
- Frequency analysis showing which windows were reused during training

## File Structure

The Excel file contains **two sheets**:

### Sheet 1: Episode Details

### Sheet 1: Episode Details

Records detailed information for each episode:

| Column | Type | Description |
|--------|------|-------------|
| `episode` | int | Episode number (0-indexed) |
| `window_start_index` | int | Starting index in the full dataset for the 24-hour window |
| `window_end_index` | int | Ending index in the full dataset (exclusive) |
| `initial_soc` | float | Initial State of Charge for battery agent (0.0 to 1.0) |

**Example:**
```
episode  window_start_index  window_end_index  initial_soc
0        7270                7294              0.737234
1        5390                5414              0.685595
2        5734                5758              0.456666
```

### Sheet 2: Window Frequency

Analyzes how often each 24-hour window was used during training, sorted by frequency (most used first):

| Column | Type | Description |
|--------|------|-------------|
| `window_start_index` | int/str | Starting index of the window (or "SUMMARY" for stats row) |
| `window_end_index` | int | Ending index of the window |
| `frequency` | int/str | Number of times this window was used (or summary text) |
| `episodes` | str | Comma-separated list of episode numbers that used this window (or summary stats) |

**Example:**
```
window_start_index  window_end_index  frequency  episodes
7392                7416              2          65, 80
262                 286               1          99
189                 213               1          13
SUMMARY                               Total Episodes: 100  Unique Windows: 99, Max Freq: 2, ...
```

The last row contains summary statistics:
- Total Episodes: Total number of training episodes
- Unique Windows: Number of distinct 24-hour windows used
- Max Freq: Maximum times any window was reused
- Min Freq: Minimum frequency (typically 1)
- Avg Freq: Average usage frequency across all unique windows

## Usage

### Reading Episode Details

```python
import pandas as pd

# Load episode details
episode_details = pd.read_excel('results/logs/episode_metadata.xlsx', 
                                sheet_name='Episode Details')

# Get window for specific episode
episode_5 = episode_details[episode_details['episode'] == 5].iloc[0]
start_idx = episode_5['window_start_index']
end_idx = episode_5['window_end_index']
initial_soc = episode_5['initial_soc']

print(f"Episode 5 used data from index {start_idx} to {end_idx}")
print(f"Initial SOC: {initial_soc:.4f}")
```

### Analyzing Window Frequency

```python
import pandas as pd

# Load frequency analysis
window_freq = pd.read_excel('results/logs/episode_metadata.xlsx',
                            sheet_name='Window Frequency')

# Filter out summary row
freq_data = window_freq[window_freq['window_start_index'] != 'SUMMARY']

# Find most frequently used windows
most_used = freq_data[freq_data['frequency'] > 1]
print(f"\nWindows used more than once: {len(most_used)}")
print(most_used)

# Get summary statistics
summary = window_freq[window_freq['window_start_index'] == 'SUMMARY'].iloc[0]
print(f"\n{summary['frequency']}")
print(f"{summary['episodes']}")
```

### Reproducing an Episode

```python
import pandas as pd
import numpy as np
from configs.loader import load_config
from core.environment import MultiAgentEnv

# Load configuration and metadata
config = load_config()
metadata = pd.read_excel('results/logs/episode_metadata.xlsx')

# Get episode info
episode_info = metadata[metadata['episode'] == 10].iloc[0]

# Create environment
env = MultiAgentEnv(config)

# Extract the exact window used in training
start = episode_info['window_start_index']
end = episode_info['window_end_index']
episode_data = env.full_dataset.iloc[start:end]

# Reset with exact same conditions
initial_soc = episode_info['initial_soc']
env.reset(episode_data, initial_soc)

# Now the environment is in the exact same state as episode 10
```

## Configuration

The initial SOC range is configured in `configs/default.yaml`:

```yaml
agents:
  battery:
    limits:
      initial_soc_min: 0.1
      initial_soc_max: 0.9
```

## Implementation

The metadata is generated in `core/simulation.py` during the `run_training()` function:
- Each episode randomly selects a 24-hour contiguous window
- A random initial SOC is generated within configured limits
- Both values are recorded in the metadata list
- After all episodes, the system:
  1. Creates a DataFrame with episode details
  2. Performs groupby analysis to count window frequency
  3. Generates summary statistics
  4. Saves both sheets to Excel using `openpyxl` ExcelWriter

## Insights from Frequency Analysis

The frequency analysis helps understand:
- **Data diversity**: How many unique scenarios the agents experienced
- **Sampling efficiency**: Whether the random sampling effectively covers the dataset
- **Overfitting risk**: High-frequency windows might lead to overfitting on specific scenarios
- **Dataset size impact**: Larger datasets naturally result in fewer repetitions

For example, with 100 episodes and a dataset of 8760 hours:
- Expected unique windows ≈ 99-100 (very low collision probability)
- With 1000 episodes: More repetitions start appearing
- With 5000 episodes: Significant overlap, some windows used 5-10 times

## Notes

- Window indices refer to the full dataset loaded from `assets/datasets/`
- Initial SOC values are clipped to battery limits (soc_min, soc_max)
- The file is overwritten on each training run
- Requires `openpyxl` library for Excel generation (included in requirements.txt)
