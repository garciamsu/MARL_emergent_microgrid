# CSV Format Standardization

## Overview

This document describes the centralized CSV format handling implemented across the MARL Microgrid application to ensure consistency and cross-platform compatibility.

## Problem Solved

**Before:** Mixed CSV formats across the application:
- Input datasets: European format (`;` separator, `,` decimal)
- Output files: Inconsistent formats
- Manual format handling in each module
- Cross-platform compatibility issues

**After:** Standardized format handling:
- Input datasets: European format (`;` separator, `,` decimal) - **READABLE**
- Output files: International format (`,` separator, `.` decimal) - **WRITABLE**
- Centralized handler in `core/csv_handler.py`
- Cross-platform support (Windows/Linux/macOS)

## Format Standards

### Input Datasets (Reading)
**Location:** `assets/datasets/*.csv`

**Format:**
```
Column separator: ; (semicolon)
Decimal separator: , (comma - European)
Encoding: UTF-8
Example: 196,25;0,5;85,5;286,185
```

**Usage:**
```python
from core.csv_handler import read_dataset_csv

df = read_dataset_csv('assets/datasets/microgrid_dataset.csv')
# Automatically uses: sep=';', decimal=',', encoding='utf-8'
```

### Output Files (Writing)
**Locations:**
- `results/evolution/*.csv`
- `results/metrics/*.csv`
- `results/logs/*.csv`
- `results/stability/*.csv`

**Format:**
```
Column separator: , (comma - standard CSV)
Decimal separator: . (dot - international)
Encoding: UTF-8
Float precision: 6 decimal places
Example: 196.25,0.5,85.5,286.185
```

**Usage:**
```python
from core.csv_handler import write_result_csv

write_result_csv(df, 'results/evolution/episode_0.csv')
# Automatically uses: sep=',', decimal='.', encoding='utf-8'
```

### Reading Output Files (Analysis)
**Usage:**
```python
from core.csv_handler import read_result_csv

df = read_result_csv('results/evolution/episode_0.csv')
# Automatically uses: sep=',', decimal='.', encoding='utf-8'
```

## Centralized Handler

### Location
`core/csv_handler.py`

### Key Functions

#### `read_dataset_csv(filepath, **kwargs)`
Read input datasets with European format.

**Parameters:**
- `filepath`: Path to CSV file
- `sep`: Column separator (default: `;`)
- `decimal`: Decimal separator (default: `,`)
- `encoding`: File encoding (default: `utf-8`)

**Returns:** pandas DataFrame

#### `write_result_csv(df, filepath, **kwargs)`
Write output files with international format.

**Parameters:**
- `df`: DataFrame to write
- `filepath`: Destination path
- `sep`: Column separator (default: `,`)
- `decimal`: Decimal separator (default: `.`)
- `encoding`: File encoding (default: `utf-8`)
- `index`: Write row indices (default: `False`)

**Creates parent directories automatically.**

#### `read_result_csv(filepath, **kwargs)`
Read output files with international format.

**Parameters:**
- `filepath`: Path to CSV file
- `sep`: Column separator (default: `,`)
- `decimal`: Decimal separator (default: `.`)
- `encoding`: File encoding (default: `utf-8`)

**Returns:** pandas DataFrame

#### `CSVConfig` class
Configuration constants for CSV format:
- `INPUT_SEP`, `INPUT_DECIMAL`, `INPUT_ENCODING`
- `OUTPUT_SEP`, `OUTPUT_DECIMAL`, `OUTPUT_ENCODING`
- `OUTPUT_FLOAT_FORMAT`

### System Information
```python
from core.csv_handler import CSVConfig

info = CSVConfig.get_system_info()
# Returns: platform, locale, decimal_point, thousands_sep
```

## Files Updated

### Core Modules
- ✅ `core/environment.py` - Dataset loading
- ✅ `core/simulation.py` - Episode evolution writing

### Analysis Tools
- ✅ `analysis/operative/A_data_check.py` - Dataset and episode validation
- ✅ `analysis/operative/C_collect_episodes.py` - Episode consolidation
- ✅ `analysis/operative/D_compute_metrics.py` - Metrics computation
- ✅ `analysis/operative/E_accumulated_reward.py` - Reward analysis
- ✅ `analysis/operative/E_graph_episode.py` - Episode visualization
- ✅ `analysis/common/utils.py` - Utility functions
- ✅ `analysis/stability/stability_analysis.py` - Stability metrics

### Scripts
- ✅ `scripts/validate_load_agent.py` - Load agent validation
- ✅ `scripts/test_scaling.py` - Power scaling test
- ✅ `scripts/test_csv_format.py` - **NEW** Format consistency test

## Cross-Platform Compatibility

### Windows
- Uses backslash `\` for paths internally, but `Path()` handles conversion
- Locale: varies (e.g., `es_VE`, `en_US`)
- Decimal point: `.` (system default)

### Linux/macOS
- Uses forward slash `/` for paths
- Locale: varies
- Decimal point: `.` (system default)

**Solution:** The `Path` class from `pathlib` and centralized CSV handler ensure cross-platform compatibility automatically.

## Migration Guide

### Old Code (Before Standardization)
```python
# Reading datasets
df = pd.read_csv('dataset.csv', sep=';', decimal=',')

# Writing results
df.to_csv('results.csv', index=False)

# Reading results
df = pd.read_csv('results.csv')
```

### New Code (After Standardization)
```python
from core.csv_handler import read_dataset_csv, write_result_csv, read_result_csv

# Reading datasets
df = read_dataset_csv('dataset.csv')

# Writing results
write_result_csv(df, 'results.csv')

# Reading results
df = read_result_csv('results.csv')
```

## Testing

Run the comprehensive format test:
```bash
python scripts/test_csv_format.py
```

**Tests:**
1. System information detection
2. Dataset reading (European format)
3. Result writing and reading (International format)
4. Cross-format compatibility with vanilla pandas

**Expected output:**
```
🎉 All tests PASSED! CSV format is consistent.
```

## Benefits

### Consistency
- Single source of truth for CSV format configuration
- No more manual format handling in each module
- Predictable behavior across all files

### Maintainability
- Easy to update format by changing `CSVConfig`
- Centralized documentation
- Clear separation between input and output formats

### Reliability
- Automatic directory creation
- Error handling built-in
- Cross-platform path handling

### Compatibility
- Output files readable by Excel, LibreOffice, Python, R, MATLAB
- Standard CSV format for international use
- UTF-8 encoding prevents character issues

## FAQ

### Why different formats for input and output?

**Input (European format):**
- Historical datasets may use European format
- Supports manual editing with European locale spreadsheet software
- Flexibility for data sources

**Output (International format):**
- Maximum compatibility with analysis tools
- Standard CSV format recognized worldwide
- No ambiguity with decimal separator

### Can I still use old format files?

Yes, for legacy files:
```python
# Specify format explicitly
df = read_dataset_csv('old_file.csv', sep=',', decimal='.')
```

For visualization tools like `E_graph_episode.py`, flexible format detection is still available.

### What about Excel files?

Excel files (`.xlsx`) are not affected by this standardization. They use their own internal format. The application uses `openpyxl` engine for Excel operations, which handles formatting automatically.

### How to check current system format?

```python
from core.csv_handler import CSVConfig

info = CSVConfig.get_system_info()
print(f"Platform: {info['platform']}")
print(f"Locale: {info['locale']}")
print(f"Decimal: {info['decimal_point']}")
```

## Troubleshooting

### Problem: "Could not convert string to float"

**Cause:** Reading file with wrong decimal separator

**Solution:**
```python
# For datasets with comma decimal
df = read_dataset_csv('file.csv')

# For results with dot decimal
df = read_result_csv('file.csv')

# Or specify explicitly
df = pd.read_csv('file.csv', sep=',', decimal='.')
```

### Problem: "Wrong number of columns"

**Cause:** Reading file with wrong column separator

**Solution:**
```python
# Check separator in file manually
# Use appropriate read function or specify separator
df = read_dataset_csv('file.csv', sep=';')  # For datasets
df = read_result_csv('file.csv', sep=',')   # For results
```

### Problem: "UnicodeDecodeError"

**Cause:** Wrong encoding

**Solution:**
```python
df = read_dataset_csv('file.csv', encoding='latin-1')
# or
df = read_dataset_csv('file.csv', encoding='cp1252')
```

## Complete Application Validation

To ensure the entire application uses consistent formats, run:

```bash
python scripts/validate_csv_consistency.py
```

This validates:
- ✅ All dataset files in `assets/datasets/` use European format
- ✅ All result files in `results/` use international format
- ✅ No conflicts with decimal separators
- ✅ All files are readable with standardized functions

**Files Updated for Consistency:**
- `core/environment.py` - Dataset loading
- `core/simulation.py` - Episode and reward file writing
- `analysis/operative/A_data_check.py` - Dataset validation
- `analysis/operative/C_collect_episodes.py` - Episode collection
- `analysis/operative/D_compute_metrics.py` - Metrics computation
- `analysis/operative/E_accumulated_reward.py` - Reward analysis
- `analysis/operative/E_graph_episode.py` - Episode plotting
- `analysis/common/utils.py` - Utility functions
- `analysis/stability/stability_analysis.py` - Stability analysis
- `scripts/hyperparameter_search.py` - Hyperparameter search
- `scripts/validate_load_agent.py` - Agent validation

All these files now use `read_dataset_csv()`, `read_result_csv()`, or `write_result_csv()` from `core/csv_handler.py`, ensuring complete consistency across the application.

## See Also

- [assets/datasets/README.md](../assets/datasets/README.md) - Dataset format requirements
- `core/csv_handler.py` - Source code
- `scripts/test_csv_format.py` - Unit test suite
- `scripts/validate_csv_consistency.py` - Full application validation
