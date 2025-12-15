# Datasets Directory

## Data Source
https://data.mendeley.com/datasets/gxc6j5btrx/1

## Dataset Format Requirements

### Power Units
All datasets in this directory must contain power values in **kilowatts (kW)**, NOT in watts (W).

The system automatically scales all power columns from kW to W using the `power_scale_factor` parameter defined in `configs/default.yaml` (default: 1000).

### Columns That Are Automatically Scaled
- `demand` (kW → W)
- `solar_potential` (kW → W) 
- `wind_potential` (kW → W)
- Any column containing "potential" in its name

### Columns NOT Scaled
- `price` (EUR/MWh)
- `Datetime` / `datetime`

### CSV Format Requirements
- **Separators**: Use either comma (`,`) or semicolon (`;`) as column separator
- **Decimal separator**: Use dot (`.`) for decimal numbers, NOT comma
- **Encoding**: UTF-8 preferred
- **Example row**: `2025-01-01 00:00:00,250.0,1.0,0.0,0.0`

### ⚠️ IMPORTANT: Do NOT Create Pre-Scaled Datasets
**DO NOT** manually create datasets with values already in watts (W). The code will apply the scaling factor automatically, causing a double-scaling error if values are already in W.

**Incorrect**: `demand=250000` (already in W)  
**Correct**: `demand=250.0` (in kW, will be scaled to 250000 W automatically)

### Example of Correct Values
```csv
Datetime,demand,price,solar_potential,wind_potential
2025-01-01 00:00:00,250.0,1.0,0.0,0.0
2025-01-01 07:00:00,250.0,1.0,155.29,0.0
2025-01-01 12:00:00,250.0,1.0,600.0,0.0
```

These values will be automatically scaled to:
- demand: 250,000 W
- solar_potential: 155,290 W (at 7am), 600,000 W (at noon)

## Available Datasets

### Training Datasets
- `microgrid_dataset_10d_hourly_demand250.csv`: 10-day hourly dataset with constant 250 kW demand
- Other training datasets as needed

### Testing/Validation Datasets
- Add test datasets following the same format requirements
