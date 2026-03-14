"""
Test script: Apply maximum weather degradation and save for inspection.

Usage:
  python forecasting/testing/test_max_degradation.py --city seoul
  python forecasting/testing/test_max_degradation.py --city london
  python forecasting/testing/test_max_degradation.py --city washington
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from weather.weather_degradation import prepare_degradation_parameters, degrade_weather_dataset

parser = argparse.ArgumentParser(description="Apply maximum degradation and save for inspection.")
parser.add_argument(
    "--city",
    choices=["seoul", "washington", "london"],
    required=True,
    help="City dataset to use.",
)
args = parser.parse_args()

from config_seoul import get_config as seoul_config
from config_washington import get_config as washington_config
from config_london import get_config as london_config

city_configs = {
    "seoul":      seoul_config,
    "washington": washington_config,
    "london":     london_config,
}
config = city_configs[args.city]()
COLUMN_MAPPING = config.weather_degradation_mapping

# Load data
data_path = Path("data") / config.data_filename
print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows")

# Prepare degradation parameters (solar cap from training data)
print("\nComputing degradation parameters...")
params = prepare_degradation_parameters(df, COLUMN_MAPPING)
print(f"Solar cap: {params['solar_cap']:.2f} MJ/m²")

# Apply worst-case degradation (168h horizon)
print("\nApplying 168h degradation (worst case)...")
df_degraded = degrade_weather_dataset(
    df=df,
    horizon_hours=168,
    degradation_params=params,
    column_mapping=COLUMN_MAPPING,
    seed=config.degradation_seed
)

# Save degraded dataset alongside the original, with a descriptive suffix
stem = Path(config.data_filename).stem
output_file = Path("data") / f"{stem}_degraded_168h.csv"
df_degraded.to_csv(output_file, index=False)
print(f"\nSaved degraded dataset to: {output_file}")

# Quick comparison stats
print("\n" + "="*70)
print("COMPARISON: Original vs Degraded (168h forecast)")
print("="*70)

for col in COLUMN_MAPPING.keys():
    if col in df.columns:
        orig_mean = df[col].mean()
        deg_mean = df_degraded[col].mean()
        orig_std = df[col].std()
        deg_std = df_degraded[col].std()

        print(f"\n{col}:")
        print(f"  Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
        print(f"  Degraded: mean={deg_mean:.2f}, std={deg_std:.2f}")
        print(f"  Diff:     mean={deg_mean-orig_mean:.2f}, std={deg_std-orig_std:.2f}")

print("\n" + "="*70)
print("Ready for manual inspection!")
print("="*70)

# Derive column names from mapping by variable type (works for any city)
def col_for(var_type):
    """Return first column mapped to var_type, or None if not present."""
    return next((c for c, v in COLUMN_MAPPING.items() if v == var_type), None)

temp_col  = col_for('temperature')
solar_col = col_for('solar_radiation')
wind_col  = col_for('wind_speed')
precip_cols = [c for c, v in COLUMN_MAPPING.items() if v == 'precipitation']

# Specific cases to inspect
print("\n" + "="*70)
print("SPECIFIC CASES TO INSPECT")
print("="*70)

# Case 1: Temperature near 0°C with precipitation
if temp_col and precip_cols:
    precip_filter = pd.Series(False, index=df.index)
    for pc in precip_cols:
        precip_filter |= df[pc] > 0
    freezing = df[(df[temp_col] >= -2) & (df[temp_col] <= 2) & precip_filter].head(10)
    inspect_cols = [temp_col] + precip_cols
    print("\nCase 1: Near-freezing temperatures with precipitation")
    print("Original:")
    print(freezing[inspect_cols])
    print("\nDegraded:")
    print(df_degraded.loc[freezing.index, inspect_cols])
else:
    print("\nCase 1: skipped (temperature or precipitation column not found)")

# Case 2: Night-time solar radiation (should stay 0)
if solar_col:
    night = df[df[solar_col] == 0].head(10)
    print("\nCase 2: Night-time (zero solar radiation)")
    print(f"Original zeros: {(df[solar_col] == 0).sum()}")
    print(f"Degraded zeros: {(df_degraded[solar_col] == 0).sum()}")
    print(f"Any night values became non-zero? {(df_degraded.loc[night.index, solar_col] != 0).any()}")
else:
    print("\nCase 2: skipped (solar radiation column not found)")

# Case 3: Low wind speeds (check truncation at zero)
if wind_col:
    low_wind = df[df[wind_col] < 1.0].head(10)
    print("\nCase 3: Low wind speeds (<1 m/s)")
    print("Original:")
    print(low_wind[wind_col].values)
    print("Degraded:")
    print(df_degraded.loc[low_wind.index, wind_col].values)
else:
    print("\nCase 3: skipped (wind speed column not found)")