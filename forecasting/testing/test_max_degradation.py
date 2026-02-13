"""
Test script: Apply maximum weather degradation and save for inspection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))



import pandas as pd
import numpy as np
from forecasting.config import ForecastConfig
from data.weather.weather_degradation import prepare_degradation_parameters, degrade_weather_dataset

config = ForecastConfig()
COLUMN_MAPPING = config.weather_degradation_mapping

# Load data
print("Loading data...")
df = pd.read_csv('data/SeoulBikeData.csv')
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
    seed=42  # Reproducible
)

# Save degraded dataset
output_file = 'data/SeoulBikeData_degraded_168h.csv'
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


# Add to test script:
print("\n" + "="*70)
print("SPECIFIC CASES TO INSPECT")
print("="*70)

# Case 1: Temperature near 0°C with precipitation
freezing = df[(df['Temperature'] >= -2) & (df['Temperature'] <= 2) & 
              ((df['Rainfall'] > 0) | (df['Snowfall'] > 0))].head(10)

print("\nCase 1: Near-freezing temperatures with precipitation")
print("Original:")
print(freezing[['Temperature', 'Rainfall', 'Snowfall']])
print("\nDegraded:")
print(df_degraded.loc[freezing.index, ['Temperature', 'Rainfall', 'Snowfall']])

# Case 2: Night-time solar radiation (should stay 0)
night = df[df['Solar Radiation'] == 0].head(10)
print("\nCase 2: Night-time (zero solar radiation)")
print(f"Original zeros: {(df['Solar Radiation'] == 0).sum()}")
print(f"Degraded zeros: {(df_degraded['Solar Radiation'] == 0).sum()}")
print(f"Any night values became non-zero? {(df_degraded.loc[night.index, 'Solar Radiation'] != 0).any()}")

# Case 3: Low wind speeds (check folded normal bias)
low_wind = df[df['Wind speed'] < 1.0].head(10)
print("\nCase 3: Low wind speeds (<1 m/s)")
print("Original:")
print(low_wind['Wind speed'].values)
print("Degraded:")
print(df_degraded.loc[low_wind.index, 'Wind speed'].values)

