import pandas as pd
import numpy as np

# Load both datasets
print("Loading datasets...")
weather_df = pd.read_csv("data/weather_data/london_weather_2015-2017.csv")
bike_df = pd.read_csv("data/london_og.csv")  # Update with your actual filename

print(f"Weather data: {len(weather_df)} rows")
print(f"Bike data: {len(bike_df)} rows")

# Convert timestamp columns to datetime
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
bike_df['timestamp'] = pd.to_datetime(bike_df['timestamp'])

# Check timestamp ranges
print(f"\nWeather data range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
print(f"Bike data range: {bike_df['timestamp'].min()} to {bike_df['timestamp'].max()}")

# Perform RIGHT merge (keep all weather data)
merged_df = pd.merge(
    bike_df,
    weather_df,
    on='timestamp',
    how='right',
    indicator=True
)

# Show merge statistics
print("\n" + "="*60)
print("MERGE STATISTICS")
print("="*60)
print(f"Total rows in merged dataset: {len(merged_df)}")
print(f"\nMerge breakdown:")
print(merged_df['_merge'].value_counts())
print(f"\n  - 'both': Rows where bike and weather data matched")
print(f"  - 'right_only': Rows where only weather data exists (no bike data)")
print(f"  - 'left_only': Should be 0 for right merge")

# Calculate how many bike records were not in weather data
unmatched_bike = len(bike_df) - merged_df['_merge'].value_counts().get('both', 0)
print(f"\nBike records NOT in weather data: {unmatched_bike}")

# Remove the merge indicator column
merged_df = merged_df.drop('_merge', axis=1)

# Show missing values
print("\n" + "="*60)
print("MISSING VALUES IN MERGED DATASET")
print("="*60)
print(merged_df.isnull().sum())

# Show first few rows
print("\n" + "="*60)
print("FIRST FEW ROWS")
print("="*60)
print(merged_df.head(10))

# Show some rows where bike data is missing (if any)
missing_bike_data = merged_df[merged_df['cnt'].isnull()] if 'cnt' in merged_df.columns else pd.DataFrame()
if len(missing_bike_data) > 0:
    print("\n" + "="*60)
    print(f"SAMPLE ROWS WITH MISSING BIKE DATA ({len(missing_bike_data)} total)")
    print("="*60)
    print(missing_bike_data.head())

# Sort by timestamp
merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

# Save merged dataset
output_filename = "data/london_merged_weather_bikes.csv"
merged_df.to_csv(output_filename, index=False)
print(f"\n{'='*60}")
print(f"Merged dataset saved to: {output_filename}")
print(f"Total rows: {len(merged_df)}")
print(f"{'='*60}")

# Show summary statistics
print("\nSummary statistics for key columns:")
print(merged_df[['temperature_c', 'humidity_percent', 'rainfall_mm', 'wind_speed_ms']].describe())