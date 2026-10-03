import pandas as pd
import numpy as np

df = pd.read_csv("data/london_merged_weather_bikes.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Track which rows were originally missing
originally_missing = df['cnt'].isna()

# Impute: 24h first, then 168h as fallback
df['cnt'] = df['cnt'].fillna(df['cnt'].shift(24))
df['cnt'] = df['cnt'].fillna(df['cnt'].shift(168))

# Last resort: hourly average
if df['cnt'].isna().any():
    hourly_avg = df.groupby(df['timestamp'].dt.hour)['cnt'].transform('mean')
    df['cnt'] = df['cnt'].fillna(hourly_avg)

# Add Functioning Day column
df['Functioning Day'] = 'Yes'
df.loc[originally_missing, 'Functioning Day'] = 'No'

df.to_csv("data/LondonBikeData.csv", index=False)