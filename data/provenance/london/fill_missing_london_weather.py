import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()
VISUALCROSSING_KEY = os.getenv('VISUALCROSSING_KEY')

LONDON_LAT = 51.5074
LONDON_LON = -0.1278
BIKE_DATA_PATH = "data/LondonBikeData.csv"
OUTPUT_PATH = "data/LondonBikeData.csv"

# Load bike data and get date range
print("Loading bike data...")
bike_df = pd.read_csv(BIKE_DATA_PATH)
bike_df['timestamp'] = pd.to_datetime(bike_df['timestamp'])
START_DATE = bike_df['timestamp'].min().strftime('%Y-%m-%d')
END_DATE = bike_df['timestamp'].max().strftime('%Y-%m-%d')
print(f"Date range: {START_DATE} to {END_DATE}")

# # Download Open-Meteo weather
# print("Downloading weather data...")
# url = "https://archive-api.open-meteo.com/v1/archive"
# params = {
#     "latitude": LONDON_LAT,
#     "longitude": LONDON_LON,
#     "start_date": START_DATE,
#     "end_date": END_DATE,
#     "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
#                "rain", "snowfall", "wind_speed_10m", "shortwave_radiation"],
#     "timezone": "Europe/London",
#     "wind_speed_unit": "ms"
# }

# response = requests.get(url, params=params)
# data = response.json()['hourly']

# weather_df = pd.DataFrame({
#     'timestamp': pd.to_datetime(data['time']),
#     'temperature_c': data['temperature_2m'],
#     'humidity_percent': data['relative_humidity_2m'],
#     'dew_point_c': data['dew_point_2m'],
#     'rainfall_mm': data['rain'],
#     'snowfall_cm': data['snowfall'],
#     'wind_speed_ms': data['wind_speed_10m'],
#     'solar_radiation_wm2': data['shortwave_radiation']
# })
# weather_df['solar_radiation_mjm2'] = weather_df['solar_radiation_wm2'] * 0.0036

# # Merge
# merged_df = pd.merge(bike_df, weather_df, on='timestamp', how='right')
# print(f"Merged: {len(merged_df)} records")

#load merged_df directly:
merged_df = pd.read_csv(BIKE_DATA_PATH)
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
# Fill missing weather from Visual Crossing
weather_cols = ['temperature_c', 'humidity_percent', 'dew_point_c', 
                'rainfall_mm', 'snowfall_cm', 'wind_speed_ms', 
                'solar_radiation_wm2', 'solar_radiation_mjm2']

missing_weather = merged_df[merged_df[weather_cols].isna().any(axis=1)]

if len(missing_weather) > 0:
    print(f"Filling {len(missing_weather)} missing weather records...")
    
    dates = sorted(set(ts.date() for ts in missing_weather['timestamp']))
    vc_data = []
    
    for date in dates:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/London,UK/{date}/{date}"
        params = {'key': VISUALCROSSING_KEY, 'unitGroup': 'metric', 'include': 'hours'}
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            for hour in response.json()['days'][0]['hours']:
                vc_data.append({
                    'timestamp': pd.to_datetime(f"{date} {hour['datetime']}"),
                    'temperature_c': hour.get('temp'),
                    'humidity_percent': hour.get('humidity'),
                    'dew_point_c': hour.get('dew'),
                    'rainfall_mm': hour.get('precip', 0) or 0,
                    'snowfall_cm': (hour.get('snow', 0) or 0) * 10,
                    'wind_speed_ms': (hour.get('windspeed', 0) or 0) / 3.6,
                    'solar_radiation_wm2': hour.get('solarradiation', 0) or 0
                })
            time.sleep(0.3)
    
    vc_df = pd.DataFrame(vc_data)
    vc_df['solar_radiation_mjm2'] = vc_df['solar_radiation_wm2'] * 0.0036
    
    for col in weather_cols:
        merged_df[col] = merged_df[col].astype(float)
    
    for idx, row in missing_weather.iterrows():
        vc_row = vc_df[vc_df['timestamp'] == row['timestamp']]
        if not vc_row.empty:
            for col in weather_cols:
                merged_df.at[idx, col] = vc_row[col].values[0]

# Map to original column names
merged_df['t1'] = merged_df['temperature_c']
merged_df['hum'] = merged_df['humidity_percent']
merged_df['wind_speed'] = merged_df['wind_speed_ms']

# Fill derived columns
merged_df['is_weekend'] = (merged_df['timestamp'].dt.dayofweek >= 5).astype(int)

for idx, row in merged_df[merged_df['is_holiday'].isna()].iterrows():
    date = row['timestamp'].date()
    same_date = merged_df[(merged_df['timestamp'].dt.date == date) & 
                          (merged_df['is_holiday'].notna())]
    
    if not same_date.empty:
        merged_df.at[idx, 'is_holiday'] = same_date['is_holiday'].iloc[0]
        merged_df.at[idx, 'season'] = same_date['season'].iloc[0]

# Manual fix for Sept 2, 2016 if needed
sept2 = merged_df['timestamp'].dt.date == pd.Timestamp('2016-09-02').date()
if sept2.any():
    merged_df.loc[sept2, 'is_holiday'] = 0
    merged_df.loc[sept2, 'is_weekend'] = 0
    merged_df.loc[sept2, 'season'] = 2

# Save
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")

# Report missing
important_cols = ['cnt', 't1', 'hum', 'wind_speed', 'dew_point_c', 
                  'solar_radiation_wm2', 'rainfall_mm', 'snowfall_cm',
                  'is_holiday', 'is_weekend', 'season']
missing = merged_df[important_cols].isna().sum()

if missing.sum() > 0:
    print("\nMissing values:")
    print(missing[missing > 0])
else:
    print("✓ All important columns complete")