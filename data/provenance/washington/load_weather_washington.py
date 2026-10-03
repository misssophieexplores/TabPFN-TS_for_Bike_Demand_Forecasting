import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os


load_dotenv()
VISUALCROSSING_KEY = os.getenv('VISUALCROSSING_KEY')

DC_LAT = 38.9073
DC_LON = -77.0369
BIKE_DATA_PATH = "data/washington_hour.csv"
WEATHER_DATA_PATH = "data/weather_data/washington_dc_weather_2011-2012.csv"
OUTPUT_PATH = "data/WashingtonBikeData.csv"


# Load bike data and create timestamp
print("Loading bike data...")
bike_df = pd.read_csv(BIKE_DATA_PATH)
bike_df['timestamp'] = pd.to_datetime(bike_df['dteday']) + pd.to_timedelta(bike_df['hr'], unit='h')


# Load weather data
print("Loading weather data...")
weather_df = pd.read_csv(WEATHER_DATA_PATH)
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])


# Create complete hourly range
START_DATE = bike_df['timestamp'].min()
END_DATE = bike_df['timestamp'].max()
complete_range = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
complete_df = pd.DataFrame({'timestamp': complete_range})
print(f"Complete range: {len(complete_df)} hours")


# Merge existing bike data
complete_df = pd.merge(complete_df, bike_df, on='timestamp', how='left')

# Fill time-based columns from timestamp
complete_df['weekday'] = complete_df['timestamp'].dt.dayofweek
complete_df['yr'] = complete_df['timestamp'].dt.year - 2011
complete_df['mnth'] = complete_df['timestamp'].dt.month
complete_df['hr'] = complete_df['timestamp'].dt.hour

# Fill season, holiday, workingday from same date
for idx, row in complete_df[complete_df['season'].isna()].iterrows():
    date = row['timestamp'].date()
    same_date = complete_df[(complete_df['timestamp'].dt.date == date) & (complete_df['season'].notna())]
    
    if not same_date.empty:
        if pd.isna(row['season']):
            complete_df.at[idx, 'season'] = same_date['season'].iloc[0]
        if pd.isna(row['holiday']):
            complete_df.at[idx, 'holiday'] = same_date['holiday'].iloc[0]
        if pd.isna(row['workingday']):
            complete_df.at[idx, 'workingday'] = same_date['workingday'].iloc[0]

print(f"Bike data filled: {len(complete_df)} records")


# Merge with weather
merged_df = pd.merge(complete_df, weather_df, on='timestamp', how='left')
print(f"After weather merge: {len(merged_df)} records")

# Check for missing weather data
weather_cols = ['temperature_c', 'humidity_percent', 'dew_point_c', 
                'rainfall_mm', 'snowfall_cm', 'wind_speed_ms', 
                'solar_radiation_wm2', 'solar_radiation_mjm2']

missing_weather = merged_df[merged_df[weather_cols].isna().any(axis=1)]

if len(missing_weather) > 0:
    print(f"Filling {len(missing_weather)} missing weather records...")
    
    dates = sorted(set(ts.date() for ts in missing_weather['timestamp']))
    vc_data = []
    
    for date in dates:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Washington,DC/{date}/{date}"
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


# Map weather to original bike column names if needed
if 'temp' in merged_df.columns:
    merged_df['temp'] = merged_df['temp'].fillna(merged_df['temperature_c'])
if 'hum' in merged_df.columns:
    merged_df['hum'] = merged_df['hum'].fillna(merged_df['humidity_percent'])
if 'windspeed' in merged_df.columns:
    merged_df['windspeed'] = merged_df['windspeed'].fillna(merged_df['wind_speed_ms'])

merged_df.isna().sum()
# TODO: Check seasonality -> then decide on imputation strategy (likely: 24h, then a week!)
# Track which rows were originally missing
originally_missing = merged_df['cnt'].isna()

# Impute: 24h first, then 168h as fallback
merged_df['cnt'] = merged_df['cnt'].fillna(merged_df['cnt'].shift(24))
merged_df['cnt'] = merged_df['cnt'].fillna(merged_df['cnt'].shift(168))

# Last resort: hourly average
if merged_df['cnt'].isna().any():
    hourly_avg = merged_df.groupby(merged_df['timestamp'].dt.hour)['cnt'].transform('mean')
    merged_df['cnt'] = merged_df['cnt'].fillna(hourly_avg)

# Add Functioning Day column
merged_df['Functioning Day'] = 'Yes'
merged_df.loc[originally_missing, 'Functioning Day'] = 'No'



# Report missing
important_cols = ['cnt', 'temp', 'hum', 'windspeed', 'temperature_c', 
                  'humidity_percent', 'wind_speed_ms', 'dew_point_c',
                  'solar_radiation_mjm2', 'rainfall_mm', 'snowfall_cm',
                  'season', 'holiday', 'weekday']

existing_cols = [col for col in important_cols if col in merged_df.columns]
missing = merged_df[existing_cols].isna().sum()

if missing.sum() > 0:
    print("\nMissing values:")
    print(missing[missing > 0])
else:
    print("✓ All important columns complete")

# Keep the published columns (added 3 Oct 2026: this selection was done in
# load_weather_washington.ipynb; with it, this script reproduces
# WashingtonBikeData.csv as it was before visibility and the v7 columns).
KEEP_COLS = [
    "timestamp", "season", "holiday", "casual", "registered", "cnt",
    "temperature_c", "humidity_percent", "dew_point_c", "rainfall_mm",
    "snowfall_cm", "wind_speed_ms", "solar_radiation_wm2",
    "solar_radiation_mjm2", "Functioning Day",
]
merged_df = merged_df[KEEP_COLS]

# Save
merged_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")

