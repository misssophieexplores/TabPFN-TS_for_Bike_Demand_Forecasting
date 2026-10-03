import requests
import pandas as pd
from datetime import datetime

# London coordinates
latitude = 51.5074
longitude = -0.1278

# Date range
start_date = "2015-01-04"
end_date = "2017-01-03"

# API endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": [
        "temperature_2m",           # Temperature in °C
        "relative_humidity_2m",     # Humidity in %
        "dew_point_2m",            # Dew point in °C
        "rain",                     # Rain in mm
        "snowfall",                # Snowfall in cm
        "wind_speed_10m",          # Wind speed at 10m
        "shortwave_radiation"      # Solar radiation in W/m²
    ],
    "timezone": "Europe/London",    # Local time
    "wind_speed_unit": "ms"        # Wind speed in m/s
}

print(f"Downloading weather data for London...")
print(f"Period: {start_date} to {end_date}")

# Make API request
response = requests.get(url, params=params)

# Check if request was successful
if response.status_code == 200:
    data = response.json()
    
    # Extract hourly data
    hourly_data = data['hourly']
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly_data['time']),
        'temperature_c': hourly_data['temperature_2m'],
        'humidity_percent': hourly_data['relative_humidity_2m'],
        'dew_point_c': hourly_data['dew_point_2m'],
        'rainfall_mm': hourly_data['rain'],
        'snowfall_cm': hourly_data['snowfall'],
        'wind_speed_ms': hourly_data['wind_speed_10m'],
        'solar_radiation_wm2': hourly_data['shortwave_radiation']
    })
    
    # Convert solar radiation from W/m² to MJ/m²
    # W/m² (hourly average) to MJ/m² (hourly total): multiply by 3.6 (3600 seconds / 1000)
    df['solar_radiation_mjm2'] = df['solar_radiation_wm2'] * 0.0036
    
    # Display info
    print(f"\nData downloaded successfully!")
    print(f"Total records: {len(df)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData summary:")
    print(df.describe())
    
    # Save to CSV
    output_filename = "data/weather_data/london_weather_2015-2017.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nData saved to: {output_filename}")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
else:
    print(f"Error: {response.status_code}")
    print(response.text)