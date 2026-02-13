from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

# Fetch dataset 
seoul_bike_sharing_demand = fetch_ucirepo(id=560) 

# Data (as pandas dataframes) 
X = seoul_bike_sharing_demand.data.features 
y = seoul_bike_sharing_demand.data.targets 

# Combine for easier handling
df = pd.concat([X, y], axis=1)

# Create datetime column and name it 'Date' (to match our config/scripts)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y') + pd.to_timedelta(df['Hour'], unit='h')

# Drop the separate Hour column (info now in Date)
df = df.drop(columns=['Hour'])

# Data insertion: use values from previous week in case of non-functioning day
# Identify non-functional periods and set to NaN for imputation
mask = df['Functioning Day'] == 'No'
df.loc[mask, 'Rented Bike Count'] = np.nan

# Impute using 7-day lag (168 hours)
df['Rented Bike Count'] = df['Rented Bike Count'].fillna(df['Rented Bike Count'].shift(168))

# Handle edge case: remaining NaNs get hourly average from functional periods
if df['Rented Bike Count'].isna().any():
    functional_hourly_avg = df[df['Functioning Day'] == 'Yes'].groupby(
        df[df['Functioning Day'] == 'Yes']['Date'].dt.hour
    )['Rented Bike Count'].mean()
    
    missing_mask = df['Rented Bike Count'].isna()
    df.loc[missing_mask, 'Rented Bike Count'] = df.loc[missing_mask, 'Date'].dt.hour.map(functional_hourly_avg).values

# Save to project root data folder
df.to_csv('data/SeoulBikeData.csv', index=False)
print(f"Saved {len(df)} rows to data/SeoulBikeData.csv")