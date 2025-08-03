import pandas as pd

# Load weather data from 2014 and convert UNIX timestamps to datetime
dfw_2014 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2014.csv')
dfw_2014['time'] = pd.to_datetime(dfw_2014['time'], unit='s')
#print(dfw_2014.isna().sum())

# Check for gaps in the datetime sequence
delta = dfw_2014['time'].diff().value_counts()
print(delta)


# Load weather data from 2015 and convert timestamps
dfw_2015 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2015.csv')
dfw_2015['time'] = pd.to_datetime(dfw_2015['time'], unit='s')
#print(dfw_2015.isna().sum())

# Check for gaps in the datetime sequence
delta = dfw_2015['time'].diff().value_counts()
#print(delta)

# Load weather data from 2016 and convert timestamps
dfw_2016 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2016.csv')
dfw_2016['time'] = pd.to_datetime(dfw_2016['time'], unit='s')
#print(dfw_2016.isna().sum())


# Check for gaps in the datetime sequence
delta = dfw_2016['time'].diff().value_counts()
#print(delta)

# Select only time, temperature, pressure, and humidity columns from each year
dfs = [
    dfw_2014[['time','temperature', 'pressure', 'humidity']], 
    dfw_2015[['time','temperature', 'pressure', 'humidity']], 
    dfw_2016[['time','temperature', 'pressure', 'humidity']]
]




# Concatenate all years into a single DataFrame
ts = pd.concat(dfs, ignore_index=True)

# Show the number of missing values per column
print (ts.isna().sum())

# Replace missing values in 'pressure' using linear interpolation from adjacent rows
mask = ts['pressure'].isna()

ts.loc[mask, 'pressure'] = (
    ts['pressure'].shift(1) + 
    ts['pressure'].shift(-1)
)[mask] / 2

# Verify that missing values have been handled
print (ts.isna().sum())

# Check the distribution of time differences between consecutive records
print(ts['time'].diff().value_counts())


# Identify timestamps where the time step is not 1 hour
delta = ts['time'].diff()
mask = delta != pd.Timedelta(hours=1)
discont = ts.loc[mask, ['time']]

print("Rows with time steps not equal to 1 hour:")
print(discont)

# Remove duplicate timestamps
ts = ts.drop_duplicates(subset='time')

# Recheck time step distribution and remaining discontinuities
print(ts['time'].diff().value_counts())

delta = ts['time'].diff()
mask = delta != pd.Timedelta(hours=1)
discont = ts.loc[mask, ['time']]

# Create a copy to generate lag features
tslag = ts.copy()
selected_lags = [1, 24, 48, 72, 168]

# Create lagged versions of the temperature column
for lag in selected_lags:
    tslag[f'temperature_lag_{lag}'] = tslag['temperature'].shift(lag)


# Drop rows with NaNs introduced by lagging
tslag.dropna(inplace=True)

# Show resulting DataFrame and save to CSV
print(tslag)
tslag.to_csv('weather_lag.csv')
