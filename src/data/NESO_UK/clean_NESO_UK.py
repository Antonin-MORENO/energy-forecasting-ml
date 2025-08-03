import pandas as pd 

# Load all the datasets
df1 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2011-2016.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')
df2 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2017.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')
df3 = pd.read_csv('D:/Data science/energy-forecasting-ml/data/raw/NESO_UK/DemandData_2018_4.csv', parse_dates=['SETTLEMENT_DATE'], date_format='%d-%b-%Y')

# Concatenation of the three datasets
df = pd.concat([df1,df2,df3], ignore_index=True)

# Check if there any missing values 
print(df.isnull().sum().sort_values(ascending=False))

# Drop column I014_TSD.1 (duplicate of I014_ND)
df.drop(columns=['I014_TSD.1'], inplace=True)

# Reconstruct timestamps by adding 30-minute intervals to the settlement date
df['datetime'] = df['SETTLEMENT_DATE'] + pd.to_timedelta((df['SETTLEMENT_PERIOD'] - 1) * 30, unit='min')
df.drop(columns=['SETTLEMENT_DATE'], inplace=True)


# Set this datetime column as the index for time series analysis
df.set_index('datetime', inplace=True)

# Check for any discontinuities in the datetime index (e.g. gaps or duplicates)
gaps = df.index.to_series().diff().value_counts()
print("Time gaps detected between consecutive rows:")
print(gaps)

# Locate the rows directly before abnormal time gaps (e.g. 01:30:00 or -00:30:00)
# These likely correspond to DST transitions 
time_diffs = df.index.to_series().diff()
anomalies = df[time_diffs.isin([pd.Timedelta('01:30:00'), pd.Timedelta('-00:30:00')])]
print("Temporal anomalies detected (likely DST-related):")
print(anomalies)


# Remove duplicated timestamps from the index, keeping only the first occurrence
df = df[~df.index.duplicated(keep='first')]



# Create a complete datetime index with a fixed 30-minute interval
# This ensures regular time steps between the start and end of the dataset
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='30min')


# Reindex the DataFrame to align it with the complete 30-min interval index
# This inserts rows with NaNs for any missing timestamps
df = df.reindex(full_index)


# Add a flag to indicate which rows were interpolated 
df['is_interpolated'] = df.isnull().any(axis=1).astype(int)


# Interpolate missing values only for numeric columns using time-based interpolation
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].interpolate(method='time').round(0)

# Check if the discontinuity has been well fixed
df = df.reset_index().rename(columns={'index': 'datetime'})
gaps = df['datetime'].diff().value_counts()
print("Time gaps detected between consecutive datetimes:")
print(gaps)


# Check for outliers in the National Demand

Q1 = df['I014_ND'].quantile(0.25)   # Compute the 25th percentile 
Q3 = df['I014_ND'].quantile(0.75)   # Compute the 75th percentile 

# Compute the interquartile range
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers (1.5 * IQR below Q1) and (1.5 * IQR above Q3)
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Create a boolean mask marking True for values outside the [lower, upper] range
outlier_mask = (df['I014_ND'] < lower) | (df['I014_ND'] > upper)

# Print the computed outlier thresholds
print("Upper bound:", upper)
print("Lower bound:", lower)

# Print the total number of outliers detected
print("Number of outliers:", outlier_mask.sum())

# Display the National Demand values flagged as outliers
print(df.loc[outlier_mask, 'I014_ND'].head(15))



#df['I014_ND'] = df['I014_ND'].clip(lower, upper)

print(f"\n Cleaning complete. Total rows: {len(df)} | Interpolated: {df['is_interpolated'].sum()}")

# Export the cleaned and indexed DataFrame to a CSV file
df.to_csv('D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/cleaned_DemandData_2011-2018.csv', index=False)