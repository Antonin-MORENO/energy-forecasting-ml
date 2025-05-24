import pandas as pd

# Load cleaned dataset with datetime parsed and set as index
df = pd.read_csv(
    'D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/cleaned_DemandData_2011-2018.csv',
    parse_dates=["datetime"],
    index_col="datetime"
)

# Add calendar features ===
df['hour'] = df.index.hour                # Hour of the day (0–23)
df['weekday'] = df.index.weekday          # Day of the week (0=Monday, ..., 6=Sunday)
df['month'] = df.index.month              # Month of the year (1–12)
df['is_weekend'] = (df['weekday'] >= 5).astype(int)  # Weekend indicator (1 if Sat/Sun, else 0)

# Add lag features (time-shifted demand) 
lags = [1, 2, 48, 96, 336]  # Lag periods in 30-minute intervals (e.g. 48 = previous day)

for lag in lags:
    df[f'I014_ND_lag_{lag}'] = df['I014_ND'].shift(lag)

# Add rolling mean features (past moving averages) ===
windows = [48, 336]  # Window sizes: 1 day, 1 week (in 30-min steps)

for w in windows:
    df[f'I014_ND_mean_{w}'] = df['I014_ND'].shift(1).rolling(w).mean()  # Exclude current time to prevent leakage

# Export the enriched dataset with new features 
df.to_csv('D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/new_features_cleaned_DemandData_2011-2018.csv')
