import pandas as pd
import os
from glob import glob

# === 1. Define input and output directories ===
input_folder = 'data/raw/UMass Smart Dataset/processed_apartments'
output_folder = 'data/processed/UMass Smart Dataset/featured_apartments'
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# === 2. List all previously generated merged CSV files ===
csv_files = glob(os.path.join(input_folder, 'Apt*_merged.csv'))

# === 3. Define the lags to compute (in hours) ===
lags = [1, 24, 168]  # 1 hour, 1 day, 1 week

# === 4. Process each apartment file ===
for file in csv_files:
    apt_name = os.path.basename(file).split('_')[0]  # Extract apartment name, e.g., "Apt1"

    # Load data and keep only relevant columns
    df = pd.read_csv(file, parse_dates=['time'], index_col='time')
    
    # Skip files with missing required columns
    if not {'power [kW]', 'temperature'}.issubset(df.columns):
        print(f"⚠️ Missing columns in {apt_name}, file skipped.")
        continue

    # Keep only power and temperature
    df = df[['power [kW]', 'temperature']].copy()

    # Add time-based features
    df['hour'] = df.index.hour             # Hour of the day (0-23)
    df['month'] = df.index.month           # Month (1-12)
    df['weekday'] = df.index.weekday       # Day of the week (0=Monday)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)  # 1 if weekend, else 0

    # Add lag features for power consumption
    for lag in lags:
        df[f'power [kW]_lag_{lag}'] = df['power [kW]'].shift(lag)

    # Drop rows with missing values (from lag creation)
    df.dropna(inplace=True)

    # Export the feature-engineered dataset
    output_path = os.path.join(output_folder, f'{apt_name}_features.csv')
    df.to_csv(output_path)
    print(f"✅ Features saved for {apt_name} → {output_path} ({df.shape[0]} rows)")
