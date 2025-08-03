import pandas as pd
import numpy as np
import os
from glob import glob

# === 1. Define input and output directories ===
input_folder = 'data/processed/UMass Smart Dataset/featured_apartments'
output_folder = 'data/processed/UMass Smart Dataset/preprocessed_apartments'
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# === 2. List all feature-engineered apartment files ===
csv_files = glob(os.path.join(input_folder, 'Apt*_features.csv'))

# === 3. Process each apartment file ===
for file in csv_files:
    apt_name = os.path.basename(file).split('_')[0]  # Extract apartment name, e.g., Apt1

    # Load the dataset and ensure time is the index
    df = pd.read_csv(file, parse_dates=['time'], index_col='time')

    # Skip files if required columns are missing
    if not {'hour', 'month', 'weekday'}.issubset(df.columns):
        print(f"⚠️ Missing columns in {apt_name}, file skipped.")
        continue

    # Apply cyclical encoding to time-based features

    # Hour of day (0–23) → [sin, cos]
    df['hour_sin'] = np.round(np.sin(2 * np.pi * df['hour'] / 24), 14)
    df['hour_cos'] = np.round(np.cos(2 * np.pi * df['hour'] / 24), 14)

    # Day of week (0–6) → [sin, cos]
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # Month of year (1–12) → [sin, cos]
    df['month_sin'] = np.round(np.sin(2 * np.pi * (df['month'] - 1) / 12), 14)
    df['month_cos'] = np.round(np.cos(2 * np.pi * (df['month'] - 1) / 12), 14)

    # Drop original time-based categorical columns
    df.drop(columns=['hour', 'month', 'weekday'], inplace=True)

    # Save the preprocessed file
    output_path = os.path.join(output_folder, f'{apt_name}_preprocessed.csv')
    df.to_csv(output_path)
    print(f"✅ Cyclical encoding applied to {apt_name} → {output_path} ({df.shape[0]} rows)")
