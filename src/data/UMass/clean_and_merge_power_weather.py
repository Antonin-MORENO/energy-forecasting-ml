import pandas as pd
import os
from glob import glob

# === 1. Load weather data once and concatenate ===
dfw_2014 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2014.csv')
dfw_2015 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2015.csv')
dfw_2016 = pd.read_csv('data/raw/UMass Smart Dataset/apartment-weather/apartment-weather/apartment2016.csv')

# Combine all years of weather data into a single DataFrame
dfw_all = pd.concat([dfw_2014, dfw_2015, dfw_2016], ignore_index=True)

# Convert UNIX timestamp to datetime and set as index
dfw_all['time'] = pd.to_datetime(dfw_all['time'], unit='s')
dfw_all = (
    dfw_all
    .drop_duplicates(subset='time', keep='first')  # Remove duplicate timestamps
    .set_index('time')                             # Set datetime as index
)
print("Merged weather data:", dfw_all.shape)

# === 2. Function to clean and resample an electricity file ===
def load_and_clean_electricity(filepath):
    # Load file with time and power columns
    df = pd.read_csv(filepath, header=None, names=['time', 'power [kW]'])
    df['time'] = pd.to_datetime(df['time'])

    # Remove duplicates and set index
    df = df.drop_duplicates(subset='time', keep='first').set_index('time')

    # Resample to hourly average
    df = df.resample('1h').mean()

    # Fill missing values with average of forward and backward fill
    mask = df['power [kW]'].isna()
    ff = df['power [kW]'].ffill()
    bb = df['power [kW]'].bfill()
    df.loc[mask, 'power [kW]'] = ((ff + bb) / 2)[mask]
    
    return df

# === 3. List all available apartments and their yearly files ===
root_folder = 'data/raw/UMass Smart Dataset/apartment'
years = ['2014', '2015', '2016']

# Dictionary: apartment → {year: filepath}
apartment_files = {}

for year in years:
    year_folder = os.path.join(root_folder, year)
    for file in glob(os.path.join(year_folder, 'Apt*.csv')):
        basename = os.path.basename(file)
        apt_name = basename.split('_')[0]  # e.g., Apt1 from Apt1_2014.csv
        apartment_files.setdefault(apt_name, {})[year] = file

# === 4. Process each apartment ===
output_folder = 'data/raw/UMass Smart Dataset/processed_apartments'
os.makedirs(output_folder, exist_ok=True)

for apt, files in apartment_files.items():
    print(f"\n=== Processing {apt} ===")
    dfs = []
    for year in years:
        if year in files:
            df = load_and_clean_electricity(files[year])
            dfs.append(df)
        else:
            print(f"  ⚠️ Missing file for {apt} in {year}")
    
    if not dfs:
        print(f"  ❌ No files found for {apt}, skipping.")
        continue

    # Concatenate and sort by time
    dfe_all = pd.concat(dfs).sort_index()

    # Merge with weather data
    df_merged = dfe_all.merge(dfw_all, on='time', how='left')

    # Drop initial rows if needed (e.g., corrupted or misaligned)
    df_merged.drop(df_merged.index[:5], inplace=True)

    # Remove unnecessary column if present
    if 'cloudCover' in df_merged.columns:
        df_merged.drop(columns=['cloudCover'], inplace=True)

    # Interpolate missing values over time
    df_merged.interpolate(method='time', limit_direction='both', inplace=True)

    # Export final merged dataset
    output_path = os.path.join(output_folder, f'{apt}_merged.csv')
    df_merged.to_csv(output_path)
    print(f"✅ File saved: {output_path} — {df_merged.shape[0]} rows")
