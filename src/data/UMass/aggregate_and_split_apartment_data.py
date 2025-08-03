import pandas as pd
import os
from glob import glob

# === Directories ===
input_folder = 'data/processed/UMass Smart Dataset/preprocessed_apartments'
merged_output_path = 'data/processed/UMass Smart Dataset/multi_apartment_train.csv'
test_output_path   = 'data/processed/UMass Smart Dataset/test_apartment.csv'

# === Test split parameters (can be modified) ===
test_apartment = 'Apt1'  # Apartment used for testing
test_year = 2016         # Year used for the test set

# === 1. Load and merge all preprocessed apartment CSV files ===
csv_files = glob(os.path.join(input_folder, 'Apt*_preprocessed.csv'))
dfs = []

for file in csv_files:
    apt = os.path.basename(file).split('_')[0]  # Extract apartment name (e.g., Apt1)
    df = pd.read_csv(file, parse_dates=['time'])  # Read CSV and parse time column
    df['apartment'] = apt                         # Add apartment label as a column
    dfs.append(df)

# Concatenate all apartments into a single DataFrame, sorted by time
df_all = pd.concat(dfs).sort_values('time').reset_index(drop=True)

# === 2. Split into training and test set based on apartment and year ===
mask_test = (df_all['apartment'] == test_apartment) & (df_all['time'].dt.year == test_year)
df_test = df_all[mask_test].copy()      # Test set = selected apartment and year
df_train = df_all[~mask_test].copy()    # Training set = everything else

# === 3. Save the datasets to disk ===
df_train.to_csv(merged_output_path, index=False)
df_test.to_csv(test_output_path, index=False)

print(f"✅ Training data: {df_train.shape[0]} rows → {merged_output_path}")
print(f"✅ Test data ({test_apartment}, {test_year}): {df_test.shape[0]} rows → {test_output_path}")
