import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load cleaned and enriched dataset with datetime parsed and set as index
df = pd.read_csv(
    'D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/new_features_cleaned_DemandData_2011-2018.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)


# Drop columns that become redundant after adding the new features
df.drop(columns=[
    'I014_FRENCH_FLOW', 'I014_BRITNED_FLOW', 'I014_MOYLE_FLOW', 'I014_EAST_WEST_FLOW',  # individual flows
    'FRENCH_FLOW', 'BRITNED_FLOW', 'MOYLE_FLOW', 'EAST_WEST_FLOW',                       
    'EMBEDDED_WIND_GENERATION', 'EMBEDDED_WIND_CAPACITY',                               # raw wind data
    'EMBEDDED_SOLAR_GENERATION', 'EMBEDDED_SOLAR_CAPACITY',                             # raw solar data
    'ENGLAND_WALES_DEMAND',                                                             # regional demand
    'ND', 'TSD', 'I014_TSD',                                                            # national demand
    'PUMP_STORAGE_PUMPING',                                                                
    'SETTLEMENT_PERIOD'                                                                                            
                                                                                            
], inplace=True)


# Cyclical encoding of temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour']   / 24)
df['hour_sin'] = np.round(df['hour_sin'], 14)   # Rounding to get exact zeros or ones where expected

df['hour_cos'] = np.cos(2 * np.pi * df['hour']   / 24)
df['hour_cos'] = np.round(df['hour_cos'], 14)   # Rounding to get exact zeros or ones where expected

df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/ 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/ 7)

# subtract 1 so months (1–12) map to 0–11 and December sits just before January on the circle
df['month_sin'] = np.sin(2 * np.pi * (df['month']-1)/ 12)
df['month_sin'] = np.round(df['month_sin'], 14)    # Rounding to get exact zeros or ones where expected

df['month_cos'] = np.cos(2 * np.pi * (df['month']-1)/ 12)
df['month_cos'] = np.round(df['month_cos'], 14)    # Rounding to get exact zeros or ones where expected


# Drop hour, month and weekday 
df.drop(columns=['weekday', 'month', 'hour'], inplace=True)

# Standard scaling of features
features = ['NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','I014_ND_lag_1','I014_ND_lag_2',
            'I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336','I014_ND_mean_48',
            'I014_ND_mean_336','net_import','wind_capacity_factor','solar_capacity_factor'
]

scaler = StandardScaler()   # z-score normalization

# Fit the scaler on the selected numeric features and transform them
df[features] = scaler.fit_transform(df[features])

df.to_csv('D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv')



