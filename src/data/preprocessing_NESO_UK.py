import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load cleaned and enriched dataset with datetime parsed and set as index
df = pd.read_csv(
    'D:/Data science/energy-forecasting-ml/data/processed/NESO_UK/new_features_cleaned_DemandData_2011-2018.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)


# Cyclical encoding of temporal features
df['hour_sin']    = np.sin(2 * np.pi * df['hour']   / 24)
df['hour_cos']    = np.cos(2 * np.pi * df['hour']   / 24)
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/ 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/ 7)

# subtract 1 so months (1–12) map to 0–11 and December sits just before January on the circle
df['month_sin']   = np.sin(2 * np.pi * (df['month']-1)/ 12)
df['month_cos']   = np.cos(2 * np.pi * (df['month']-1)/ 12)

# Standard scaling of numeric features
numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'ND', 'I014_ND', 'TSD', 'I014_TSD']
numeric_feats = [c for c in numeric_feats if c not in exclude]



scaler = StandardScaler()   # z-score normalization

# Fit the scaler on the selected numeric features and transform them
df[numeric_feats] = scaler.fit_transform(df[numeric_feats])





