import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from src.data.data_handler import DataHandler
from src.models.xgboost_model import XGBoostModel

# === EXPÉRIMENTATION ===
exp_name = 'xgboost_smartumass_experiment_multiappart_temp_feature'
exp_metrics = os.path.join('outputs', 'experiments_UMass', exp_name, 'metrics')
os.makedirs(exp_metrics, exist_ok=True)

# === Chemins ===
csv_path_train = 'data/processed/UMass Smart Dataset/multi_apartment_train.csv'
csv_path_test  = 'data/processed/UMass Smart Dataset/test_apartment.csv'

# === Colonnes ===
date_col      = 'time'
target_col    = 'power [kW]'
feature_cols  = [
    'temperature','hour_sin', 'hour_cos', 'month_cos', 'month_sin', 'weekday_cos', 'weekday_sin', 'is_weekend'
]
no_scale_cols = ['hour_sin', 'hour_cos', 'month_cos', 'month_sin', 'weekday_cos', 'weekday_sin', 'is_weekend']

# === DataHandler ===
dh = DataHandler(
    csv_path      = csv_path_train,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_years = None,
    holdout_ratio = 0,
    scaler_type   = 'standard'
)

# === Chargement et scaling complet ===
df_full = dh.load_data()
df_full[target_col] = np.log1p(df_full[target_col])
X_trval, y_trval, _, _ = dh.scale_split(df_full, df_full)

# === GridSearch XGBoost ===
param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [3, 5, 7],
    'learning_rate'    : [0.01, 0.05, 0.1],
    'subsample'        : [0.6, 0.8, 1.0],
    'colsample_bytree' : [0.6, 0.8, 1.0]
}
tscv = TimeSeriesSplit(n_splits=5)

gs = GridSearchCV(
    estimator           = XGBoostModel(),
    param_grid          = param_grid,
    cv                  = tscv,
    scoring             = 'neg_mean_absolute_error',
    return_train_score  = True,
    n_jobs              = -1,
    verbose             = 3,
    refit               = True
)

print("🚀 Lancement GridSearchCV sur multi-appartements...")
start = time.perf_counter()
gs.fit(X_trval, y_trval)
print(f"✅ GridSearch terminé en {time.perf_counter() - start:.1f}s")

# === Meilleur modèle ===
best_model  = gs.best_estimator_
best_params = gs.best_params_
best_model.fit(X_trval, y_trval)

# === Test final sur un seul appartement (ex: Apt1, 2016) ===
print("\n📊 Évaluation spécifique sur test_apartment.csv")

df_test_apartment = pd.read_csv(csv_path_test, parse_dates=['time'])

X_test_apartment = df_test_apartment[feature_cols].copy()
y_test_apartment = df_test_apartment[target_col].values

# Appliquer le scaler appris
cols_to_scale = [col for col in feature_cols if col not in no_scale_cols]
X_test_apartment[cols_to_scale] = dh.scaler.transform(X_test_apartment[cols_to_scale])

# === Prédictions
t0 = time.perf_counter()
y_pred_apartment = best_model.predict(X_test_apartment.values)
pred_time = time.perf_counter() - t0

# === Inverse log1p
y_pred_apartment = np.expm1(y_pred_apartment)

# === Calcul manuel des métriques
errors = y_pred_apartment - y_test_apartment

def smape(y_true, y_pred):
    num   = np.abs(y_pred - y_true)
    denom = np.abs(y_pred) + np.abs(y_true)
    denom[denom == 0] = 1e-8
    return np.mean(2 * num / denom) * 100

final_test_metrics = {
    'rmse': round(np.sqrt((errors ** 2).mean()), 4),
    'mae' : round(np.abs(errors).mean(), 4),
    'mape': round(np.mean(np.abs(errors / (y_test_apartment + 1e-8))) * 100, 4),
    'sd'  : round(errors.std(), 4),
    'r2'  : round(r2_score(y_test_apartment, y_pred_apartment), 4),
    'smape': round(smape(y_test_apartment, y_pred_apartment), 4),
    'test_pred_time_s': round(pred_time, 4),
    'params': best_params
}


# === Sauvegarde ===
model_path   = os.path.join(exp_metrics, 'best_model.pkl')
params_path  = os.path.join(exp_metrics, 'best_model_params.json')
metrics_path = os.path.join(exp_metrics, 'best_model_metrics.json')

best_model.save(model_path)
with open(params_path,  'w') as f: json.dump(best_params, f, indent=2)
with open(metrics_path, 'w') as f: json.dump(final_test_metrics, f, indent=2)

# === Logs finaux
print(f"✅ Modèle sauvegardé :       {model_path}")
print(f"✅ Paramètres sauvegardés :  {params_path}")
print(f"✅ Métriques sauvegardées :  {metrics_path}")


