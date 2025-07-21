import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.data.data_handler import DataHandler

# Paths to the trained models 
base_xgb = 'outputs/experiments/all_features_std_norm_xgboost_experiment_retrain_on_train_only'
base_rf  = 'outputs/experiments/all_features_std_norm_rf_experiment_retrain_on_train_only'
model_xgb_path = os.path.join(base_xgb, 'xgb_retrained_on_train.pkl')
model_rf_path  = os.path.join(base_rf,  'rf_retrained_on_train.pkl')

# Load the pretrained models
xgb = joblib.load(model_xgb_path)
rf  = joblib.load(model_rf_path)

# Prepare the data (using the same chronological train/val/test split)
dh = DataHandler(
    csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv',
    date_col      = 'datetime',
    feature_cols  = [
        'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
        'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
        'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
        'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
    ],
    target_col    = 'I014_ND',
    no_scale_cols = [
        'is_interpolated','is_weekend',
        'hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
    ],
    holdout_years = 1,
    scaler_type   = 'standard'
)
X_tr, y_tr, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)


# Find the optimal weight for the ensemble (based on validation RMSE)
# Try all weights from 0.0 to 1.0 (step=0.001) for XGBoost in the linear combination with RF
pred_val_xgb = xgb.predict(X_val)
pred_val_rf  = rf.predict(X_val)

ws    = np.linspace(0, 1, 1001)
best  = (0.0, np.inf)
for w in ws:
    comb = w * pred_val_xgb + (1 - w) * pred_val_rf
    rmse = np.sqrt(mean_squared_error(y_val, comb))
    if rmse < best[1]:
        best = (w, rmse)
w_opt, val_rmse = best

# Make predictions on the test set and measure inference time


# XGBoost

t0 = time.perf_counter()
pred_xgb_test = xgb.predict(X_test)
time_xgb = time.perf_counter() - t0

# RF
t0 = time.perf_counter()
pred_rf_test  = rf.predict(X_test)
time_rf = time.perf_counter() - t0

# Ensemble
t0 = time.perf_counter()
pred_ens_test = w_opt * pred_xgb_test + (1 - w_opt) * pred_rf_test
time_comb = time.perf_counter() - t0

# Total time for ensemble prediction includes both individual model times
time_ens_total = time_xgb + time_rf + time_comb


# Define evaluation function and compute performance metrics
def calc_metrics(y_true, y_pred, inf_time):
    errors = y_true - y_pred
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs(errors / y_true)) * 100
    r2    = r2_score(y_true, y_pred)
    sd    = errors.std()
    return {
        'rmse':             round(rmse,4),
        'mae':              round(mae,4),
        'mape':             round(mape,2),
        'r2':               round(r2,4),
        'sd':               round(sd,4),
        'inference_time_s': round(inf_time,4)
    }

# Compute metrics for each model
metrics_xgb = calc_metrics(y_test, pred_xgb_test, time_xgb)
metrics_rf  = calc_metrics(y_test, pred_rf_test,  time_rf)
metrics_ens = calc_metrics(y_test, pred_ens_test, time_ens_total)


# Print test performance to console
print("=== Test metrics ===")
print("XGBoost standalone:", metrics_xgb)
print("RF    standalone:", metrics_rf)
print("Ensemble        :", metrics_ens)

# Save results to JSON file
out_dir = 'outputs/evaluations/ensemble_xgb_rf'
os.makedirs(out_dir, exist_ok=True)

results = {
    'weight_xgb':    round(w_opt,3),
    'weight_rf':     round(1-w_opt,3),
    'val_rmse':      round(val_rmse,4),
    'test_metrics': {
        'xgb': metrics_xgb,
        'rf':  metrics_rf,
        'ens': metrics_ens
    }
}

with open(os.path.join(out_dir, 'ensemble_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Résultats enregistrés dans {out_dir}/ensemble_results.json")
