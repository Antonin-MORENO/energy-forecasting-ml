import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.data.data_handler import DataHandler

# Paths to pretrained models
base_xgb  = 'outputs/experiments/all_features_std_norm_xgboost_experiment_retrain_on_train_only'
base_svm  = 'outputs/experiments/retrain_on_train_only'
xgb_path  = os.path.join(base_xgb, 'xgb_retrained_on_train.pkl')
svm_path  = os.path.join(base_svm, 'svm_retrained_on_train.pkl')

# Load pretrained XGBoost and SVM models
xgb = joblib.load(xgb_path)
svm = joblib.load(svm_path)

# Prepare train / validation / test sets using chronological split
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
X_train, y_train, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)

# Grid search to find best linear weight (w_svm) for ensemble on validation set
# The ensemble combines SVM and XGBoost predictions with weights w and (1 - w)
pred_val_xgb = xgb.predict(X_val)
pred_val_svm = svm.predict(X_val)

ws   = np.linspace(0, 1, 1001)
best = (0.0, np.inf)
for w in ws:
    comb = w * pred_val_svm + (1 - w) * pred_val_xgb
    rmse = np.sqrt(mean_squared_error(y_val, comb))
    if rmse < best[1]:
        best = (w, rmse)
w_svm, val_rmse = best

# Inference on test set and time measurements


# XGBoost
t0     = time.perf_counter()
pred_xgb_test = xgb.predict(X_test)
time_xgb = time.perf_counter() - t0

# SVM
t0     = time.perf_counter()
pred_svm_test = svm.predict(X_test)
time_svm = time.perf_counter() - t0

# Ensemble prediction
t0              = time.perf_counter()
pred_ens_test   = w_svm * pred_svm_test + (1 - w_svm) * pred_xgb_test
time_comb       = time.perf_counter() - t0


# Total inference time for ensemble model
time_ens_total = time_xgb + time_svm + time_comb


def calc_metrics(y_true, y_pred, inf_time):
    errs = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(errs / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    sd   = errs.std()
    return {
        'rmse':             round(rmse,4),
        'mae':              round(mae,4),
        'mape':             round(mape,2),
        'r2':               round(r2,4),
        'sd':               round(sd,4),
        'inference_time_s': round(inf_time,4)
    }
    
    
# Compute test metrics for each model
metrics_xgb = calc_metrics(y_test, pred_xgb_test, time_xgb)
metrics_svm = calc_metrics(y_test, pred_svm_test, time_svm)
metrics_ens = calc_metrics(y_test, pred_ens_test, time_ens_total)


print("=== Test metrics ===")
print("XGBoost standalone:", metrics_xgb)
print("SVM     standalone:", metrics_svm)
print("Ensemble          :", metrics_ens)

out_dir = 'outputs/evaluations/ensemble_xgb_svm'
os.makedirs(out_dir, exist_ok=True)

results = {
    'weight_svm':   round(w_svm,3),
    'weight_xgb':   round(1-w_svm,3),
    'val_rmse':     round(val_rmse,4),
    'test_metrics': {
        'xgb': metrics_xgb,
        'svm': metrics_svm,
        'ens': metrics_ens
    }
}

with open(os.path.join(out_dir, 'ensemble_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Résultats enregistrés dans {out_dir}/ensemble_results.json")
