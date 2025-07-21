import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.data.data_handler import DataHandler
from src.models.svm_model import SVMModel
from src.models.rf_model import RandomForestModel


# Paths to pretrained models
base_rf   = 'outputs/experiments/all_features_std_norm_rf_experiment_retrain_on_train_only'
base_svm  = 'outputs/experiments/retrain_on_train_only'
rf_path   = os.path.join(base_rf, 'rf_retrained_on_train.pkl')
svm_path  = os.path.join(base_svm, 'svm_retrained_on_train.pkl')

# Load pretrained RF and SVM models
rf  = joblib.load(rf_path)
svm = joblib.load(svm_path)

# Prepare train/val/test data using chronological split (1 year for test, 1 for val)
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

# Grid search to find optimal linear weight (w_svm) for SVM-RF ensemble based on validation RMSE
pred_val_rf  = rf.predict(X_val)
pred_val_svm = svm.predict(X_val)

ws   = np.linspace(0, 1, 1001)
best = (0.0, np.inf)
for w in ws:
    comb = w * pred_val_svm + (1 - w) * pred_val_rf
    rmse = np.sqrt(mean_squared_error(y_val, comb))
    if rmse < best[1]:
        best = (w, rmse)
w_svm, val_rmse = best


# Generate predictions on test set and measure inference times


# RF
t0 = time.perf_counter()
pred_rf_test = rf.predict(X_test)
time_rf = time.perf_counter() - t0

# SVM
t0 = time.perf_counter()
pred_svm_test = svm.predict(X_test)
time_svm = time.perf_counter() - t0

# Ensemble
t0 = time.perf_counter()
pred_ens_test = w_svm * pred_svm_test + (1 - w_svm) * pred_rf_test
time_comb = time.perf_counter() - t0



# Total time for ensemble prediction
time_ens_total = time_rf + time_svm + time_comb

# Define evaluation function and compute metrics for each model
def calc_metrics(y_true, y_pred, inf_time):
    errs = y_true - y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(errs / y_true)) * 100
    r2   = r2_score(y_true, y_pred)
    sd   = errs.std()
    return {
        'rmse':               round(rmse,4),
        'mae':                round(mae,4),
        'mape':               round(mape,2),
        'r2':                 round(r2,4),
        'sd':                 round(sd,4),
        'inference_time_s':   round(inf_time,4)
    }

# Calculate test metrics
metrics_rf  = calc_metrics(y_test, pred_rf_test, time_rf)
metrics_svm = calc_metrics(y_test, pred_svm_test, time_svm)
metrics_ens = calc_metrics(y_test, pred_ens_test, time_ens_total)

# Print evaluation results
print("=== Test metrics ===")
print("RF   standalone:", metrics_rf)
print("SVM  standalone:", metrics_svm)
print("Ensemble      :", metrics_ens)

# Save ensemble results to disk
out_dir = 'outputs/evaluations/ensemble_rf_svm'
os.makedirs(out_dir, exist_ok=True)

results = {
    'weight_svm':   round(w_svm,3),
    'weight_rf':    round(1-w_svm,3),
    'val_rmse':     round(val_rmse,4),
    'test_metrics': {
        'rf':   metrics_rf,
        'svm':  metrics_svm,
        'ens':  metrics_ens
    }
}

with open(os.path.join(out_dir, 'ensemble_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Résultats enregistrés dans {out_dir}/ensemble_results.json")
