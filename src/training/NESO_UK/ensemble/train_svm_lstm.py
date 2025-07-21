import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.data.data_handler import DataHandler
from src.models.svm_model import SVMModel
from src.models.lstm_model import LSTMModel

# Paths to pretrained models
base_svm = 'outputs/experiments/retrain_on_train_only'
svm_path = os.path.join(base_svm, 'svm_retrained_on_train.pkl')

lstm_exp    = 'lstm_all_lags_experiment_bs_32_bidirectional_256_minmax_save'
lstm_base   = os.path.join('outputs', 'experiments', lstm_exp)
lstm_ckpt   = os.path.join(lstm_base, 'checkpoints', 'best_lstm_model.keras')
lstm_scaler = os.path.join(lstm_base, 'y_scaler.pkl')

# Load pretrained SVM and instantiate LSTM
svm = joblib.load(svm_path)

# Create temporary DataHandler instance for loading data for LSTM
# with the same scaling used during training (min-max)
dh_tmp = DataHandler(
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
    scaler_type   = 'minmax'
)

# Load LSTM sequence/static data for val/test to match training configuration
_, _, _, X_seq_val, X_stat_val, _, X_seq_te, X_stat_te, y_test_seq = \
    dh_tmp.get_sequence_data(val_years=1)


# Define and load the LSTM model (same parameters as used during training)
lstm_params = {
    'input_shape_seq':  X_seq_val.shape[1:],
    'input_shape_stat': X_stat_val.shape[1],
    'lstm_units':       64,
    'dense_units':      256,
    'optimizer':        'adam',
    'loss':             'mse',
    'metrics':          ['mae'],
    'epochs':           200,
    'batch_size':       32,
    'early_stop_patience': 10,
    'checkpoint_path':  lstm_ckpt,
    'scale_y':          True,
    'scale_mode':       'standard',
    'verbose':          0
}
lstm = LSTMModel(lstm_params)
lstm.model.load_weights(lstm_ckpt)
lstm.y_scaler = joblib.load(lstm_scaler)

# Prepare standard input data for SVM and compute ensemble weights
dh_std = DataHandler(
    csv_path      = dh_tmp.csv_path,
    date_col      = dh_tmp.date_col,
    feature_cols  = dh_tmp.feature_cols,
    target_col    = dh_tmp.target_col,
    no_scale_cols = dh_tmp.no_scale_cols,
    holdout_years = 1,
    scaler_type   = 'standard'
)
X_train, y_train, X_val, y_val, X_test, y_test = \
    dh_std.get_train_val_test_split(val_years=1)

# Find optimal linear weight for the SVM-LSTM ensemble on validation set ===
pred_val_svm  = svm.predict(X_val)
pred_val_lstm = lstm.predict({'seq_input': X_seq_val, 'static_input': X_stat_val}).ravel()
pred_val_lstm = lstm.y_scaler.inverse_transform(pred_val_lstm.reshape(-1,1)).ravel()

# Perform grid search over weights [0, 1] for the SVM component
ws   = np.linspace(0, 1, 1001)
best = (0.0, np.inf)
for w in ws:
    comb = w * pred_val_svm + (1 - w) * pred_val_lstm
    rmse = np.sqrt(mean_squared_error(y_val, comb))
    if rmse < best[1]:
        best = (w, rmse)
w_svm, val_rmse = best


# Predict on test set and measure inference time 
# SVM
t0 = time.perf_counter()
pred_svm_test = svm.predict(X_test)
time_svm = time.perf_counter() - t0

# LSTM
t0 = time.perf_counter()
pred_lstm_test = lstm.predict({'seq_input': X_seq_te, 'static_input': X_stat_te}).ravel()
# inverse scaling
pred_lstm_test = lstm.y_scaler.inverse_transform(pred_lstm_test.reshape(-1,1)).ravel()
time_lstm = time.perf_counter() - t0

# Ensemble
t0 = time.perf_counter()
pred_ens_test = w_svm * pred_svm_test + (1 - w_svm) * pred_lstm_test
time_comb     = time.perf_counter() - t0

time_ens_total = time_svm + time_lstm + time_comb

# Compute evaluation metrics
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


# Compute metrics for SVM, LSTM, and Ensemble
metrics_svm  = calc_metrics(y_test, pred_svm_test, time_svm)
metrics_lstm = calc_metrics(y_test, pred_lstm_test, time_lstm)
metrics_ens  = calc_metrics(y_test, pred_ens_test, time_ens_total)

# Print results
print("=== Test metrics ===")
print("SVM   standalone:", metrics_svm)
print("LSTM  standalone:", metrics_lstm)
print("Ensemble      :", metrics_ens)

# Save results to disk 
out_dir = 'outputs/evaluations/ensemble_svm_lstm'
os.makedirs(out_dir, exist_ok=True)

results = {
    'weight_svm':   round(w_svm,3),
    'weight_lstm':  round(1-w_svm,3),
    'val_rmse':     round(val_rmse,4),
    'test_metrics': {
        'svm':  metrics_svm,
        'lstm': metrics_lstm,
        'ens':  metrics_ens
    }
}
with open(os.path.join(out_dir, 'ensemble_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Résultats enregistrés dans {out_dir}/ensemble_results.json")
