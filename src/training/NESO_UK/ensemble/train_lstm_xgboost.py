import os
import json
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from src.data.data_handler import DataHandler
from src.models.lstm_model import LSTMModel


# Paths to trained models
# Define paths to previously retrained XGBoost and LSTM models
base_xgb  = 'outputs/experiments/all_features_std_norm_xgboost_experiment_retrain_on_train_only'
xgb_path  = os.path.join(base_xgb, 'xgb_retrained_on_train.pkl')

lstm_exp  = 'lstm_all_lags_experiment_bs_32_bidirectional_256_minmax_save'
lstm_base = os.path.join('outputs', 'experiments', lstm_exp)
lstm_ckpt = os.path.join(lstm_base, 'checkpoints', 'best_lstm_model.keras')
lstm_scal = os.path.join(lstm_base, 'y_scaler.pkl')

# Load XGBoost and LSTM models
xgb = joblib.load(xgb_path)


# Load LSTM input data using min-max scaling
dh_lstm = DataHandler(
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


# Load LSTM sequence and static data for training, validation, and test sets
X_seq_tr, X_stat_tr, y_tr_seq, \
X_seq_val, X_stat_val, y_val_seq, \
X_seq_te,  X_stat_te,  y_test_seq = dh_lstm.get_sequence_data(val_years=1)


# Initialize the LSTM model and load weights and y-scaler
lstm_params = {
    'input_shape_seq':     X_seq_tr.shape[1:],
    'input_shape_stat':    X_stat_tr.shape[1],
    'lstm_units':          64,
    'dense_units':         256,
    'optimizer':           'adam',
    'loss':                'mse',
    'metrics':             ['mae'],
    'batch_size':          32,
    'epochs':              200,
    'early_stop_patience': 10,
    'checkpoint_path':     lstm_ckpt,
    'scale_y':             True,
    'scale_mode':          'standard',
    'verbose':             0
}
lstm = LSTMModel(lstm_params)
lstm.model.load_weights(lstm_ckpt)
lstm.y_scaler = joblib.load(lstm_scal)

# Load data for XGBoost (with standard scaling)
# This ensures the XGBoost model receives input formatted as it was trained on
dh_std = DataHandler(
    csv_path      = dh_lstm.csv_path,
    date_col      = dh_lstm.date_col,
    feature_cols  = dh_lstm.feature_cols,
    target_col    = dh_lstm.target_col,
    no_scale_cols = dh_lstm.no_scale_cols,
    holdout_years = 1,
    scaler_type   = 'standard'
)

# Get train, validation, and test splits
X_train_std, y_train_std, X_val_std, y_val_std, X_test_std, y_test_std = \
    dh_std.get_train_val_test_split(val_years=1)

# Generate predictions on the validation set 
pred_val_xgb   = xgb.predict(X_val_std)
pred_val_lstm = lstm.predict({
    'seq_input':    X_seq_val,
    'static_input': X_stat_val
}).ravel()

# Rescale predictions from LSTM
pred_val_lstm = lstm.y_scaler.inverse_transform(pred_val_lstm.reshape(-1,1)).ravel()



# Find optimal weight for LSTM in the ensemble 
# Combine LSTM and XGBoost predictions with weights from 0 to 1
# Use validation RMSE to find best weight
ws   = np.linspace(0, 1, 1001)
best = (0.0, np.inf)
for w in ws:
    comb = w * pred_val_lstm + (1 - w) * pred_val_xgb
    rmse = np.sqrt(mean_squared_error(y_val_std, comb))
    if rmse < best[1]:
        best = (w, rmse)
w_lstm, val_rmse = best

# Predict on test set and measure inference time


# XGBoost
t0 = time.perf_counter()
pred_xgb_test = xgb.predict(X_test_std)
time_xgb = time.perf_counter() - t0

# LSTM
t0 = time.perf_counter()
pred_lstm_test = lstm.predict({
    'seq_input':    X_seq_te,
    'static_input': X_stat_te
}).ravel()
pred_lstm_test = lstm.y_scaler.inverse_transform(pred_lstm_test.reshape(-1,1)).ravel()
time_lstm = time.perf_counter() - t0

# Ensemble
t0 = time.perf_counter()
pred_ens_test = w_lstm * pred_lstm_test + (1 - w_lstm) * pred_xgb_test
time_comb = time.perf_counter() - t0

time_ens_total = time_xgb + time_lstm + time_comb

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

metrics = {
    'xgb':  calc_metrics(y_test_std, pred_xgb_test, time_xgb),
    'lstm': calc_metrics(y_test_std, pred_lstm_test, time_lstm),
    'ens':  calc_metrics(y_test_std, pred_ens_test, time_ens_total)
}


# Print out evaluation metrics
print("=== Test metrics ===")
print("XGBoost standalone:", metrics['xgb'])
print("LSTM standalone:   ", metrics['lstm'])
print("Ensemble:          ", metrics['ens'])

# Save results to disk 
# Store ensemble weights, validation RMSE, and test metrics
out = {
    'weight_lstm':    round(w_lstm,3),
    'weight_xgb':     round(1-w_lstm,3),
    'val_rmse':       round(val_rmse,4),
    'test_metrics':   metrics
}
out_dir = 'outputs/evaluations/ensemble_lstm_xgb'
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'ensemble_results.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f"→ Ensemble LSTM+XGBoost enregistré dans {out_dir}")
