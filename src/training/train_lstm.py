import os
import time
import json
import matplotlib.pyplot as plt
from src.data.data_handler import DataHandler
from src.models.lstm_model import LSTMModel
import numpy as np 
from sklearn.preprocessing import StandardScaler

# Edit experiment settings
exp_name    = 'lstm_all_lags_experiment_bs_32'
base_dir    = os.path.join('outputs','experiments', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir     = os.path.join(base_dir, 'figures')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# # Define data paths and features
csv_path    = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col    = 'datetime'
target_col  = 'I014_ND'

# all your features, including all lag cols + static cols
feature_cols = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]
no_scale_cols = [
    'is_interpolated','is_weekend',
    'hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]

# Load and split the data, holding out the last full year as test
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 1,         # hold out the last full year
    scaler_type   = 'standard'
)


# Retrieve prepared arrays for sequence (CNN/LSTM) models:
#   X_seq_tr, X_stat_tr, y_tr  : train
#   X_seq_val, X_stat_val, y_val : validation (last full year of trainval)
#   X_seq_te, X_stat_te, y_te  : test (holdout_years)
X_seq_tr, X_stat_tr, y_tr, X_seq_val, X_stat_val, y_val, X_seq_te, X_stat_te, y_te = dh.get_sequence_data(val_years=1)

# Standard scale the target variable 
y_scaler   = StandardScaler()
y_tr_z     = y_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
y_val_z    = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
y_te_z     = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

# Configure model hyperparameter 
params = {
    'input_shape_seq': X_seq_tr.shape[1:],
    'input_shape_stat': X_stat_tr.shape[1],
    'lstm_units': 64,
    'dense_units': 128,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 200,
    'batch_size': 32,
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(base_dir, 'checkpoints', 'best_lstm_model.keras'),
    'scale_y': True,
    'verbose': 1
}


# Instantiate the CNN model 
lstm = LSTMModel(params)

# Prepare input dictionaries 
X_train_dict = {"seq_input": X_seq_tr,   "static_input": X_stat_tr}
X_val_dict   = {"seq_input": X_seq_val,  "static_input": X_stat_val}
X_test_dict  = {"seq_input": X_seq_te,   "static_input": X_stat_te}

# Training
t0       = time.perf_counter()
history  = lstm.fit(X_train_dict, y_tr_z, X_val=X_val_dict, y_val=y_val_z)
train_tm = time.perf_counter() - t0

# Hold-out evaluation (inverse transform predictions)
preds_z  = lstm.predict(X_test_dict)
preds    = y_scaler.inverse_transform(preds_z.reshape(-1, 1)).ravel()



# Compute classical error metrics on the unscaled values
errors   = preds - y_te
metrics  = {
    "rmse": np.sqrt((errors ** 2).mean()),
    "mae" : np.abs(errors).mean(),
    "mape": np.mean(np.abs(errors / y_te)) * 100,
    "sd"  : errors.std()
}


# Save hold-out metrics to JSON
with open(os.path.join(metrics_dir, "holdout_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


# Save full training history (loss & metrics per epoch)
with open(os.path.join(metrics_dir, "training_history.json"), "w") as f:
    json.dump(history.history, f, indent=2)

# Plot & save learning curves 
# MSE curve
plt.figure()
plt.plot(history.history["loss"],    label="train_mse")
plt.plot(history.history["val_loss"],label="val_mse")
plt.title("MSE per epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.savefig(os.path.join(fig_dir, "mse_curve.png"))
plt.close()

# MAE curve
plt.figure()
plt.plot(history.history["mae"],     label="train_mae")
plt.plot(history.history["val_mae"], label="val_mae")
plt.title("MAE per epoch")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.savefig(os.path.join(fig_dir, "mae_curve.png"))
plt.close()

print(f"Training done. Hold-out metrics saved to {metrics_dir}")