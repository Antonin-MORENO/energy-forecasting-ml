import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib

from src.data.data_handler import DataHandler
from src.models.autoencoder_model import AutoEncoderModel
from src.models.lstm_model import LSTMModel

# === Experiment configuration ===
exp_name       = 'lstm_with_ae_static_8'
base_dir       = os.path.join('outputs', 'experiments', exp_name)
metrics_dir    = os.path.join(base_dir, 'metrics')
fig_dir        = os.path.join(base_dir, 'figures')
ckpt_dir       = os.path.join(base_dir, 'checkpoints')

# Path to the pre-trained static AutoEncoder model
ae_model_path  = os.path.join(
    'outputs', 'experiments', 'ae_smooth_funnel_8_stat',
    'checkpoints', 'ae_static.keras'
)

# Create necessary directories if they don't exist
for d in (metrics_dir, fig_dir, ckpt_dir):
    os.makedirs(d, exist_ok=True)

# === Data loading and preparation ===
csv_path    = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col    = 'datetime'
target_col  = 'I014_ND'

# Full list of input features
feature_cols = [
    'NON_BM_STOR', 'I014_PUMP_STORAGE_PUMPING', 'is_interpolated', 'is_weekend',
    'I014_ND_lag_1', 'I014_ND_lag_2', 'I014_ND_lag_48', 'I014_ND_lag_96', 'I014_ND_lag_336',
    'I014_ND_mean_48', 'I014_ND_mean_336', 'net_import', 'wind_capacity_factor',
    'solar_capacity_factor', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
    'month_sin', 'month_cos'
]

# Features that should not be scaled
no_scale_cols = [
    'is_interpolated', 'is_weekend',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]

# Initialize DataHandler for loading, splitting, and scaling data
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_years = 1,
    scaler_type   = 'minmax'
)

# Retrieve train/validation/test data split for LSTM input
X_seq_tr, X_stat_tr, y_tr, \
X_seq_val, X_stat_val, y_val, \
X_seq_te, X_stat_te, y_te = dh.get_sequence_data(val_years=1)

# === Load the pre-trained AutoEncoder for static features ===
ae_params = {
    'input_shape': X_stat_tr.shape[1],
    'optimizer' : 'adam',
    'loss'      : 'mse',
    'metrics'   : ['mae']
}
ae = AutoEncoderModel(ae_params)
ae.autoencoder.load_weights(ae_model_path)

# Encode the static features using the encoder part of the AE
Z_tr  = ae.encode(X_stat_tr)
Z_val = ae.encode(X_stat_val)
Z_te  = ae.encode(X_stat_te)

# === Define and initialize the LSTM model ===
lstm_params = {
    'input_shape_seq': X_seq_tr.shape[1:],   # Shape of the sequential input (timesteps, features)
    'input_shape_stat': Z_tr.shape[1],       # Dimension of the static AE code
    'lstm_units': 64,
    'dense_units': 256,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 200,
    'batch_size': 32,
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(ckpt_dir, 'best_lstm_model.keras'),
    'scale_y': True,
    'verbose': 1,
    'scale_mode': 'standard'
}

lstm = LSTMModel(lstm_params)

# Input dictionaries for the model (sequence and static inputs)
X_train = {'seq_input': X_seq_tr, 'static_input': Z_tr}
X_val   = {'seq_input': X_seq_val, 'static_input': Z_val}
X_test  = {'seq_input': X_seq_te,  'static_input': Z_te}

# === Train the LSTM model ===
print('Training LSTM with static embeddings from AE...')
t0 = time.perf_counter()
history = lstm.fit(X_train, y_tr, X_val=X_val, y_val=y_val)
train_time = time.perf_counter() - t0
print(f'Training completed in {train_time:.2f} seconds')

# === Evaluate on the test set ===
t1 = time.perf_counter()
preds = lstm.predict(X_test)
pred_time = time.perf_counter() - t1

# Compute metrics (on de-scaled target)
errors = preds - y_te
metrics = {
    'train_time_s': train_time,
    'test_pred_time_s': pred_time,
    'rmse':  float(np.sqrt(np.mean(errors**2))),
    'mae':   float(np.mean(np.abs(errors))),
    'mape':  float(np.mean(np.abs(errors / y_te)) * 100),
    'sd':    float(np.std(errors))
}

# Save the metrics to a JSON file
with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# Save training history
with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

# Save the scaler for the target variable
joblib.dump(lstm.y_scaler, os.path.join(base_dir, 'y_scaler.pkl'))

# === Plot training curves (MSE and MAE) ===
plt.figure()
plt.plot(history.history['loss'],     label='train_mse')
plt.plot(history.history['val_loss'], label='val_mse')
plt.title('LSTM – MSE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mse_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history['mae'],      label='train_mae')
plt.plot(history.history['val_mae'],  label='val_mae')
plt.title('LSTM – MAE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mae_curve.png'))
plt.close()

print(f'Training complete. All outputs saved under: {base_dir}')
