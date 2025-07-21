import os
import time
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from src.data.data_handler import DataHandler
from src.models.lstm_model       import LSTMModel

# General experiment settings
exp_name    = 'lstm_temp_1_24_lags_pressure_humidity'


base_dir    = os.path.join('outputs', 'experiments_UMass', 'temp', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir     = os.path.join(base_dir, 'figures')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# DataHandler parameters
csv_path      = 'data/processed/UMass Smart Dataset/weather_lag.csv'  
date_col      = 'time'
target_col    = 'temperature'

# Sequence features (lags only)
seq_features  = [
    'temperature_lag_1', 'temperature_lag_24', 'pressure', 'humidity']




no_scale_cols = []  


# Load and chronologically split dataset
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = seq_features,  
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 1,              
    scaler_type   = 'standard'      
)

# Get sequence data: (X_seq, X_stat, y) for train, val, and test
X_seq_tr, X_stat_tr, y_tr, X_seq_val, X_stat_val, y_val, X_seq_te, X_stat_te, y_te = \
    dh.get_sequence_data(val_years=1)



# LSTM model configuration
params = {
    'input_shape_seq': X_seq_tr.shape[1:],
    'input_shape_stat': X_stat_tr.shape[1],
    'lstm_units': 64,
    'dense_units': 128,
    'optimizer': Adam(learning_rate=1e-4),
    'loss': 'mse',
    'metrics': ['mae'], 
    'epochs': 200,
    'batch_size': 32,
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(base_dir, 'checkpoints', 'best_lstm_model.keras'),
    'scale_y': True,
    'verbose': 1,
    'scale_mode': 'standard',  
}


lstm = LSTMModel(params)

# Model training
X_train_dict = {'seq_input': X_seq_tr,   'static_input': X_stat_tr}
X_val_dict   = {'seq_input': X_seq_val,  'static_input': X_stat_val}
print(X_train_dict)

t0      = time.perf_counter()
history = lstm.fit(X_train_dict, y_tr, X_val=X_val_dict, y_val=y_val)
train_t = time.perf_counter() - t0

# Evaluate on test set (hold-out)
X_test_dict = {'seq_input': X_seq_te,  'static_input': X_stat_te}
t0     = time.perf_counter()
preds  = lstm.predict(X_test_dict)
pred_t = time.perf_counter() - t0

# Compute error metrics
errors = preds - y_te
metrics = {
    'rmse'             : np.sqrt((errors**2).mean()),
    'mae'              : np.abs(errors).mean(),
    'mape'             : np.mean(np.abs(errors/y_te))*100,
    'std_error'        : errors.std(),
    'train_time_s'     : round(train_t,3),
    'test_pred_time_s' : round(pred_t,3),
}

# Save metrics and target scaler
with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

scaler_path = os.path.join(metrics_dir, 'y_scaler.pkl')
joblib.dump(lstm.y_scaler, scaler_path)

# Learning curves (MSE and MAE vs Epochs)
plt.figure()
plt.plot(history.history['loss'],    label='train_mse')
plt.plot(history.history['val_loss'],label='val_mse')
plt.title('MSE par epoch')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
plt.savefig(os.path.join(fig_dir, 'mse_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history['mae'],     label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE par epoch')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
plt.savefig(os.path.join(fig_dir, 'mae_curve.png'))
plt.close()

print("✅ Training complete. Artifacts saved in", base_dir)


# Plot: Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(y_te,   label='Actual',   linewidth=1)
plt.plot(preds,  label='Predicted',linewidth=1)
plt.xlabel('Sample index')
plt.ylabel('Temperature')
plt.title('LSTM: Actual vs Predicted Temperature')
plt.legend()
plt.tight_layout()

# Save prediction plot
fig_path = os.path.join(metrics_dir, 'lstm_pred_vs_actual.png')
plt.savefig(fig_path)
plt.close()


print(f"💾 Actual vs Predicted curve (LSTM) saved at: {fig_path}")