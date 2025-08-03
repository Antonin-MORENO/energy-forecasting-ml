import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.data_handler import DataHandler
from src.models.cnn_model import CNNModel

# ---------------------- Configuration ----------------------

exp_name = 'cnn_all_lags_experiment_save1'
base_dir = os.path.join('outputs', 'experiments', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir = os.path.join(base_dir, 'figures')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

csv_path = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col = 'datetime'
target_col = 'I014_ND'

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

# ---------------------- Load Data ----------------------

dh = DataHandler(
    csv_path=csv_path,
    date_col=date_col,
    feature_cols=feature_cols,
    target_col=target_col,
    no_scale_cols=no_scale_cols,
    holdout_ratio=None,
    holdout_years=1,
    scaler_type='standard'
)

# For CV (full trainval set, no test split)
X_seq_all, X_stat_all, y_all, _, _, _, _, _, _ = dh.get_sequence_data(val_years=0)

# ---------------------- 5-Fold Cross-Validation ----------------------

n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
cv_metrics = {'rmse': [], 'mae': [], 'mape': [], 'sd': []}

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq_all)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    X_tr_seq, X_val_seq = X_seq_all[train_idx], X_seq_all[val_idx]
    X_tr_stat, X_val_stat = X_stat_all[train_idx], X_stat_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    X_train_dict = {'seq_input': X_tr_seq, 'static_input': X_tr_stat}
    X_val_dict   = {'seq_input': X_val_seq, 'static_input': X_val_stat}

    model = CNNModel({
        'input_shape_seq': X_tr_seq.shape[1:], 
        'input_shape_stat': X_tr_stat.shape[1],
        'filters': 64,
        'kernel_size': 3,
        'pool_size': 2,
        'dense_units': 128,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae'],
        'epochs': 200,
        'batch_size': 32,
        'early_stop_patience': 10,
        'checkpoint_path': None,
        'verbose': 0,
        'scale_y': True
    })

    model.fit(X_train_dict, y_tr, X_val=X_val_dict, y_val=y_val)
    preds = model.predict(X_val_dict)
    errors = preds - y_val

    cv_metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, preds)))
    cv_metrics['mae'].append(mean_absolute_error(y_val, preds))
    cv_metrics['mape'].append(np.mean(np.abs(errors / y_val)) * 100)
    cv_metrics['sd'].append(errors.std())

# Moyennes CV
cv_summary = {
    'cv_rmse': round(np.mean(cv_metrics['rmse']), 3),
    'cv_mae':  round(np.mean(cv_metrics['mae']), 3),
    'cv_mape': round(np.mean(cv_metrics['mape']), 3),
    'cv_sd':   round(np.mean(cv_metrics['sd']), 3)
}
print("\nAverage CV metrics:", cv_summary)

# ---------------------- Final Training & Test Evaluation ----------------------

# Split for final training and test
X_seq_tr, X_stat_tr, y_tr, X_seq_val, X_stat_val, y_val, X_seq_te, X_stat_te, y_te = dh.get_sequence_data(val_years=1)

params = {
    'input_shape_seq': X_seq_tr.shape[1:], 
    'input_shape_stat': X_stat_tr.shape[1],
    'filters': 64,
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': 128,
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae'],
    'epochs': 200,
    'batch_size': 32,
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(base_dir, 'models', 'best_cnn.keras'),
    'verbose': 1,
    'scale_y': True
}

cnn = CNNModel(params)

X_train_dict = {'seq_input': X_seq_tr, 'static_input': X_stat_tr}
X_val_dict   = {'seq_input': X_seq_val, 'static_input': X_stat_val}
X_test_dict  = {'seq_input': X_seq_te,  'static_input': X_stat_te}

# Final training
t0 = time.perf_counter()
history = cnn.fit(X_train_dict, y_tr, X_val=X_val_dict, y_val=y_val)
train_time = time.perf_counter() - t0

# Prediction on test set
t0 = time.perf_counter()
preds = cnn.predict(X_test_dict)
pred_time = time.perf_counter() - t0

errors = preds - y_te
metrics = {
    'rmse': np.sqrt((errors ** 2).mean()),
    'mae' : np.abs(errors).mean(),
    'mape': np.mean(np.abs(errors / y_te)) * 100,
    'sd'  : errors.std(),
    'test_pred_time_s': round(pred_time, 4),
    'train_time_s': round(train_time, 2)
}
metrics.update(cv_summary)

# ---------------------- Save Results ----------------------

with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

scaler_path = os.path.join(base_dir, 'y_scaler.pkl')
joblib.dump(cnn.y_scaler, scaler_path)

# Plot learning curves
plt.figure()
plt.plot(history.history['loss'],    label='train_mse')
plt.plot(history.history['val_loss'],label='val_mse')
plt.title('MSE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mse_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history['mae'],     label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mae_curve.png'))
plt.close()

print(f'\n✅ Training complete. Metrics saved to {metrics_dir}')
