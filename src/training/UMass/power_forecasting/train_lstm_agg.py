import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from src.data.data_handler import DataHandler
from src.models.lstm_model import LSTMModel

# === EXPÉRIMENT SETUP ===
exp_name    = 'lstm_smartumass_multiappart_lag_temp_cyclical'
base_dir    = os.path.join('outputs','experiments_UMass', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir     = os.path.join(base_dir, 'figures')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# === Données ===
csv_path_train = 'data/processed/UMass Smart Dataset/multi_apartment_train.csv'
csv_path_test  = 'data/processed/UMass Smart Dataset/test_apartment.csv'

# === Colonnes ===
date_col      = 'time'
target_col    = 'power [kW]'
feature_cols  = [
    'power [kW]_lag_1',  'power [kW]_lag_24',  'power [kW]_lag_168', 'temperature',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]
no_scale_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']

# === DataHandler configuration ===
dh = DataHandler(
    csv_path      = csv_path_train,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = 0,         # <— Pas de holdout ici
    holdout_years = None,
    scaler_type   = 'standard'
)

# Charger et scaler tout le train set
df_full = dh.load_data()
X_trval, y_trval, _, _ = dh.scale_split(df_full, df_full)

# Créer les séquences
X_seq, X_stat, y_seq = dh.create_sequences(df_full)

# Séparer validation manuellement (ex. 3 mois)
val_months = 12
samples_per_month = 30 * 24
n_val = val_months * samples_per_month

X_seq_tr = X_seq[:-n_val]
X_seq_val = X_seq[-n_val:]
if X_stat is not None:
    X_stat_tr = X_stat[:-n_val]
    X_stat_val = X_stat[-n_val:]
else:
    X_stat_tr = None
    X_stat_val = None

y_tr = y_seq[:-n_val]
y_val = y_seq[-n_val:]

X_train_dict = {'seq_input': X_seq_tr, 'static_input': X_stat_tr}
X_val_dict   = {'seq_input': X_seq_val, 'static_input': X_stat_val}

# === LSTM hyperparameters ===
params = {
    'input_shape_seq':   X_seq_tr.shape[1:],  # (timesteps, 1)
    'input_shape_stat':  X_stat_tr.shape[1] if X_stat_tr is not None else 0,  # nb static features
    'lstm_units':        64,
    'dense_units':       256,
    'optimizer':         Adam(learning_rate=5e-5),
    'loss':              'mse',
    'metrics':           ['mae'],
    'epochs':            200,
    'batch_size':        32,
    'early_stop_patience': 10,
    'checkpoint_path':   os.path.join(base_dir, 'checkpoints', 'best_lstm_model.keras'),
    'scale_y':           True,
    'verbose':           1,
    'scale_mode':        'standard'
}

# === Entraînement ===
lstm = LSTMModel(params)
t0 = time.perf_counter()
history = lstm.fit(X_train_dict, y_tr, X_val=X_val_dict, y_val=y_val)
train_time = time.perf_counter() - t0

# === Préparer test set externe ===
df_test = pd.read_csv(csv_path_test, parse_dates=['time'])
X_test_ap = df_test[feature_cols].copy()
y_test_ap = df_test[target_col].values

# Apply same scaling
cols_to_scale = [c for c in feature_cols if c not in no_scale_cols]
X_test_ap[cols_to_scale] = dh.scaler.transform(X_test_ap[cols_to_scale])

# Créer les séquences test
df_test_scaled = X_test_ap.copy()
df_test_scaled[target_col] = y_test_ap
X_seq_te, X_stat_te, y_te = dh.create_sequences(df_test_scaled)
X_test_dict = {'seq_input': X_seq_te, 'static_input': X_stat_te}


print("\n=== Taille et date max par split ===")
print(f"Train : {X_seq_tr.shape[0]} séquences | max date : {df_full['time'].iloc[-(n_val + 1)]}")
print(f"Val   : {X_seq_val.shape[0]} séquences | max date : {df_full['time'].iloc[-1]}")
print(f"Test  : {X_seq_te.shape[0]} séquences | max date : {df_test['time'].iloc[-1]}")

# === Prédiction ===
t0 = time.perf_counter()
y_pred = lstm.predict(X_test_dict)
pred_time = time.perf_counter() - t0

# === Si target était scalée → inverse_transform


# === Erreurs & métriques
errors = y_pred - y_te
metrics = {
    'rmse': np.sqrt((errors ** 2).mean()),
    'mae' : np.abs(errors).mean(),
    'mape': np.mean(np.abs(errors / y_te)) * 100,
    'sd'  : errors.std(),
    'test_pred_time_s': round(pred_time, 4)
}

# === Sauvegardes
with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

scaler_y_path = os.path.join(base_dir, 'y_scaler.pkl')
scaler_x_path = os.path.join(base_dir, 'x_scaler.pkl')

joblib.dump(lstm.y_scaler, scaler_y_path)
joblib.dump(dh.scaler, scaler_x_path)

print(f"✅ Scaler Y sauvegardé : {scaler_y_path}")
print(f"✅ Scaler X sauvegardé : {scaler_x_path}")



# === Courbes de loss
plt.figure()
plt.plot(history.history['loss'], label='train_mse')
plt.plot(history.history['val_loss'], label='val_mse')
plt.title('MSE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mse_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mae_curve.png'))
plt.close()

print(f"✅ Entraînement terminé. Résultats et courbes sauvegardés dans {metrics_dir}")
