import os
import json
import numpy as np
import joblib
import tensorflow as tf

from src.data.data_handler import DataHandler
from src.models.lstm_quantile_model import LSTMQuantileModel

# --- 1) Helpers pour la pinball loss ----------------------------------------
def pinball_loss_numpy(y_true, y_pred, q):
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))

# --- 2) Paths et paramètres -------------------------------------------------
xgb_model_dir    = 'outputs/experiments/xgboost_quantile_prediction_standard_200_nestimators_99_quantiles_nestimators_1000_3000/models'
path_xgb_lower   = os.path.join(xgb_model_dir, 'best_model_lower.pkl')
path_xgb_median  = os.path.join(xgb_model_dir, 'best_model_median.pkl')
path_xgb_upper   = os.path.join(xgb_model_dir, 'best_model_upper.pkl')

lstm_base_dir    = 'outputs/experiments/lstm_quantile_experiment_bs_32_bidirectional_64_q01-50-99_minmax_save'
path_lstm_ckpt   = os.path.join(lstm_base_dir, 'checkpoints', 'best_lstm_quantile_model.keras')
path_lstm_scaler = os.path.join(lstm_base_dir, 'y_scaler.pkl')

out_dir        = 'outputs/evaluations/ensemble_model'
os.makedirs(out_dir, exist_ok=True)
weights_file   = os.path.join(out_dir, 'ensemble_weights.json')
metrics_file   = os.path.join(out_dir, 'test_metrics.json')

csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
target_col    = 'I014_ND'
feature_cols  = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos',
    'month_sin','month_cos'
]
no_scale_cols = [
    'is_interpolated','is_weekend',
    'hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]

QUANTILE_MAP = {0.01: 'lower', 0.5: 'median', 0.99: 'upper'}
QUANTILES    = list(QUANTILE_MAP.keys())

# --- 3) Charger modèles XGB & LSTMQuantile -------------------------------
xgb_lower = joblib.load(path_xgb_lower)
xgb_med   = joblib.load(path_xgb_median)
xgb_upper = joblib.load(path_xgb_upper)

dh_tmp = DataHandler(
    csv_path=csv_path, date_col=date_col,
    feature_cols=feature_cols, target_col=target_col,
    no_scale_cols=no_scale_cols,
    holdout_years=1, scaler_type='minmax'
)
_, _, _, X_seq_val, X_stat_val, _, X_seq_te, X_stat_te, y_test = \
    dh_tmp.get_sequence_data(val_years=1)

lstm_params = {
    'input_shape_seq':  X_seq_val.shape[1:],
    'input_shape_stat': X_stat_val.shape[1],
    'lstm_units':       64,
    'dense_units':      256,
    'quantiles':        QUANTILES
}
lstm_inst  = LSTMQuantileModel(params=lstm_params)
lstm_model = lstm_inst.model
lstm_model.load_weights(path_lstm_ckpt)
y_scaler   = joblib.load(path_lstm_scaler)

# --- 4) Préparer X_val & X_test pour XGB ----------------------------------
dh_xgb = DataHandler(
    csv_path=csv_path, date_col=date_col,
    feature_cols=feature_cols, target_col=target_col,
    no_scale_cols=no_scale_cols,
    holdout_years=1, scaler_type='standard'
)
_, _, X_val_xgb, y_val, X_test_xgb, y_test = \
    dh_xgb.get_train_val_test_split(val_years=1)

# --- 5) Générer prédictions VALIDATION & trouver poids -------------------
preds_xgb_val = {
    'lower':  xgb_lower.predict(X_val_xgb),
    'median': xgb_med.predict(X_val_xgb),
    'upper':  xgb_upper.predict(X_val_xgb)
}
scaled_val_lstm = lstm_model.predict({'seq_input':X_seq_val,'static_input':X_stat_val})
preds_lstm_val = {
    'lower':  y_scaler.inverse_transform(scaled_val_lstm[0].reshape(-1,1)).ravel(),
    'median': y_scaler.inverse_transform(scaled_val_lstm[1].reshape(-1,1)).ravel(),
    'upper':  y_scaler.inverse_transform(scaled_val_lstm[2].reshape(-1,1)).ravel()
}

ensemble_weights = {}
for q, key in QUANTILE_MAP.items():
    best_w, best_loss = 0.0, float('inf')
    px = preds_xgb_val[key]
    pl = preds_lstm_val[key]
    for w in np.linspace(0,1,1001):
        comb = w*pl + (1-w)*px
        loss = pinball_loss_numpy(y_val, comb, q)
        if loss < best_loss:
            best_loss, best_w = loss, w
    ensemble_weights[key] = {'lstm': float(best_w), 'xgb': float(1-best_w)}

with open(weights_file,'w') as f:
    json.dump(ensemble_weights, f, indent=2)

# --- 6) Générer prédictions TEST pour chaque modèle & ensemble ----------
test_preds = {'xgb':{}, 'lstm':{}, 'ensemble':{}}

test_preds['xgb'] = {
    'lower':  xgb_lower.predict(X_test_xgb),
    'median': xgb_med.predict(X_test_xgb),
    'upper':  xgb_upper.predict(X_test_xgb)
}

scaled_te = lstm_model.predict({'seq_input':X_seq_te,'static_input':X_stat_te})
test_preds['lstm'] = {
    'lower':  y_scaler.inverse_transform(scaled_te[0].reshape(-1,1)).ravel(),
    'median': y_scaler.inverse_transform(scaled_te[1].reshape(-1,1)).ravel(),
    'upper':  y_scaler.inverse_transform(scaled_te[2].reshape(-1,1)).ravel()
}

for key in QUANTILE_MAP.values():
    w = ensemble_weights[key]['lstm']
    test_preds['ensemble'][key] = (
        w * test_preds['lstm'][key] + (1-w)*test_preds['xgb'][key]
    )

# --- 7) Calculer métriques TEST --------------------------------------------
results = {}
for model, preds in test_preds.items():
    y_hat = preds['median']
    rmse = float(np.sqrt(np.mean((y_test - y_hat)**2)))
    mae  = float(np.mean(np.abs(y_test - y_hat)))
    mape = float(np.mean(np.abs((y_test - y_hat)/y_test)) * 100)

    lower, upper = preds['lower'], preds['upper']
    picp = float(np.mean((y_test>=lower)&(y_test<=upper)) * 100)
    mpiw = float(np.mean(upper - lower))

    pl_low  = pinball_loss_numpy(y_test, lower, 0.01)
    pl_med  = pinball_loss_numpy(y_test, y_hat, 0.5)
    pl_high = pinball_loss_numpy(y_test, upper, 0.99)

    results[model] = {
        'point_prediction_metrics': {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        },
        'prediction_interval_metrics': {
            'picp': picp,
            'mpiw': mpiw
        },
        'pinball_loss': {
            'lower_lower':  pl_low,
            'median_median': pl_med,
            'upper_upper':  pl_high
        }
    }

with open(metrics_file,'w') as f:
    json.dump(results, f, indent=2)

print("Ensemble weights:", ensemble_weights)
print("Test metrics:", results)
