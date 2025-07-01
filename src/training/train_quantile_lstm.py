
import os
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from src.data.data_handler import DataHandler
from src.models.lstm_quantile_model import LSTMQuantileModel
import numpy as np

# --- 1. Experiment Setup ---
# Define a unique name for the experiment to organize outputs
exp_name    = 'lstm_quantile_experiment_bs_32_bidirectional_64_q01-50-99_minmax'

# Define base directories for outputs
base_dir    = os.path.join('outputs', 'experiments', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir     = os.path.join(base_dir, 'figures')
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# --- 2. Data Paths and Column Definitions ---
csv_path    = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col    = 'datetime'
target_col  = 'I014_ND'

# Define feature columns to be used by the model
feature_cols = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]

# Specify columns that should not be scaled (e.g., one-hot encoded or cyclical features)
no_scale_cols = [
    'is_interpolated','is_weekend',
    'hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]

# --- 3. Data Loading and Splitting ---
# Initialize the data handler which manages data loading, splitting, and scaling
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_years = 1,
    scaler_type   = 'minmax',
)

# Generate sequence data for the LSTM model
X_seq_tr, X_stat_tr, y_tr, X_seq_val, X_stat_val, y_val, X_seq_te, X_stat_te, y_te = dh.get_sequence_data(val_years=1)

# --- 4. Model Hyperparameters ---
params = {
    'input_shape_seq': X_seq_tr.shape[1:],
    'input_shape_stat': X_stat_tr.shape[1],
    'lstm_units': 64,
    'dense_units': 256,
    'optimizer': 'adam',
    'quantiles': [0.01, 0.5, 0.99],
    'scale_mode': 'standard',  
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(base_dir, 'checkpoints', 'best_lstm_quantile_model.keras'),
    'epochs': 200,
    'batch_size': 32,
    'verbose': 1,
}

# --- 5. Model Instantiation and Training ---
lstm_q = LSTMQuantileModel(params)
# Prepare input data as dictionaries matching the model's input names
X_train_dict = {'seq_input': X_seq_tr, 'static_input': X_stat_tr}
X_val_dict   = {'seq_input': X_seq_val, 'static_input': X_stat_val}
X_test_dict  = {'seq_input': X_seq_te,  'static_input': X_stat_te}



# --- 6. Prediction and Hold-out Metrics ---
# Measure prediction time on the test set
t0       = time.perf_counter()
history  = lstm_q.fit(X_train_dict, y_tr, X_val=X_val_dict, y_val=y_val)
train_tm = time.perf_counter() - t0

# Prédiction et métriques de hold-out
t0    = time.perf_counter()
preds = lstm_q.predict(X_test_dict)
pred_tm = time.perf_counter() - t0

# Unpack predictions
lower, med, upper = preds['lower'], preds['median'], preds['upper']

# Calculate errors based on the median prediction
errors = med - y_te

# Calculate performance metrics
metrics = {
    'rmse': np.sqrt((errors**2).mean()),
    'mae': np.abs(errors).mean(),
    'mape': np.mean(np.abs(errors / y_te)) * 100,
    'sd': errors.std(),
    'coverage_pct': float(np.mean((y_te >= lower) & (y_te <= upper)) * 100),
    'mean_width': float(np.mean(upper - lower)),
    'test_pred_time_s': pred_tm
}

# --- 7. Save Metrics and History ---
with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

# --- 8. Plot Learning Curves ---
plt.figure()
plt.plot(history.history['loss'],    label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('Loss par époque')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'loss_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history['median_mae'],    label='train_median_mae')
plt.plot(history.history['val_median_mae'],label='val_median_mae')
plt.title('Median MAE par époque')
plt.xlabel('Époque')
plt.ylabel('MAE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'median_mae_curve.png'))
plt.close()

print(f'Training done. Hold-out metrics saved to {metrics_dir}')

print("\n6. Creating and saving prediction visualization...")

# --- Prepare DataFrame for Visualization ---
# 1. Get the dates corresponding to the test set
# The logic relies on the length of y_te to slice the original dataframe
df_full_dates = dh.load_data()[[date_col]]
df_test_dates = df_full_dates.iloc[-len(y_te):]

# 2. Create the results DataFrame
# The predictions 'lower', 'med', 'upper' are already available
results_df = pd.DataFrame({
    'datetime': df_test_dates[date_col].values,
    'y_true': y_te,
    'y_pred': med,  # 'med' vient de vos prédictions LSTM
    'lower_bound': lower, # 'lower' vient de vos prédictions LSTM
    'upper_bound': upper  # 'upper' vient de vos prédictions LSTM
}).set_index('datetime')

# --- Create the Plot ---
# 3. Select a shorter period for better readability (e.g., 7 days)
start_date = results_df.index.min()
end_date = start_date + pd.Timedelta(days=7)
results_subset = results_df.loc[start_date:end_date]

# 4. Generate the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 8))

# The prediction interval (shaded area)
# The label reflects the quantiles used (0.01 and 0.99 -> 98% interval)
ax.fill_between(
    results_subset.index, 
    results_subset['lower_bound'], 
    results_subset['upper_bound'], 
    color='darkorange', 
    alpha=0.3, 
    label='98% Prediction Interval'
)

# The actual values curve
ax.plot(
    results_subset.index, 
    results_subset['y_true'], 
    color='navy', 
    linewidth=2, 
    marker='o', 
    markersize=3, 
    alpha=0.8, 
    label='Actual Values'
)

# The median prediction curve
ax.plot(
    results_subset.index, 
    results_subset['y_pred'], 
    color='crimson', 
    linestyle='--', 
    linewidth=2.5, 
    label='Point Prediction (Median)'
)

# 5. Formatting and Saving
ax.set_title(f'LSTM National Energy Demand Forecast ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})', fontsize=18, fontweight='bold')
ax.set_xlabel('Date & Time', fontsize=14)
ax.set_ylabel('Energy Demand (MW)', fontsize=14)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()

# Use the `fig_dir` defined at the top of the script
output_path = os.path.join(fig_dir, 'lstm_prediction_with_interval.png')
plt.savefig(output_path, dpi=300)
print(f"Prediction visualization saved to: {output_path}")

plt.show()