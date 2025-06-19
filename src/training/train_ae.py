import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from src.data.data_handler import DataHandler
from src.models.autoencoder_model import AutoEncoderModel

# --- Configuration ---

# Experiment name and directory setup
exp_name    = 'ae_smooth_funnel_32'
base_dir    = os.path.join('outputs', 'experiments', exp_name)
metrics_dir = os.path.join(base_dir, 'metrics')
fig_dir     = os.path.join(base_dir, 'figures')
ckpt_dir    = os.path.join(base_dir, 'checkpoints')

# Create output directories if they don't exist
for d in (metrics_dir, fig_dir, ckpt_dir):
    os.makedirs(d, exist_ok=True)

# --- Data Loading and Preprocessing Setup ---

# Path to the dataset and column definitions
csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
target_col    = 'I014_ND' 
feature_cols  = [
    'NON_BM_STOR', 'I014_PUMP_STORAGE_PUMPING', 'is_interpolated', 'is_weekend',
    'I014_ND_lag_1', 'I014_ND_lag_2', 'I014_ND_lag_48', 'I014_ND_lag_96', 'I014_ND_lag_336',
    'I014_ND_mean_48', 'I014_ND_mean_336', 'net_import', 'wind_capacity_factor',
    'solar_capacity_factor', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]

no_scale_cols = [
    'is_interpolated', 'is_weekend',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos'
]

# --- Data Processing ---

# Instantiate the data handler with the specified configuration
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_years = 1,          # Use the last year of data as a holdout test set
    scaler_type   = 'standard'  # Use StandardScaler for feature scaling
)

# Load the data and perform a time-based split for the test set
df = dh.load_data()
df_trainval, df_test = dh.temporal_split(df)

# Scale features and separate features (X) from the target (y)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)

# --- Manual Train/Validation Split ---

# Split the training+validation set into a final training set and a validation set
n = X_trval.shape[0]
split_idx = int(n * 0.9)  # 90% for training, 10% for validation
X_train, X_val = X_trval[:split_idx], X_trval[split_idx:]
y_train, y_val = y_trval[:split_idx], y_trval[split_idx:] # y is kept for completeness but not used by the autoencoder

# --- Model Configuration ---

# Define hyperparameters for the autoencoder model
params = {
    'input_shape': X_train.shape[1],
    'epochs': 200,
    'batch_size': 64,
    'early_stop_patience': 10,
    'checkpoint_path': os.path.join(ckpt_dir, 'smooth_funnel_ae.keras'),
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mae']
}

# Instantiate the autoencoder model
ae = AutoEncoderModel(params)

# --- Model Training ---

print("Starting model training...")
t0      = time.perf_counter()
history = ae.fit(X_train, X_val=X_val) # Train the model
train_s = time.perf_counter() - t0
print(f"Training finished in {train_s:.2f} seconds.")

# --- Evaluation on Test Set ---

# Reconstruct the test set features using the trained autoencoder
X_recon = ae.predict(X_test)
errors  = X_test - X_recon
mse     = np.mean(errors**2)
mae     = np.mean(np.abs(errors))

# --- Save Results ---

# Save holdout set metrics to a JSON file
metrics = {
    'train_time_s': train_s,
    'recon_mse_test': float(mse),
    'recon_mae_test': float(mae)
}
with open(os.path.join(metrics_dir, 'holdout_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# Save the complete training history (loss, mae, etc.) to a JSON file
with open(os.path.join(metrics_dir, 'training_history.json'), 'w') as f:
    json.dump(history.history, f, indent=2)

# --- Plotting and Visualization ---

# Plot training & validation MSE curve
plt.figure()
plt.plot(history.history['loss'],     label='train_mse')
plt.plot(history.history['val_loss'], label='val_mse')
plt.title('MSE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mse_curve.png'))
plt.close()

# Plot training & validation MAE curve
plt.figure()
plt.plot(history.history['mae'],      label='train_mae')
plt.plot(history.history['val_mae'],  label='val_mae')
plt.title('MAE per epoch')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'mae_curve.png'))
plt.close()

# Visualize the comparison between original and reconstructed feature vectors for a few samples
n_samples_viz = 5
for i in range(min(n_samples_viz, X_test.shape[0])):
    plt.figure(figsize=(10, 4))
    data = np.vstack([X_test[i], X_recon[i]])
    im = plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Value (scaled)')
    plt.yticks([0, 1], ['Input', 'Reconstruction'])
    plt.xlabel('Feature Index')
    plt.title(f'Input vs. Output Comparison (Sample {i})')
    plt.savefig(os.path.join(fig_dir, f'comparison_sample_{i}.png'))
    plt.close()

print(f"Training complete. Metrics, figures and comparison plots saved under {base_dir}")