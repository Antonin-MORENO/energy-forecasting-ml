import os
import time
import json
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.data.data_handler import DataHandler
from src.models.svm_model import SVMModel 


# Edit experiment name
exp_name = 'all_features_std_norm_svm_experiment' 

# Prepare output directory for metrics
# Creates: outputs/experiments/<exp_name>/metrics/
exp_metrics = os.path.join('outputs', 'experiments', exp_name, 'metrics')
os.makedirs(exp_metrics, exist_ok=True)


# --- Data loading and splitting remains identical ---
csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
feature_cols = ['NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
                'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
                'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
                'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]
target_col    = 'I014_ND'
no_scale_cols = ['is_interpolated','is_weekend','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']

# Load and split the data, holding out the last full year as test
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 5,              # hold out the last full year
    scaler_type   = 'standard'
)

df_full = dh.load_data()
df_trainval, df_test = dh.temporal_split(df_full)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)



# This approach is more efficient because it avoids testing useless combinations,
# such as the 'gamma' parameter with a 'linear' kernel.
# Total fits in GridSearchCV = 6 (linear) + 24 (rbf) = 30

param_grid = [
    
    # Block 1: Parameters for the 'linear' kernel.
    # 'gamma' is not included here as it is ignored by the linear kernel.
    # Combinations: 3 (C) * 2 (epsilon) = 6
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.01]
    },
    
    # Block 2: Parameters for the 'rbf' (Radial Basis Function) kernel.
    # This block includes 'gamma', which is a key parameter for the RBF kernel.
    # Combinations: 3 (C) * 4 (gamma) * 2 (epsilon) = 24
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1, 'scale'],
        'epsilon': [0.1, 0.01]
    }
]
tscv = TimeSeriesSplit(n_splits=5)

# Set up GridSearchCV 
gs = GridSearchCV(
    estimator=SVMModel(params={'scale_y': True}), 
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',  # sklearn maximizes, so negative RMSE
    return_train_score=True,                # also record train scores
    n_jobs=-1,                              # parallelize across all CPUs
    verbose=3,                              # show progress
    refit=True                              # retrain best model on full train
)

# Run the grid search 
print("Running GridSearchCV for SVMModel...")
start = time.perf_counter() # Measure the time to fit the model with all parameter combinations
gs.fit(X_trval, y_trval)
print(f'GridSearchCV done in {time.perf_counter() - start:.1f}s')


# Load and split the data, holding out the last full year as test
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 1,              # hold out the last full year
    scaler_type   = 'standard'
)

df_full = dh.load_data()
df_trainval, df_test = dh.temporal_split(df_full)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)


# Retrieve the best model and its parameters
best_model = gs.best_estimator_
best_params = gs.best_params_

best_model.fit(X_trval, y_trval)  # Fit the best model on the full training set

# Evaluate on the test set
t0 = time.perf_counter()
y_pred = best_model.predict(X_test)
pred_time = time.perf_counter() - t0

# Compute holdout metrics
holdout_metrics = best_model.evaluate(X_test, y_test)
holdout_metrics.update({
    'test_pred_time_s': round(pred_time, 4),
    'params': best_params
})


# Create model output directory
model_dir = os.path.join('outputs', 'experiments', exp_name, 'metrics')
os.makedirs(model_dir, exist_ok=True)

# 1. Save trained model
model_path = os.path.join(model_dir, 'best_model.pkl')
best_model.save(model_path)

# 2. Save best hyperparameters
params_path = os.path.join(model_dir, 'best_model_params.json')
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

# 3. Save test set metrics
metrics_path = os.path.join(model_dir, 'best_model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(holdout_metrics, f, indent=2)

print(f' Best model saved to: {model_path}')
print(f' Params saved to:     {params_path}')
print(f' Test metrics saved to: {metrics_path}')
