import os
import time
import json
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.data.data_handler import DataHandler
from src.models.svm_model import SVMModel 
import numpy as np


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


# Extract CV results
cv = gs.cv_results_
params_list      = cv['params']
mean_test_rmse   = -cv['mean_test_score']  # convert back to positive RMSE
std_test_rmse    =  cv['std_test_score']
mean_train_rmse  = -cv['mean_train_score']
std_train_rmse   =  cv['std_train_score']
mean_fit_time    =  cv['mean_fit_time']
std_fit_time     =  cv['std_fit_time']



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


# Identify top-3 parameter combinations by CV performance
ranked_idxs = np.argsort(mean_test_rmse)
top3 = []
for rank, idx in enumerate(ranked_idxs[:3], start=1):
    p = params_list[idx]
    
    # Refit on full train for test‐set evaluation
    model = SVMModel(params={'scale_y': True}, **p) 
    model.fit(X_trval, y_trval)
    
    # Measure prediction time on the hold-out test set
    t0 = time.perf_counter()
    model.predict(X_test)
    pred_time = time.perf_counter() - t0
    
    # Compute final hold-out metrics (RMSE, MAE, MAPE, SD)
    # This assumes your BaseModel has an 'evaluate' method
    hold = model.evaluate(X_test, y_test)
    
    
    top3.append({
        'rank':               rank,
        'params':             p,
        'cv_rmse_mean':       round(mean_test_rmse[idx], 3),
        'cv_rmse_std':        round(std_test_rmse[idx], 3),
        'train_rmse_mean':    round(mean_train_rmse[idx], 3),
        'train_rmse_std':     round(std_train_rmse[idx], 3),
        'fit_time_mean_s':    round(mean_fit_time[idx], 3),
        'fit_time_std_s':     round(std_fit_time[idx], 3),
        'test_pred_time_s':   round(pred_time, 4),
        'holdout_rmse':       round(hold['rmse'], 3),
        'holdout_mae':        round(hold['mae'], 3),
        'holdout_mape':       round(hold['mape'], 2),
        'holdout_sd_error':   round(hold['sd'], 3)
    })

# Save JSON files under metrics directory

# Raw cv_results 
with open(os.path.join(exp_metrics, 'cv_results.json'), 'w') as f:
    json.dump(
        {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in cv.items()},
        f, indent=2
    )

# Top-3 configurations + metrics
with open(os.path.join(exp_metrics, 'top3_results.json'), 'w') as f:
    json.dump(top3, f, indent=2)

# Hold-out metrics for all top-3
holdout_all = [
    {
        'rank':             e['rank'],
        'params':           e['params'],
        'holdout_rmse':     e['holdout_rmse'],
        'holdout_mae':      e['holdout_mae'],
        'holdout_mape':     e['holdout_mape'],
        'holdout_sd_error': e['holdout_sd_error'],
        'test_pred_time_s': e['test_pred_time_s']
    }
    for e in top3
]
with open(os.path.join(exp_metrics, 'holdout_metrics.json'), 'w') as f:
    json.dump(holdout_all, f, indent=2)

print(f'Metrics written to {exp_metrics}')