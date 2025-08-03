import os
import time
import json
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error
from src.data.data_handler import DataHandler
from src.models.xgboost_model import XGBoostModel 

# Edit experiment name
exp_name = 'all_features_std_norm_xgboost_experiment' 

# Prepare output directory
exp_metrics = os.path.join('outputs', 'experiments', exp_name, 'metrics')
os.makedirs(exp_metrics, exist_ok=True)

# Define data paths and features
csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
feature_cols = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]
target_col    = 'I014_ND'
no_scale_cols = ['is_interpolated','is_weekend','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']

# Load and split the data
dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 1,
    scaler_type   = 'standard'
)

df_full = dh.load_data()
df_trainval, df_test = dh.temporal_split(df_full)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)

# Configure hyperparameter grid
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Define multi-metric scoring
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    'sd': make_scorer(lambda y, y_pred: np.std(y - y_pred), greater_is_better=False)
}

# GridSearch with refit on RMSE
gs = GridSearchCV(
    estimator=XGBoostModel(), 
    param_grid=param_grid,
    cv=tscv,
    scoring=scoring,
    refit='rmse',
    return_train_score=False,
    n_jobs=-1,
    verbose=3
)

# Run GridSearch
print("Running GridSearchCV for XGBoostModel...")
start = time.perf_counter()
gs.fit(X_trval, y_trval)
print(f'GridSearchCV done in {time.perf_counter() - start:.1f}s')

# Retrieve best model and parameters
best_model = gs.best_estimator_
best_params = gs.best_params_

# Evaluate on test set
t0 = time.perf_counter()
y_pred = best_model.predict(X_test)
pred_time = time.perf_counter() - t0

# Test set metrics
holdout_metrics = best_model.evaluate(X_test, y_test)
holdout_metrics.update({
    'test_pred_time_s': round(pred_time, 4),
    'params': best_params
})

# Extract cross-validation (5-fold) metrics from best model
idx = gs.best_index_
cv_results = gs.cv_results_
cv_metrics = {
    'cv_rmse': round(-cv_results['mean_test_rmse'][idx], 3),
    'cv_mae':  round(-cv_results['mean_test_mae'][idx], 3),
    'cv_mape': round(-cv_results['mean_test_mape'][idx], 3),
    'cv_sd':   round(-cv_results['mean_test_sd'][idx], 3)
}
holdout_metrics.update(cv_metrics)

# Save outputs
model_path = os.path.join(exp_metrics, 'best_model.pkl')
params_path = os.path.join(exp_metrics, 'best_model_params.json')
metrics_path = os.path.join(exp_metrics, 'best_model_metrics.json')

best_model.save(model_path)

with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

with open(metrics_path, 'w') as f:
    json.dump(holdout_metrics, f, indent=2)

print(f" Best model saved to:    {model_path}")
print(f" Params saved to:        {params_path}")
print(f" Metrics saved to:       {metrics_path}")
