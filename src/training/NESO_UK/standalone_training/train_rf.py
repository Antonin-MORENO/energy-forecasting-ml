import os
import time
import json
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error
from src.data.data_handler import DataHandler
from src.models.rf_model import RandomForestModel 

# Experiment name
exp_name = 'all_features_std_norm_rf_experiment'
exp_metrics = os.path.join('outputs', 'experiments', exp_name, 'metrics')
os.makedirs(exp_metrics, exist_ok=True)

# Data settings
csv_path = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col = 'datetime'
feature_cols = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]
target_col = 'I014_ND'
no_scale_cols = ['is_interpolated','is_weekend','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']

# Load and split the data
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

df_full = dh.load_data()
df_trainval, df_test = dh.temporal_split(df_full)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0]
}
tscv = TimeSeriesSplit(n_splits=5)

# Scoring
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    'sd': make_scorer(lambda y, y_pred: np.std(y - y_pred), greater_is_better=False)
}

# GridSearchCV
gs = GridSearchCV(
    estimator=RandomForestModel(),
    param_grid=param_grid,
    cv=tscv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
    verbose=3,
    refit='rmse'
)

print("Running GridSearchCV for RandomForestModel...")
start = time.perf_counter()
gs.fit(X_trval, y_trval)
print(f'GridSearchCV done in {time.perf_counter() - start:.1f}s')

# Best model and parameters
best_model = gs.best_estimator_
best_params = gs.best_params_

# Test set evaluation
t0 = time.perf_counter()
y_pred = best_model.predict(X_test)
pred_time = time.perf_counter() - t0

holdout_metrics = best_model.evaluate(X_test, y_test)
holdout_metrics.update({
    'test_pred_time_s': round(pred_time, 4),
    'params': best_params
})

# Extract CV metrics
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

print(f' Best model saved to:    {model_path}')
print(f' Params saved to:        {params_path}')
print(f' Metrics saved to:       {metrics_path}')
