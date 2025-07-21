import os
import json

from src.data.data_handler import DataHandler
from src.models.xgboost_model import XGBoostModel

# Load the best hyperparameters 
# Load the best parameters previously saved from a hyperparameter tuning experiment
exp_name     = 'all_features_std_norm_xgboost_experiment'
params_path  = os.path.join('outputs', 'experiments', exp_name, 'metrics', 'best_model_params.json')

with open(params_path, 'r') as f:
    best_params = json.load(f)

# Initialize the DataHandler (reserve 1 year for testing) 
# Load the processed dataset and define the input/output columns and scaling configuration
csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
feature_cols  = [
    'NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
    'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
    'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
    'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]
target_col    = 'I014_ND'

no_scale_cols = [
    'is_interpolated','is_weekend',
    'hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos'
]

dh = DataHandler(
    csv_path      = csv_path,
    date_col      = date_col,
    feature_cols  = feature_cols,
    target_col    = target_col,
    no_scale_cols = no_scale_cols,
    holdout_ratio = None,
    holdout_years = 1,       # reserve 1 year for the test set
    scaler_type   = 'standard'
)

# Split into train / validation / test sets ---
# Chronological split:
#   • train  : all data except the last 2 years
#   • val    : second to last year
#   • test   : last year
X_train, y_train, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)

# Train XGBoost only on the training set 
xgb = XGBoostModel(params=best_params)
xgb.fit(X_train, y_train)


# Evaluate the model 
test_metrics = xgb.evaluate(X_test, y_test)
print("Test metrics:      ", test_metrics)

# Save the model and test metrics 
# Create output directory for saving results
out_dir = os.path.join('outputs', 'experiments', f'{exp_name}_retrain_on_train_only')
os.makedirs(out_dir, exist_ok=True)

# Save trained model
xgb.save(os.path.join(out_dir, 'xgb_retrained_on_train.pkl'))


# Save evaluation metrics
with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f:
    json.dump(test_metrics, f, indent=2)
