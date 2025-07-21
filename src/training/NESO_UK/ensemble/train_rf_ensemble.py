import os
import json

from src.data.data_handler import DataHandler
from src.models.rf_model import RandomForestModel


# Load the best hyperparameters 
# Load best parameters obtained from previous hyperparameter tuning for the Random Forest--
exp_name    = 'all_features_std_norm_rf_experiment'
params_path = os.path.join('outputs', 'experiments', exp_name, 'metrics', 'best_model_params.json')
with open(params_path, 'r') as f:
    best_params = json.load(f)


# Set up the DataHandler (1 year reserved for testing)
# Load the cleaned dataset and define input features, target, and scaling configuration
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

# Initialize the DataHandler with the specified configuration
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

# Chronologically split data into train, validation, and test sets ---
# This method returns:
#   - training set: everything except the last two years
#   - validation set: second-to-last year
#   - test set: last year
X_train, y_train, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)


# Train Random Forest using only the training set
rf = RandomForestModel(params=best_params)
rf.fit(X_train, y_train)

# Evaluate model on the test set only 
test_metrics = rf.evaluate(X_test, y_test)

print("Test metrics:      ", test_metrics)

# Save the model and test metrics
out_dir = os.path.join('outputs', 'experiments', f'{exp_name}_retrain_on_train_only')
os.makedirs(out_dir, exist_ok=True)

# Save the trained model
rf.save(os.path.join(out_dir, 'rf_retrained_on_train.pkl'))

# Save evaluation metrics on the test set
with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f:
    json.dump(test_metrics, f, indent=2)
