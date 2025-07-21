import json
import os

from src.data.data_handler import DataHandler
from src.models.svm_model import SVMModel

# Load the best hyperparameters
# Load best parameters previously obtained from a hyperparameter tuning experiment
params_path = os.path.join(
    'outputs', 'experiments', 
    'all_features_std_norm_svm_experiment', 
    'metrics', 
    'best_model_params.json'
)
with open(params_path, 'r') as f:
    best_params = json.load(f)


# Set up the DataHandler (hold out 1 year for testing) 
# Load preprocessed dataset and define relevant columns and scaling configuration
csv_path      = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col      = 'datetime'
feature_cols = [
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


# Initialize the data handler with 1 year held out for testing
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

# Chronologically split into train / validation / test sets 
#   • Train: all data except the last 2 years
#   • Val: second to last year
#   • Test: last year
X_train, y_train, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)

# Instantiate and train the SVM using only the training set 
# Include 'scale_y=True' to normalize the target during training, which is often recommended for SVM regression
svm = SVMModel(params={'scale_y': True, **best_params})
svm.fit(X_train, y_train)

# Evaluate the trained model on the test set 
test_metrics = svm.evaluate(X_test, y_test)


print("Test metrics:      ", test_metrics)


# Save the trained model and test metrics
out_dir = os.path.join('outputs', 'experiments', 'retrain_on_train_only')
os.makedirs(out_dir, exist_ok=True)

# Save the SVM model
svm.save(os.path.join(out_dir, 'svm_retrained_on_train.pkl'))

# Save test evaluation metrics
with open(os.path.join(out_dir, 'metrics_test.json'), 'w') as f:
    json.dump(test_metrics, f, indent=2)
