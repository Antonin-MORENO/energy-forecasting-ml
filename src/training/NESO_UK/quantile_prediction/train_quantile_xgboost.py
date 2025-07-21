import os
import time
import json
import joblib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from src.models.xgboost_quantile_model import XGBoostQuantileModel
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import make_scorer, mean_pinball_loss

from src.data.data_handler import DataHandler


# Edit experiment name
exp_name = 'xgboost_quantile_prediction_standard_200_nestimators_99_quantiles_nestimators_1000_3000'

# Prepare output directory for metrics
# Creates: outputs/experiments/<exp_name>/metrics/
output_dir = os.path.join('outputs', 'experiments', exp_name)
models_dir = os.path.join(output_dir, 'models')
metrics_dir = os.path.join(output_dir, 'metrics')
fig_dir = os.path.join(output_dir, 'figures')
for d in [models_dir, metrics_dir, fig_dir]:
    os.makedirs(d, exist_ok=True)

# Define data paths and features
csv_path = 'data/processed/NESO_UK/preprocessed_new_features_cleaned_DemandData_2011-2018.csv'
date_col = 'datetime'
feature_cols = ['NON_BM_STOR','I014_PUMP_STORAGE_PUMPING','is_interpolated','is_weekend',
                'I014_ND_lag_1','I014_ND_lag_2','I014_ND_lag_48','I014_ND_lag_96','I014_ND_lag_336',
                'I014_ND_mean_48','I014_ND_mean_336','net_import','wind_capacity_factor',
                'solar_capacity_factor','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']
target_col = 'I014_ND'
no_scale_cols = ['is_interpolated','is_weekend','hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']

# Load and split the data, holding out the last full year as test
print("1. Loading and preparing data...")
dh = DataHandler(
    csv_path=csv_path,
    date_col=date_col,
    feature_cols=feature_cols,
    target_col=target_col,
    no_scale_cols=no_scale_cols,
    holdout_years=1,    # Hold out the last full year as test
    scaler_type='standard'  
)


X_train, y_train, X_val, y_val, X_test, y_test = dh.get_train_val_test_split(val_years=1)
print("Data ready.")


# Setup for Hyperparameter Optimization - BayesSearchCV
print("\n2. Optimizing hyperparameters with BayesSearchCV for each quantile...")

# Define the search space for Bayesian optimization
search_spaces = {
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
    'subsample': Real(0.6, 1.0, 'uniform'),
    'colsample_bytree': Real(0.6, 1.0, 'uniform'),
    'gamma': Real(0, 5, 'uniform'),
    'min_child_weight': Integer(1, 10)
}

# Concatenate train and validation sets for cross-validation
X_trval = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], axis=0)
y_trval = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define quantiles to train
quantiles_to_train = {'lower': 0.01, 'median': 0.5, 'upper': 0.99}
best_models_dict = {}
best_params_all = {}

for name, q_value in quantiles_to_train.items():
    # Inform which quantile is being processed
    print(f"\n--- Running BayesSearchCV for '{name}' quantile ({q_value}) ---")
    start_time = time.time()
    
    # Define the scoring function for pinball loss
    pinball_scorer = make_scorer(
        lambda y_true, y_pred: mean_pinball_loss(y_true, y_pred, alpha=q_value),
        greater_is_better=False)
    
    # Base estimator for quantile regression
    estimator = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q_value,
        seed=42,
        n_jobs=-1,
        n_estimators=1000)
    
    # Configure Bayesian search
    bs = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=50,
        cv=tscv,
        scoring=pinball_scorer,
        return_train_score=False,
        n_jobs=-1,
        verbose=3,
        refit=False, 
        random_state=42
    )
    
    # Run hyperparameter optimization
    bs.fit(X_trval.values, y_trval.values)
    
    print(f"BayesSearchCV for '{name}' finished in {time.time() - start_time:.1f}s")
    print(f"Refitting best model for '{name}' with early stopping...")
    
    # Extract best parameters and add early stopping in final estimator
    best_params_from_search = dict(bs.best_params_)
    final_estimator = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q_value,
        seed=42,
        n_jobs=-1,
        early_stopping_rounds=200, 
        n_estimators=3000,
        **best_params_from_search
    )
    
    # Fit on train and validation with early stopping
    final_estimator.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=0
    )
    # Evaluate the best model on validation set
    evals_result = final_estimator.evals_result()

    train_metrics = evals_result['validation_0']
    val_metrics = evals_result['validation_1']
    metric_name = list(train_metrics.keys())[0]
    train_loss = train_metrics[metric_name]
    val_loss = val_metrics[metric_name]
    rounds = list(range(len(train_loss)))
    
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, train_loss, label='Train loss')
    plt.plot(rounds, val_loss, label='Val loss')
    plt.axvline(final_estimator.best_iteration, color='red', linestyle='--', label='Best iteration')
    plt.title(f'Learning Curve for {name} quantile (alpha={q_value})')
    plt.xlabel('Boosting Round')
    plt.ylabel('Pinball Loss')
    plt.legend()
    plt.tight_layout()
    lc_path = os.path.join(fig_dir, f'learning_curve_{name}.png')
    plt.savefig(lc_path, dpi=300)
    plt.close()
    best_models_dict[name] = final_estimator
    best_params_all[name] = best_params_from_search
    best_params_all[name]['n_estimators_found'] = final_estimator.best_iteration
    print(f"Optimal number of estimators found: {final_estimator.best_iteration}")
    
    # Save the model to disk
    model_path = os.path.join(models_dir, f'best_model_{name}.pkl')
    joblib.dump(final_estimator, model_path)
    print(f"Best model for '{name}' saved to: {model_path}")


    
# Build and Save Unified Quantile Wrapper Model ---
print("\n3. Building and saving the unified quantile model...")
final_model = XGBoostQuantileModel()
final_model.models = best_models_dict
final_model.params = best_params_all

final_model_path = os.path.join(models_dir, 'final_quantile_model.pkl')
final_model.save(final_model_path)
print(f"Unified model wrapper saved to: {final_model_path}")

# Evaluate on Test Set 
print("\n4. Evaluating the unified model on the test set...")

df_full_dates = dh.load_data()[[date_col]]
df_test_dates = df_full_dates.iloc[-len(X_test):]

holdout_metrics_from_model = final_model.evaluate(X_test, y_test)
holdout_metrics = {'params': best_params_all, **holdout_metrics_from_model}
metrics_path = os.path.join(metrics_dir, 'holdout_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(holdout_metrics, f, indent=4, default=str)

print(f"\nMetrics and best parameters saved to: {metrics_path}")
print("\n--- Holdout Set Metrics (from .evaluate() method) ---")
print(json.dumps(holdout_metrics, indent=4, default=str))
print("------------------------------------------------------\n")


# Visualization of Test Predictions 

print("5. Creating prediction visualization...")
predictions = final_model.predict(X_test)

results_df = pd.DataFrame({
    'datetime': df_test_dates[date_col].values,
    'y_true': y_test,
    'y_pred': predictions['median'],
    'lower_bound': predictions['lower'],
    'upper_bound': predictions['upper']
}).set_index('datetime')

start_date, end_date = results_df.index.min().strftime('%Y-%m-%d'), (results_df.index.min() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
results_subset = results_df[start_date:end_date]


# Plot first 7 days
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 8))
ax.fill_between(results_subset.index, results_subset['lower_bound'], results_subset['upper_bound'], color='darkorange', alpha=0.3, label='95% Prediction Interval')
ax.plot(results_subset.index, results_subset['y_true'], color='navy', linewidth=2, marker='o', markersize=3, alpha=0.8, label='Actual Values')
ax.plot(results_subset.index, results_subset['y_pred'], color='crimson', linestyle='--', linewidth=2.5, label='Point Prediction (Median)')
ax.set_title(f'National Energy Demand Forecast ({start_date} to {end_date})', fontsize=18, fontweight='bold')
ax.set_xlabel('Date & Time', fontsize=14)
ax.set_ylabel('Energy Demand (MW)', fontsize=14)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()

output_path = os.path.join(fig_dir, 'xgboost_prediction_with_interval.png')
plt.savefig(output_path, dpi=300)
print(f"Graph saved to: {output_path}")

plt.show()