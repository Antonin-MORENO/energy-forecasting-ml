import os
import time
import json
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from src.data.data_handler import DataHandler
from src.models.xgboost_model import XGBoostModel 
import matplotlib.pyplot as plt 


# Experiment name
exp_name = 'temp_pressure_xgboost_experiment'

# Output directories
exp_metrics = os.path.join('outputs', 'experiments_UMass', 'temp', exp_name, 'metrics')
os.makedirs(exp_metrics, exist_ok=True)

# Dataset configuration
csv_path      = 'data/processed/UMass Smart Dataset/weather_lag.csv'  
date_col      = 'time'
feature_cols  = [
    'pressure'

]
target_col    = 'temperature'
no_scale_cols = []       

# Load and temporally split data (last year used as test set)
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

df_full        = dh.load_data()
df_trainval, df_test = dh.temporal_split(df_full)
X_trval, y_trval, X_test, y_test = dh.scale_split(df_trainval, df_test)
print(X_trval)

# Define hyperparameter grid for XGBoost
param_grid = {
    'n_estimators'      : [100, 200, 500],
    'max_depth'         : [3, 5, 7],
    'learning_rate'     : [0.01, 0.05, 0.1],
    'subsample'         : [0.6, 0.8, 1.0],
    'colsample_bytree'  : [0.6, 0.8, 1.0]
}
tscv = TimeSeriesSplit(n_splits=5)

# Perform Grid Search CV to find the best XGBoost configuration
gs = GridSearchCV(
    estimator        = XGBoostModel(),
    param_grid       = param_grid,
    cv               = tscv,
    scoring          = 'neg_root_mean_squared_error',
    return_train_score = True,
    n_jobs           = -1,
    verbose          = 3,
    refit            = True
)

print("🔍 Lancement du GridSearchCV pour la prévision de température…")
start = time.perf_counter()
gs.fit(X_trval, y_trval)
print(f"✅ GridSearch terminé en {time.perf_counter() - start:.1f}s")

# Retrieve best model and hyperparameters
best_model  = gs.best_estimator_
best_params = gs.best_params_

# Refit the best model on the full training+validation set
best_model.fit(X_trval, y_trval)

# Predict and evaluate performance on test set
t0 = time.perf_counter()
y_pred     = best_model.predict(X_test)
pred_time  = time.perf_counter() - t0

holdout_metrics = best_model.evaluate(X_test, y_test)
holdout_metrics.update({
    'test_pred_time_s': round(pred_time, 4),
    'best_params'     : best_params
})

# Save model, parameters, and metrics to disk
model_dir   = exp_metrics
os.makedirs(model_dir, exist_ok=True)

# Save trained model
model_path  = os.path.join(model_dir, 'best_model.pkl')
best_model.save(model_path)

# Save best hyperparameters
params_path = os.path.join(model_dir, 'best_model_params.json')
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=2)

# Save test evaluation metrics
metrics_path = os.path.join(model_dir, 'best_model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(holdout_metrics, f, indent=2)


# Plot actual vs predicted temperature for the test set
dates = df_test[date_col]
plt.figure(figsize=(12, 6))
plt.plot(dates, y_test,  label='Actual',   linewidth=1)
plt.plot(dates, y_pred,  label='Predicted', linewidth=1)
plt.xlabel('Time')
plt.ylabel(target_col.capitalize())
plt.title('Actual vs Predicted Temperature')
plt.legend()
plt.tight_layout()


# Save the plot
fig_path = os.path.join(exp_metrics, 'pred_vs_actual.png')
plt.savefig(fig_path)
plt.close()

print(f"💾 Courbe Actual vs Pred saved to: {fig_path}")
print(f"💾 Modèle sauvegardé dans      : {model_path}")
print(f"💾 Paramètres sauvegardés dans : {params_path}")
print(f"💾 Métriques test sauvegardées : {metrics_path}")
