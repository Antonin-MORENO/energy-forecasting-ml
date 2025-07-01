import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.models.base_model import BaseModel

class XGBoostQuantileModel(BaseModel):
    """
    Wrapper for quantile regression with XGBoost.
    Manages three internal models to predict a confidence interval and a median.
    """


    def __init__(self, params: dict = None):
        """
        Initializes the XGBoostQuantileModel.

        Args:
            params (dict, optional): A dictionary of parameters for the internal XGBoost models.
                It can contain keys 'lower', 'median', and 'upper', where each key corresponds
                to a dictionary of parameters for the respective quantile model.
                Defaults to None.
        """
        super().__init__(params)
        # These quantiles define a 98% prediction interval (0.99 - 0.01 = 0.98) and the median (0.5).
        self.quantiles = [0.01, 0.5, 0.99]
        self.quantile_map = {0.01: 'lower', 0.5: 'median', 0.99: 'upper'}
        self.models = {}
        

        if self.params:
            self.build_models()

    def build_models(self):
        """ Builds the three internal XGBoost models, one for each quantile. """
        for q in self.quantiles:
            key = self.quantile_map[q]
            # Get specific parameters for the current quantile model (e.g., params['lower']).
            # If not provided, it uses an empty dictionary, so default XGBoost params are used.
            model_params = self.params.get(key, {})
            
            # Each model is a standard XGBoost regressor, but with a special objective function.
            self.models[q] = xgb.XGBRegressor(
                # 'reg:quantileerror' is the objective that enables quantile regression
                objective='reg:quantileerror',
                quantile_alpha=q,
                seed=42,
                n_jobs=-1,
                **model_params 
            )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """ Trains the three models, one for each quantile. """
        for q, model in self.models.items():
            print(f"   - Training model for quantile {q}...")
            model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """ Predicts the lower bound, median, and upper bound for the input data. """
        preds = {}
        for key, model in self.models.items():
            preds[key] = model.predict(X) 
        return preds

    def evaluate(self, X_test, y_test):
        """ Evaluates the model's performance. """
        predictions = self.predict(X_test)
        y_pred_median = predictions['median']
        y_pred_lower = predictions['lower']
        y_pred_upper = predictions['upper']

        # Point prediction metrics (based on the median)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_median))
        mae = mean_absolute_error(y_test, y_pred_median)
        mape = mean_absolute_percentage_error(y_test, y_pred_median) * 100
        
        # Prediction interval metrics
        coverage = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)) * 100
        interval_width = np.mean(y_pred_upper - y_pred_lower)

        metrics = {
            'point_prediction_metrics': {'rmse': rmse, 'mae': mae, 'mape': f"{mape:.2f}%"},
            'prediction_interval_metrics': {'picp': f"{coverage:.2f}%", 'mpiw': interval_width},
        }
        return metrics

    def save(self, path: str):
        """ Saves the complete wrapper (with its 3 models). """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """ Loads a saved wrapper. """
        return joblib.load(path)