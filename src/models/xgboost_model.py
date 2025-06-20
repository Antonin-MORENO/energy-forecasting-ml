import os
import joblib
from src.models.base_model import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class XGBoostModel(BaseModel, BaseEstimator, RegressorMixin):
    """
    Wrapper around xgboost.XGBRegressor.
    Implements build_model, fit, predict.
    Serialization (save/load) is not implemented here.
    """

    def __init__(self, params: dict = None, **kwargs):
        """
        Initialize the model by merging a dictionary of parameters with any provided keyword args.

        Args:
            params (dict, optional): Initial parameter dictionary.
            **kwargs: Named parameters passed by sklearn.clone().
        """
        
        if params is None:
            params = {}

        params.update(kwargs)
        super().__init__(params)
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Instantiate the XGBRegressor model using values from self.params.
        """
        p = self.params
        # Get XGBoost specific hyperparameters
        n_estimators = p.get("n_estimators", 100)
        max_depth = p.get("max_depth", 3)
        learning_rate = p.get("learning_rate", 0.1)
        subsample = p.get("subsample", 1.0)
        colsample_bytree = p.get("colsample_bytree", 1.0)
        objective = p.get("objective", "reg:squarederror")
        random_state = p.get("random_state", 42)

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost on (X_train, y_train).
        X_val and y_val are ignored but kept for interface consistency.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Return a 1D array of predictions from the trained XGBoost model.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Return hyperparameters for GridSearchCV and clone().
        """
        return self.params.copy()

    def set_params(self, **new_params):
        """
        Update hyperparameters, rebuild the model, and return self.
        
        Returns:
            self: Enables method chaining.
        """
        self.params.update(new_params)
        self.build_model()
        return self

    def save(self, path: str):
        """
        Save the trained XGBoost model to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a trained XGBoost model from disk.
        """
        return joblib.load(path)