import os
import joblib
from sklearn.svm import SVR
from src.models.base_model import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class SVMModel(BaseModel, BaseEstimator, RegressorMixin):
    """
    Wrapper around sklearn.svm.SVR.
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
        Instantiates the SVR model using values from self.params.
        """
        p = self.params
        # Get SVR specific hyperparameters
        C = p.get('C', 1.0)
        kernel = p.get('kernel', 'rbf')
        gamma = p.get('gamma', 'scale')
        epsilon = p.get('epsilon', 0.1)

        self.model = SVR(
            C=C,
            kernel=kernel,
            gamma=gamma,
            epsilon=epsilon
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train SVR on (X_train, y_train).
        X_val and y_val are ignored but kept for interface consistency.
        """
        
        # Scale target if needed
        if self.params.get('scale_y', False):
            self.y_scaler = StandardScaler()
            y_train = self.y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
            
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Returns a 1D array of predictions from the trained SVR model.
        """
        preds = self.model.predict(X)
        # Inverse transform the target if it was scaled during training
        if self.params.get('scale_y', False):
            preds = self.y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            
        return preds

    def get_params(self, deep=True):
        """
        Returns hyperparameters for GridSearchCV and clone().
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
        Save the trained rf model to disk.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a trained rf model from disk.
        """
        return joblib.load(path)