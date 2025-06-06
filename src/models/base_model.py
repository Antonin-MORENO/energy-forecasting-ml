# src/model/base_model.py

import numpy as np

class BaseModel():
    """
    Abstract base class defining the common interface for all models.
    Each subclass must implement:
      - __init__(self, params: dict)
      - build_model(self)          # for neural nets (CNN/LSTM)
      - fit(self, X_train, y_train, X_val=None, y_val=None)
      - predict(self, X)
      - save(self, path: str)
      - @classmethod load(cls, path: str)
      - evaluate(self, X_test, y_test)
    """

    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def build_model(self):
        """
        Instantiate the underlying model architecture.
        To be overridden by subclasses (only necessary for NN-based models).
        """
        raise NotImplementedError("build_model() must be implemented in the child class")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on (X_train, y_train), optionally using (X_val, y_val) for validation.
        To be overridden by subclasses.
        """
        raise NotImplementedError("fit() must be implemented in the child class")

    def predict(self, X):
        """
        Return model predictions for input X.
        To be overridden by subclasses.
        """
        raise NotImplementedError("predict() must be implemented in the child class")

    def evaluate(self, X_test, y_test):
        """
        Default evaluation: computes RMSE, MAE, MAPE and SD from predictions.
        Subclasses can override if they need custom logic.
        """        
        preds = self.predict(X_test)
        
        errors = preds - y_test
        sd_err = errors.std()                                       # sd
        rmse = np.sqrt(((errors) ** 2).mean())                      # RMSE
        mae = np.abs(errors).mean()                                 # MAE
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100     # MAPE  
        
        return {'rmse': rmse, 'mae': mae, 'mape':mape, 'sd': sd_err}

    def save(self, path: str):
        """
        Save the model to disk.
        To be overridden in subclasses if serialization is needed.
        """
        raise NotImplementedError("save() must be implemented in the child class")

    @classmethod
    def load(cls, path: str):
        """
        Load a saved model from disk.
        To be overridden in subclasses if serialization is needed.
        """
        raise NotImplementedError("load() must be implemented in the child class")
