
from sklearn.neighbors import KNeighborsRegressor
from src.models.base_model import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin




class KNNModel(BaseModel, BaseEstimator, RegressorMixin):
    """
    Wrapper around sklearn.neighbors.KNeighborsRegressor.
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
        Instantiate the KNeighborsRegressor using values from self.params.
        """
        p = self.params
        n_neighbors = p.get("n_neighbors", 5)
        weights     = p.get("weights", "uniform")
        algorithm   = p.get("algorithm", "auto")
        leaf_size   = p.get("leaf_size", 30)
        p_norm      = p.get("p", 2)

        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p_norm
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train KNN on (X_train, y_train).
        X_val and y_val are ignored but kept for interface consistency.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Return a 1D array of predictions from the trained KNN.
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
        Not implemented for KNNModel.
        """
        raise NotImplementedError("save() is not implemented for KNNModel")

    @classmethod
    def load(cls, path: str):
        """
        Not implemented for KNNModel.
        """
        raise NotImplementedError("load() is not implemented for KNNModel")
