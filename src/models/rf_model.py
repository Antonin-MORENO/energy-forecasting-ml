from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin

class RandomForestModel(BaseModel, BaseEstimator, RegressorMixin):
    """
    Wrapper around sklearn.ensemble.RandomForestRegressor.
    Implements build_model, fit, predict following the KNNModel skeleton.
    """

    def __init__(self, params: dict = None, **kwargs):
        """
        Initializes the model by merging a dictionary of parameters 
        with the provided keyword arguments.
        """
        if params is None:
            params = {}

        params.update(kwargs)
        super().__init__(params)
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Instantiates the RandomForestRegressor using values from self.params.
        """
        p = self.params
        # Get Random Forest specific hyperparameters
        n_estimators = p.get("n_estimators", 100)
        max_depth    = p.get("max_depth", None)
        min_samples_split = p.get("min_samples_split", 2)
        min_samples_leaf = p.get("min_samples_leaf", 1)
        random_state = p.get("random_state", 42)
        max_features = p.get("max_features", 1.0) 

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            max_features=max_features
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the Random Forest on (X_train, y_train).
        X_val and y_val are ignored but kept for interface consistency.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Returns a 1D array of predictions from the trained Random Forest.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Returns hyperparameters for GridSearchCV and clone().
        """
        return self.params.copy()

    def set_params(self, **new_params):
        """
        Updates hyperparameters, rebuilds the model, and returns self.
        """
        self.params.update(new_params)
        self.build_model()
        return self

    def save(self, path: str):
        """
        Not implemented for RandomForestModel.
        """
        raise NotImplementedError("save() is not implemented for RandomForestModel")

    @classmethod
    def load(cls, path: str):
        """
        Not implemented for RandomForestModel.
        """
        raise NotImplementedError("load() is not implemented for RandomForestModel")