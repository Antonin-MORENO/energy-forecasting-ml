
from sklearn.neighbors import KNeighborsRegressor
from src.models.base_model import BaseModel

class KNNModel(BaseModel):
    """
    Wrapper around sklearn.neighbors.KNeighborsRegressor.
    Implements build_model, fit, predict.
    Serialization (save/load) is not implemented here.
    """

    def __init__(self, params: dict):
        """
        params expected (examples):
          - "n_neighbors" (int)
          - "weights" (str, e.g. "uniform" or "distance")
          - "algorithm" (str, e.g. "auto", "ball_tree", "kd_tree", "brute")
          - "leaf_size" (int)
          - "p" (int: 1 for Manhattan, 2 for Euclidean, etc.)
        """
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
