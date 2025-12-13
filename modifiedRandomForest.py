from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


def sample_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    pass


class ModifiedRandomForest:
    def __init__(self, n_estimators, **kwargs):
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.estimators = []
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for _ in range(self.n_estimators):
            X_sampled, y_sampled = sample_data(X, y)
            tree = DecisionTreeRegressor(**self.kwargs)
            tree.fit(X_sampled, y_sampled)
            self.estimators.append(tree)
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
