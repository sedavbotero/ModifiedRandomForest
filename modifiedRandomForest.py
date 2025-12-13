from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


def sample_data(
    X: pd.DataFrame,
    y: pd.Series,
    number_of_rows: int,
    number_of_columns: int,
    probs=None,
) -> tuple[pd.DataFrame, pd.Series]:
    rows = np.random.choice(
        np.arange(X.shape[0]), number_of_rows, replace=True, p=probs
    )
    cols = np.random.choice(np.arange(X.shape[1]), number_of_columns, replace=False)
    X_sampled = X.iloc[rows, cols]
    y_sampled = y.iloc[rows]
    return X_sampled, y_sampled


class ModifiedRandomForest:
    def __init__(self, n_estimators, **kwargs):
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.estimators = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for _ in range(self.n_estimators):
            X_sampled, y_sampled = sample_data(
                X, y, np.sqrt(X.shape[0]), np.sqrt(X.shape[1])
            )
            tree = DecisionTreeRegressor(**self.kwargs)
            tree.fit(X_sampled, y_sampled)
            self.estimators.append(tree)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass
