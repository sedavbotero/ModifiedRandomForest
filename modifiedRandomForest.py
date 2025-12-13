from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def sample_data(
    X: pd.DataFrame,
    y: pd.Series,
    number_of_rows: int,
    number_of_columns: int,
    probs=None,
) -> tuple[pd.DataFrame, pd.Series]:
    rows = np.random.choice(X.index.to_numpy(), number_of_rows, replace=True, p=probs)
    cols = np.random.choice(X.columns.to_numpy(), number_of_columns, replace=False)
    X_sampled = X.loc[rows, cols]
    y_sampled = y.loc[rows]
    return X_sampled, y_sampled


class ModifiedRandomForest:
    def __init__(self, n_estimators, **kwargs):
        self.n_estimators = n_estimators
        self.kwargs = kwargs
        self.estimators = []
        self.estimator_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for _ in range(self.n_estimators):
            X_sampled, y_sampled = sample_data(
                X,
                y,
                # TODO: określić jak dużo kolumn wybierać i jak dużo wierszy
                # np.ceil(np.sqrt(X.shape[0]), casting="unsafe", dtype=np.int32),
                # np.ceil(np.sqrt(X.shape[1]), casting="unsafe", dtype=np.int32),
                X.shape[0],
                X.shape[1],
            )
            tree = DecisionTreeRegressor(**self.kwargs)
            tree.fit(X_sampled, y_sampled)
            self.estimators.append(tree)
            self.estimator_features.append(X_sampled.columns.to_numpy())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = np.zeros(X.shape[0])
        for tree, features in zip(self.estimators, self.estimator_features):
            predictions += tree.predict(X.loc[:, features])
        return predictions / self.n_estimators


## Proste sprawdzenie
# if __name__ == "__main__":
#     X, y = load_diabetes(return_X_y=True, as_frame=True)
#     model = ModifiedRandomForest(100)
#     model.fit(X, y)
#     print(model.estimator_features)
#     print(model.predict(X))
#     print(mean_squared_error(y, model.predict(X)))
#     random_forest = RandomForestRegressor(n_estimators=100)
#     random_forest.fit(X, y)
#     print(mean_squared_error(y, random_forest.predict(X)))
