from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax


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

        self.estimators = []
        self.estimator_features = []
        errors = []

        n_features_total = X.shape[1]
        n_features_to_select = int(np.ceil(np.sqrt(n_features_total)))
        n_rows_to_select = X.shape[0]

        for _ in range(self.n_estimators):
            X_sampled, y_sampled = sample_data(
                X,
                y,
                # TODO: określić jak dużo kolumn wybierać i jak dużo wierszy
                number_of_rows=n_rows_to_select,
                number_of_columns=n_features_to_select,
            )
            tree = DecisionTreeRegressor(**self.kwargs)
            tree.fit(X_sampled, y_sampled)
            self.estimators.append(tree)

            features = X_sampled.columns.to_numpy()
            self.estimator_features.append(features)

            predictions_train = tree.predict(X[features])
            mse = mean_squared_error(y, predictions_train)
            errors.append(mse)

        errors_array = np.array(errors)
        self.weights = softmax(-errors_array)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = np.zeros(X.shape[0])
        for tree, features in zip(self.estimators, self.estimator_features):
            predictions += tree.predict(X.loc[:, features])
        return predictions / self.n_estimators

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

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
