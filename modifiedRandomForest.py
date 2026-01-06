# from sklearn.tree import DecisionTreeRegressor
from decisionTree import DecisionTreeRegressor

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


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
    def __init__(self, n_estimators, gamma=1.0, max_exponent=10.0, **kwargs):
        self.n_estimators = n_estimators  # hiperparametr
        self.gamma = gamma  # hiperparametr
        self.max_exponent = max_exponent  # hiperparametr
        self.kwargs = kwargs
        self.estimators = []
        self.estimator_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:

        self.estimators = []
        self.estimator_features = []

        n_features_total = X.shape[1]
        n_features_to_select = n_features_total
        n_rows_to_select = X.shape[0]

        errors = None
        weights = np.ones(n_rows_to_select) / n_rows_to_select

        ex = 0.001
        for _ in range(self.n_estimators):
            print(f"fitting {_}")
            ex = ex * 1.01**self.gamma if ex < self.max_exponent else ex
            if errors is not None:
                weights = errors**ex
                weights /= np.sum(weights)
            if np.any(np.isnan(weights)):
                print(f"weights contains NaN: {weights}")
                print(f"errors contains NaN: {errors}")
            X_sampled, y_sampled = sample_data(
                X,
                y,
                number_of_rows=n_rows_to_select,
                number_of_columns=n_features_to_select,
                probs=weights,
            )
            tree = DecisionTreeRegressor(**self.kwargs)
            tree.fit(X_sampled, y_sampled)
            self.estimators.append(tree)

            features = X_sampled.columns.to_numpy()
            self.estimator_features.append(features)

            predictions_train = self.predict(X)
            errors = np.absolute(y - predictions_train)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = np.zeros(X.shape[0])
        for tree, features in zip(self.estimators, self.estimator_features):
            predictions += tree.predict(X.loc[:, features])
        return predictions / self.n_estimators

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


## Proste sprawdzenie
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


def main():
    # X, y = load_diabetes(return_X_y=True, as_frame=True)

    arff_file = arff.loadarff("dataset_2204_house_8L.arff")
    df = pd.DataFrame(arff_file[0])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # print(X)
    # print(y)

    def cross_val_score(modelClass, **param_kwargs):
        splitter = KFold(n_splits=5, shuffle=True)
        scores = []
        for i, (train_index, test_index) in enumerate(splitter.split(X, y)):
            model = modelClass(**param_kwargs)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = model.predict(X.iloc[test_index])
            score = r2_score(y.iloc[test_index], y_pred)
            scores.append(score)
        return sum(scores) / len(scores)

    params = {
        "n_estimators": 10,
        # "gamma": best_params["gamma"],
        "gamma": 2,
        # "max_exponent": best_params["max_exponent"],
        "max_exponent": 90,
    }
    model = ModifiedRandomForest(**params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    random_forest = RandomForestRegressor(n_estimators=params["n_estimators"])
    random_forest.fit(X_train, y_train)
    print("train scores:")
    print("my mse", mean_squared_error(y_train, model.predict(X_train)))
    print("sklearn mse", mean_squared_error(y_train, random_forest.predict(X_train)))
    print("my r2", r2_score(y_train, model.predict(X_train)))
    print("sklearn r2", r2_score(y_train, random_forest.predict(X_train)))
    print("test scores:")
    print("my mse", mean_squared_error(y_test, model.predict(X_test)))
    print("sklearn mse", mean_squared_error(y_test, random_forest.predict(X_test)))
    print("my r2", r2_score(y_test, model.predict(X_test)))
    print("sklearn r2", r2_score(y_test, random_forest.predict(X_test)))
    # print(
    #     "cross val score my rf",
    #     cross_val_score(ModifiedRandomForest, **params),
    # )
    # print(
    #     "cross val score my rf",
    #     cross_val_score(RandomForestRegressor, n_estimators=params["n_estimators"]),
    # )


if __name__ == "__main__":
    main()
