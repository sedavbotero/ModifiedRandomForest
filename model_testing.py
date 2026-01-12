## Proste sprawdzenie
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from modifiedRandomForest import ModifiedRandomForest
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pathlib import Path

import urllib.request


def main():
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    filepath = Path("dataset_2204_house_8L.arff")
    if not filepath.exists():
        print(
            urllib.request.urlretrieve(
                "https://www.openml.org/data/download/3655/dataset_2204_house_8L.arff",
                filepath,
            )
        )

    arff_file = arff.loadarff(filepath)
    df = pd.DataFrame(arff_file[0])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # print(X)
    # print(y)

    def cross_val_score(modelClass, **param_kwargs):
        splitter = KFold(n_splits=5, shuffle=True)
        scores = []
        for i, (train_index, test_index) in enumerate(splitter.split(X, y)):
            # model = modelClass(**param_kwargs)
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
