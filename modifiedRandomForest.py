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
    probs: np.typing.ArrayLike | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Funkcja odpowiedzialna za losowe próbkowanie zbioru danych.
    :param X: Ramka danych z cechami
    :param y: zmienna celu
    :param number_of_rows: liczba wierszy w X
    :param number_of_columns: liczba kolumn w X
    :param probs: dyskretny rozkład prawdopodobieństwa na wierszach X.
     Jeśli None to zostanie użyty rozkład jednostajny. Patrz argument p w numpy.random.choice
    :return: Zwraca spróbkowane X, y
    """
    rows = np.random.choice(X.index.to_numpy(), number_of_rows, replace=True, p=probs)
    cols = np.random.choice(X.columns.to_numpy(), number_of_columns, replace=False)
    X_sampled = X.loc[rows, cols]
    y_sampled = y.loc[rows]
    return X_sampled, y_sampled


class ModifiedRandomForest:
    """
    Zmodyfikowana wersja algorytmu lasu losowego.
    Modyfikacja polega na zmodyfikowaniu bootstrapu tak, aby prawdopodobieństwo
    wylosowania n-tego wiersza jest proporcjonalne do err^ex, gdzie err to
    wartość bezwzględna n-tego residuum.
    ex jest wykładnikiem, który startuje z wartością 0,001 i jest w każdej iteracji zwiększana
    1.01^gamma-krotnie, aż osiągnie wartość max_exponent, gdzie gamma i max_exponent
    to hiperparametry modelu
    """

    def __init__(self, n_estimators, gamma=1.0, max_exponent=10.0, **kwargs):
        """
        :param n_estimators: liczba drzew decyzyjnych wytrenowanych w modelu
        :param gamma: hiperparametr użyty przy trenowaniu
        :param max_exponent: hiperparametr użyty przy trenowaniu
        :param kwargs: argumenty przekazywane do konstruktora drzew decyzyjnych
        """
        self.n_estimators = n_estimators  # hiperparametr
        self.gamma = gamma  # hiperparametr
        self.max_exponent = max_exponent  # hiperparametr
        self.kwargs = kwargs
        self.estimators = []
        self.estimator_features = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Metoda dopasowująca model do danych
        :param X: Ramka danych z cechami
        :param y: Zmienna celu
        """
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
        """
        :param X: Ramka danych z cechami
        :return: Predykcje dla danych X
        """
        predictions = np.zeros(X.shape[0])
        for tree, features in zip(self.estimators, self.estimator_features):
            predictions += tree.predict(X.loc[:, features])
        return predictions / self.n_estimators

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Metoda licząca metrykę R^2 dla wytrenowanego modelu
        :param X: Ramka danych z cechami
        :param y: Zmienna celu
        :return: Wartość metryki R^2
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
