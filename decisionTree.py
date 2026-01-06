import numpy as np

from sklearn.metrics import mean_squared_error


class DecisionNode:
    def __init__(self, feature, split_point):
        self.feature = feature
        self.split_point = split_point
        self.left = None
        self.right = None
        self.leaf_value = None
        self.drop_column = None

    @staticmethod
    def make_leaf(value) -> "DecisionNode":
        dn = DecisionNode(None, None)
        dn.leaf_value = value
        return dn


def parse_tree(node: DecisionNode, x: np.ndarray):
    if node.leaf_value is not None:
        return node.leaf_value
    new_x = np.delete(x, node.drop_column) if node.drop_column is not None else x
    if node.split_point >= x[node.feature]:
        return parse_tree(node.left, new_x)
    else:
        return parse_tree(node.right, new_x)


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_bins: int | float = 100,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins

    def recursively_build_decision_tree(
        self, X: np.ndarray, y: np.ndarray, histograms: list[np.ndarray], depth: int = 0
    ) -> DecisionNode | None:
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(histograms)
        best_split = {
            "feature": None,
            "split_point": None,
            "impurity": np.inf,
            "mask": None,
        }
        if (
            X.shape[0] < self.min_samples_split
            or X.shape[1] == 0
            or (self.max_depth and depth < self.max_depth)
        ):
            return DecisionNode.make_leaf(np.mean(y))
        this_impurity = mean_squared_error(y, np.repeat(np.mean(y), y.shape[0]))
        for feature in range(X.shape[1]):
            x = X[:, feature]
            for split_point in histograms[feature]:
                mask = x <= split_point
                if (
                    np.sum(mask) < self.min_samples_leaf
                    or np.sum(~mask) < self.min_samples_leaf
                ):
                    continue
                left_mean = np.mean(y[mask])
                right_mean = np.mean(y[~mask])
                y_hat = np.where(mask, left_mean, right_mean)
                impurity = mean_squared_error(y, y_hat)
                if impurity < best_split["impurity"]:
                    best_split["feature"] = feature
                    best_split["split_point"] = split_point
                    best_split["impurity"] = impurity
                    best_split["mask"] = mask
        if best_split["feature"] is None:
            return DecisionNode.make_leaf(np.mean(y))
        if this_impurity - best_split["impurity"] < self.min_impurity_decrease:
            return DecisionNode.make_leaf(np.mean(y))
        node = DecisionNode(
            feature=best_split["feature"], split_point=best_split["split_point"]
        )
        X_left = X[best_split["mask"], :]
        y_left = y[best_split["mask"]]
        X_right = X[~best_split["mask"], :]
        y_right = y[~best_split["mask"]]
        histograms_split = histograms.copy()
        tmp = histograms_split[best_split["feature"]]
        histograms_split[best_split["feature"]] = np.array(
            [el for el in tmp if el != best_split["split_point"]]
        )
        if len(histograms_split[best_split["feature"]]) == 0:
            X_left = np.delete(X_left, best_split["feature"], axis=1)
            X_right = np.delete(X_right, best_split["feature"], axis=1)
            histograms_split.pop(best_split["feature"])
            node.drop_column = best_split["feature"]
        node.left = self.recursively_build_decision_tree(
            X_left, y_left, histograms_split, depth + 1
        )
        node.right = self.recursively_build_decision_tree(
            X_right, y_right, histograms_split, depth + 1
        )
        return node

    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike):
        X = np.array(X)
        y = np.array(y)
        assert y.ndim == 1
        assert X.ndim == 2
        assert X.shape[0] == y.shape[0]
        histograms: list[np.ndarray] = []
        max_bins = (
            self.max_bins
            if type(self.max_bins) is int
            else np.ceil(X.shape[0] / self.max_bins)
        )
        for feature in range(X.shape[1]):
            bins = len(np.unique(X[:, feature]))
            bins = min(bins, max_bins)
            histograms.append(np.histogram_bin_edges(X[:, feature], bins)[:-1])
        dt = self.recursively_build_decision_tree(X, y, histograms)
        self.model = dt

    def predict(self, X: np.typing.ArrayLike) -> np.ndarray:
        X = np.array(X)
        assert X.ndim == 2
        preds = np.array([parse_tree(self.model, X[i, :]) for i in range(X.shape[0])])
        assert len(preds) == X.shape[0]
        return preds


# Proste sprawdzenie
# from sklearn.datasets import load_diabetes
# if __name__ == "__main__":
#     diab = load_diabetes()
#     dt = DecisionTreeRegressor(max_bins=0.2)
#     dt.fit(diab.data, diab.target)
#     print("fitted")
#     y_pred = dt.predict(diab.data)
#     print("mse mine", r2_score(diab.target, y_pred))
#     # print(diab.target - y_pred)
#     dtsk = tree.DecisionTreeRegressor()
#     dtsk.fit(diab.data, diab.target)
#     y_pred_dtsk = dtsk.predict(diab.data)
#     print("mse sklearn", r2_score(diab.target, y_pred_dtsk))
