"""CatBoost with a fixed number of iterations and native categorical handling.

Nothing is chosen on the scored fold: no eval_set, no early stopping.
"""
from catboost import CatBoostClassifier

NAME = "catboost"
KIND = "model"
PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0)


class _CatBoost:
    """Tells CatBoost which columns are categorical and feeds them as strings."""

    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).fillna("nan").astype(str)
        return X

    def fit(self, X, y):
        self.cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
        self.model = CatBoostClassifier(**self.params, random_seed=self.seed, verbose=0,
                                        allow_writing_files=False, cat_features=self.cat_cols)
        self.model.fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def build(params, seed):
    return _CatBoost(params, seed)
