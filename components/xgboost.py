"""XGBoost with a fixed number of rounds and native categorical handling.

Nothing is chosen on the scored fold: no eval_set, no early stopping.
"""
import pandas as pd
from xgboost import XGBClassifier

NAME = "xgboost"
KIND = "model"
PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=5,
              subsample=0.8, colsample_bytree=0.8)


class _XGBoost:
    """Fixes each categorical column's categories at fit time so codes match in predict."""

    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def _prep(self, X):
        X = X.copy()
        for c, cats in self.categories.items():
            X[c] = pd.Categorical(X[c].astype(object).fillna("nan").astype(str), categories=cats)
        return X

    def fit(self, X, y):
        cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
        self.categories = {
            c: sorted(X[c].astype(object).fillna("nan").astype(str).unique()) for c in cat_cols
        }
        self.model = XGBClassifier(**self.params, random_state=self.seed, tree_method="hist",
                                   enable_categorical=True, n_jobs=-1, verbosity=0)
        self.model.fit(self._prep(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def build(params, seed):
    return _XGBoost(params, seed)
