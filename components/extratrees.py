"""sklearn ExtraTrees: bagged, randomised-split trees with fixed parameters.

Nothing is chosen on the scored fold. Categorical/object columns are ordinal-encoded and
numeric NaNs are median-filled, with both mappings learned in `fit` from training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

NAME = "extratrees"
KIND = "model"
PARAMS = dict(n_estimators=500, min_samples_leaf=3, max_features="sqrt")


class _ExtraTrees:
    """Learns category codes and numeric fill values at fit time; unseen categories -> -1."""

    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    @staticmethod
    def _as_str(s):
        return s.astype(object).where(s.notna(), "nan").astype(str)

    def _prep(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.columns:
            if c in self.codes:
                out[c] = self._as_str(X[c]).map(self.codes[c]).fillna(-1).astype(float)
            else:
                out[c] = pd.to_numeric(X[c], errors="coerce").astype(float).fillna(self.fills[c])
        return out.to_numpy(dtype=float)

    def fit(self, X, y):
        self.columns = list(X.columns)
        cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "bool")]
        self.codes = {
            c: {v: i for i, v in enumerate(sorted(self._as_str(X[c]).unique()))} for c in cat_cols
        }
        self.fills = {}
        for c in self.columns:
            if c not in self.codes:
                med = pd.to_numeric(X[c], errors="coerce").astype(float).median()
                self.fills[c] = 0.0 if np.isnan(med) else float(med)
        self.model = ExtraTreesClassifier(**self.params, random_state=self.seed, n_jobs=-1)
        self.model.fit(self._prep(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def build(params, seed):
    return _ExtraTrees(params, seed)
