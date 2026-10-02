"""CatBoost plus the Transported rate of each row's labelled travel-group and surname mates.

Four columns are added: GroupMateRate / GroupMateN (other rows with the same PassengerId
prefix) and SurnameMateRate / SurnameMateN (other rows with the same Name surname). N counts
labelled mates; the rate is NaN when N is 0. Keys come from components/mate_keys.py and are
popped out of X, so CatBoost never sees them.

Leakage guarantees:
- Labels come only from the rows `fit` receives (the training fold). Their keys and labels
  are stored in `fit`; `predict_proba` looks mates up among those rows only, so a scored or
  test row's own label is never used.
- A training row's own label never enters its own stats: `fit` splits its rows into 5 inner
  stratified folds (shuffled, random_state=seed) and computes each row's stats from the rows
  in the other 4 inner folds only (inner out-of-fold, not leave-one-out).
- A missing surname (NaN, or the string "nan" if to_matrix stringified it) has no surname mates.
CatBoost itself is built exactly as components/catboost.py builds it.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from components import catboost

NAME = "mates_cb"
KIND = "model"
PARAMS = dict(catboost.PARAMS)

INNER_FOLDS = 5
KEYS = {"Group": "MateGroup", "Surname": "MateSurname"}  # stat prefix -> key column
MISSING = {"nan", "None", ""}


def _split_keys(X):
    """(X without the key columns, {prefix: key array, missing -> NaN})."""
    keys = {}
    for prefix, col in KEYS.items():
        k = X[col].astype(object).astype(str)
        keys[prefix] = k.where(~k.isin(MISSING)).to_numpy()
    return X.drop(columns=list(KEYS.values())), keys


def _stats(query, ref, y_ref):
    """For each query key: (mean y over ref rows with that key or NaN, their count)."""
    table = pd.Series(y_ref, dtype=float).groupby(pd.Series(ref)).agg(["sum", "count"])
    q = pd.Series(query)
    s = q.map(table["sum"]).fillna(0).to_numpy()
    n = q.map(table["count"]).fillna(0).to_numpy()  # missing / unseen keys -> 0 mates
    rate = np.full(len(q), np.nan)
    rate[n > 0] = s[n > 0] / n[n > 0]
    return rate, n


def _with(X, cols):
    X = X.copy()
    for name, v in cols.items():
        X[name] = v
    return X


class _MatesCatBoost:
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        X, keys = _split_keys(X)
        self.ref_keys_, self.ref_y_ = keys, y
        cols = {f"{p}Mate{s}": np.full(len(y), np.nan) for p in KEYS for s in ("Rate", "N")}
        skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(np.zeros(len(y)), y):
            for p, k in keys.items():
                cols[f"{p}MateRate"][va], cols[f"{p}MateN"][va] = _stats(k[va], k[tr], y[tr])
        self.model = catboost.build(dict(self.params), self.seed).fit(_with(X, cols), y)
        return self

    def _features(self, X):
        X, keys = _split_keys(X)
        cols = {}
        for p, k in keys.items():
            cols[f"{p}MateRate"], cols[f"{p}MateN"] = _stats(k, self.ref_keys_[p], self.ref_y_)
        return _with(X, cols)

    def predict_proba(self, X):
        return self.model.predict_proba(self._features(X))


def build(params, seed):
    return _MatesCatBoost(params, seed)
