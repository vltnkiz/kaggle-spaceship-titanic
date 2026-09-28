"""CatBoost with a decision threshold tuned by nested CV inside the training fold.

`fit` splits the rows it is given into 3 inner stratified folds, fits CatBoost on each inner
train part, and picks the threshold t (grid 0.40..0.60, step 0.01; ties go to the one nearest
0.50, then the smaller) that maximises accuracy on the inner out-of-fold probabilities. The
final CatBoost is then fit on all the rows `fit` was given. `predict_proba` shifts the final
model's p in logit space by -logit(t), so the harness's fixed 0.5 cut equals cutting p at t.

Nothing is chosen on the scored fold: the threshold comes only from rows `fit` receives.
CatBoost itself is built exactly as components/catboost.py builds it.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from components import catboost

NAME = "threshold_cb"
KIND = "model"
PARAMS = dict(catboost.PARAMS)

INNER_FOLDS = 3
GRID = [c / 100 for c in range(40, 61)]  # 0.40, 0.41, ..., 0.60
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _pick_threshold(p, y):
    """Most accurate cut on the grid; ties -> nearest 0.50, then the smaller."""
    scored = [(-float(((p > t) == y).mean()), abs(t - 0.5), t) for t in GRID]
    return min(scored)[2]


class _ThresholdCatBoost:
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        inner = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(np.zeros(len(y)), y):
            m = catboost.build(dict(self.params), self.seed).fit(X.iloc[tr], y[tr])
            inner[va] = m.predict_proba(X.iloc[va])[:, 1]
        self.threshold_ = _pick_threshold(inner, y)
        self.model = catboost.build(dict(self.params), self.seed).fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        q = 1 / (1 + np.exp(-(_logit(p) - _logit(self.threshold_))))
        return np.column_stack([1 - q, q])


def build(params, seed):
    return _ThresholdCatBoost(params, seed)
