"""Logistic regression on one-hot categoricals and scaled numerics, with fixed parameters.

A linear model makes different errors from the tree models, which is what a blend wants.
Nothing is chosen on the scored fold. Every learned step (category vocabularies, median
fills, scaling) is fit inside `fit` on training rows only. Spend-like columns (the five
bills and their total, picked by name up front) get log1p before scaling.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NAME = "logreg"
KIND = "model"
PARAMS = dict(C=1.0, max_iter=5000)

SPEND_LIKE = {"RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpend"}


def _log1p_nonneg(a):
    return np.log1p(np.clip(a, 0, None))


class _LogReg:
    """Splits columns by dtype at fit time; unseen categories encode to all-zeros."""

    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def _prep(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.cat_cols:
            out[c] = X[c].astype(object).where(X[c].notna(), "nan").astype(str)
        for c in self.spend_cols + self.num_cols:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return out

    def fit(self, X, y):
        def is_cat(c):
            return str(X[c].dtype) in ("category", "object", "bool")

        self.cat_cols = [c for c in X.columns if is_cat(c)]
        self.spend_cols = [c for c in X.columns if not is_cat(c) and c in SPEND_LIKE]
        self.num_cols = [c for c in X.columns if not is_cat(c) and c not in SPEND_LIKE]
        pre = ColumnTransformer([
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="nan")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), self.cat_cols),
            ("spend", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(_log1p_nonneg)),
                ("scale", StandardScaler()),
            ]), self.spend_cols),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), self.num_cols),
        ])
        self.model = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(**self.params, random_state=self.seed)),
        ])
        self.model.fit(self._prep(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def build(params, seed):
    return _LogReg(params, seed)
