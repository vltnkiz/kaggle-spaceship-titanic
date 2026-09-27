"""Model kinds for the experiment harness.

A kind is fn(X_tr, y_tr, X_va, y_va, seed, params) -> fitted object with
predict_proba(X)[:, 1]. No model sees its scoring fold: fixed rounds or
an inner split carved from the training fold, never eval_X=X_va.
"""
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def _codes(X):
    X = X.copy()
    for c in X.select_dtypes("category"):
        X[c] = X[c].cat.codes
    return X


class _NumericProxy:
    """Wraps a model that only sees numeric-coded categoricals."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(_codes(X))


def k_lgbm(X_tr, y_tr, X_va, y_va, seed, params):
    m = lgb.LGBMClassifier(random_state=seed, verbose=-1, **params)
    return m.fit(X_tr, y_tr)


def _fill_cats(X):
    X = X.copy()
    for c in X.select_dtypes("category"):
        X[c] = X[c].cat.add_categories(["__nan__"]).fillna("__nan__")
    return X


class _CatBoostProxy:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(_fill_cats(X))


def k_catboost(X_tr, y_tr, X_va, y_va, seed, params):
    cat_cols = list(X_tr.select_dtypes("category").columns)
    m = cb.CatBoostClassifier(random_seed=seed, verbose=False, cat_features=cat_cols, **params)
    return _CatBoostProxy(m.fit(_fill_cats(X_tr), y_tr))


def k_xgb(X_tr, y_tr, X_va, y_va, seed, params):
    m = xgb.XGBClassifier(random_state=seed, enable_categorical=True, **params)
    return m.fit(X_tr, y_tr)


def k_rf(X_tr, y_tr, X_va, y_va, seed, params):
    m = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    return _NumericProxy(m.fit(_codes(X_tr), y_tr))


MODEL_KINDS = {"lgbm": k_lgbm, "catboost": k_catboost, "xgb": k_xgb, "rf": k_rf}
