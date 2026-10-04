"""TabM (an MLP ensemble with shared weights) through pytabkit's `TabM_D_Classifier`.

pytabkit refuses numeric NaN, so the training fold's medians fill them in `fit` and are
re-applied in `predict_proba` (learned from training rows only). Categorical columns get a
vocabulary fixed at fit time; a level unseen at predict time falls back to "nan". pytabkit
holds out its own 20% of whatever `fit` receives for early stopping, so nothing is chosen on
the scored fold. `device` is left to pytabkit (CUDA if present, else CPU). torch and pytabkit
are imported inside `build` so registry discovery stays cheap.
"""
import numpy as np
import pandas as pd

NAME = "tabm"
KIND = "model"
PARAMS = dict(arch_type="tabm-mini", num_emb_type="pbld", tabm_k=32, lr=0.002,
              weight_decay=0.0, dropout=0.1, d_block=512, n_blocks="auto", n_epochs=1000)


class _TabM:
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def _prep(self, X):
        X = X.copy()
        for c, cats in self.categories.items():
            vals = X[c].astype(object).fillna("nan").astype(str)
            X[c] = pd.Categorical(vals.where(vals.isin(cats), "nan"), categories=cats)
        for c, med in self.medians.items():
            X[c] = X[c].astype(float).fillna(med)
        return X

    def fit(self, X, y):
        from pytabkit import TabM_D_Classifier

        cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
        self.categories = {
            c: sorted(set(X[c].astype(object).fillna("nan").astype(str)) | {"nan"})
            for c in cat_cols
        }
        # every numeric column, not just those with NaN in this fold: predict rows may have some
        self.medians = {c: float(np.nan_to_num(np.nanmedian(X[c].astype(float))))
                        for c in X.columns if c not in cat_cols}
        self.model = TabM_D_Classifier(**self.params, random_state=self.seed, verbosity=0)
        self.model.fit(self._prep(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def build(params, seed):
    return _TabM(params, seed)
