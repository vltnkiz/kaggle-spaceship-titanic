"""TabPFN-3.5, a pretrained tabular foundation model: `fit` stores the training fold as the
in-context prompt, `predict_proba` does the learning.

Needs the licence accepted and `TABPFN_TOKEN` set (set `TABPFN_NO_BROWSER=1` where nobody can
click through); the weights are downloaded once and cached by the library. `version` picks the
checkpoint: "3.5" (default), "3.5-fast" (~3x quicker), or "2" (ungated, but TabPFN-2).

Two things the harness makes necessary:
- `predict_proba` is chunked to <= 1000 rows. The harness predicts a fold's whole test set
  (~4.3k rows) in one call and a single call that size hung a 12 GB GPU for > 18 min.
- `random_state` is passed explicitly. The library defaults it to 0, not None, so without it
  every harness seed would build the same ensemble.

Categorical columns get a vocabulary fixed at fit time, unseen levels falling back to "nan"; NaN
numerics go in as they are (TabPFN takes them natively). Nothing is chosen on the scored fold.
torch and tabpfn are imported inside `fit` so registry discovery stays cheap.
"""
import numpy as np
import pandas as pd

NAME = "tabpfn"
KIND = "model"
PARAMS = dict(version="3.5", n_estimators=8)
CHUNK = 1000


class _TabPFN:
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed

    def _prep(self, X):
        X = X.copy()
        for c, cats in self.categories.items():
            vals = X[c].astype(object).fillna("nan").astype(str)
            X[c] = pd.Categorical(vals.where(vals.isin(cats), "nan"), categories=cats)
        return X

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        versions = {"3.5": ModelVersion.V3_5, "3.5-fast": ModelVersion.V3_5_FAST,
                    "2": ModelVersion.V2}
        cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
        self.categories = {
            c: sorted(set(X[c].astype(object).fillna("nan").astype(str)) | {"nan"})
            for c in cat_cols
        }
        self.model = TabPFNClassifier.create_default_for_version(
            versions[self.params["version"]], n_estimators=int(self.params["n_estimators"]),
            random_state=self.seed)
        self.model.fit(self._prep(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        X = self._prep(X)
        return np.concatenate([self.model.predict_proba(X.iloc[i:i + CHUNK])
                               for i in range(0, len(X), CHUNK)])


def build(params, seed):
    return _TabPFN(params, seed)
