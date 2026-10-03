"""The fit loop: build the model matrix, and fit one component across the CV folds.

This file, with the data, split, registry and config modules, is what `score.fingerprint()`
hashes: it is the code that produces predictions. Everything downstream of a prediction (the
combiner, scoring, reporting) lives elsewhere, so editing it never invalidates a cache.
"""
import numpy as np
import pandas as pd

from harness import config, data, split


def build_matrix(cfg: config.Config, comps: dict, frame: pd.DataFrame) -> pd.DataFrame:
    for name in cfg.features:
        out = comps[name].build(frame.copy())
        if not isinstance(out, pd.DataFrame) or len(out) != len(frame):
            raise ValueError(f"feature {name!r} must return a DataFrame with the same rows")
        frame = out
    return data.to_matrix(frame)


def fit_predict(comp, params, X, y, X_test, seeds=split.SEEDS, only_fold=None):
    """Out-of-fold predictions (seeds x dev rows) and fold-averaged test predictions."""
    oof = np.full((len(seeds), len(X)), np.nan)
    test = np.zeros((len(seeds), len(X_test)))
    for s, seed in enumerate(seeds):
        for k, (tr, va) in enumerate(split.folds(y, seed)):
            if only_fold is not None and k != only_fold:
                continue
            est = comp.build(dict(params), seed)
            est.fit(X.iloc[tr], y[tr])
            oof[s, va] = est.predict_proba(X.iloc[va])[:, 1]
            test[s] += est.predict_proba(X_test)[:, 1] / split.N_FOLDS
    return oof, test
