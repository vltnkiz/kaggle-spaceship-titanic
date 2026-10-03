"""The combiner: turns the enabled models' probabilities into one probability per passenger.

    [combiner]
    method = "mean"         # default: the weighted probability average (the blend)

    mean         weighted mean of probabilities, weights from [models]
    logit_mean   weighted mean of logits, squashed back to a probability
    stack        logistic regression (C = 1.0) on the models' logits
    gate         one stack per segment; `segment` names a column of the built feature matrix,
                 and a segment with fewer than `min_rows` training rows (default 200) uses the
                 global stack

`mean` and `logit_mean` are fixed: they learn nothing. `stack` and `gate` are learned, and read
a model's weight as on/off only. A learned combiner is meta-fit **per outer fold** on the other
folds' out-of-fold predictions, then applied to the held-out fold, so a score never sees a
combiner fit on its own rows. No base model is retrained, which costs seconds over the cache.
The one accepted leak: the base models behind those out-of-fold predictions saw the held-out
fold's labels.

For the test set the combiner is fit once per seed on all dev out-of-fold predictions and applied
to that seed's fold-averaged test predictions; the caller averages over seeds.

This file is deliberately outside `score.fingerprint()`: editing a combiner never invalidates a
cached model prediction.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from harness import split

EPS = 1e-6
STACK_C = 1.0
DEFAULT_MIN_ROWS = 200

FIXED = ("mean", "logit_mean")
LEARNED = ("stack", "gate")
PARAMS = {"mean": {}, "logit_mean": {}, "stack": {}, "gate": {"segment": None, "min_rows": DEFAULT_MIN_ROWS}}
DEFAULT = {"method": "mean"}


def validate(raw: dict | None) -> dict:
    """The canonical spec for a `[combiner]` table: method plus every parameter, defaults filled
    in, so two spellings of one combiner hash to one digest."""
    raw = dict(raw or DEFAULT)
    method = raw.pop("method", None)
    if method not in PARAMS:
        raise ValueError(f"combiner.method must be one of {sorted(PARAMS)}, got {method!r}")
    unknown = set(raw) - set(PARAMS[method])
    if unknown:
        raise ValueError(f"combiner {method!r} takes no {sorted(unknown)}; it takes {sorted(PARAMS[method])}")
    spec = {"method": method, **{**PARAMS[method], **raw}}
    if method == "gate":
        if not isinstance(spec["segment"], str) or not spec["segment"]:
            raise ValueError("combiner 'gate' needs segment = \"<column of the feature matrix>\"")
        n = spec["min_rows"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError(f"combiner.min_rows must be an integer >= 2, got {n!r}")
    return spec


def is_learned(spec: dict) -> bool:
    return spec["method"] in LEARNED


def segment_values(spec: dict, matrix) -> np.ndarray | None:
    """The gate's segment label for every row of the built feature matrix, else None."""
    if spec["method"] != "gate":
        return None
    col = spec["segment"]
    if col not in matrix.columns:
        raise ValueError(f"combiner gate segment {col!r} is not a column of the built feature "
                         f"matrix (is the feature that produces it switched off?); "
                         f"columns: {sorted(matrix.columns)}")
    # str() per value: a categorical column's missing values survive astype(str) as float NaN
    return matrix[col].astype(object).map(str).to_numpy(dtype=object)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


class Fitted:
    """A combiner after its fit step. `apply` only reads probabilities (and segment labels):
    nothing here ever sees a target."""

    def __init__(self, spec: dict, weights: dict, stacks: dict | None = None):
        self.spec, self.weights, self.stacks = spec, weights, stacks or {}

    def apply(self, probs: dict, segment: np.ndarray | None = None) -> np.ndarray:
        """`probs`: {model: probabilities}, any shape for a fixed method, 1-D for a learned one."""
        method = self.spec["method"]
        if method == "mean":  # the same expression harness.score.blend always evaluated
            return sum(w * probs[m] for m, w in self.weights.items()) / sum(self.weights.values())
        if method == "logit_mean":
            z = sum(w * logit(probs[m]) for m, w in self.weights.items()) / sum(self.weights.values())
            return sigmoid(z)
        Z = np.column_stack([logit(probs[m]) for m in self.weights])
        out = self.stacks[None].predict_proba(Z)[:, 1]
        if method == "gate":
            for seg, lr in self.stacks.items():
                if seg is not None:
                    rows = segment == seg
                    if rows.any():
                        out[rows] = lr.predict_proba(Z[rows])[:, 1]
        return out


def fit(spec: dict, weights: dict, probs: dict, y: np.ndarray,
        segment: np.ndarray | None = None) -> Fitted:
    """Fit on training rows: `probs` are the models' out-of-fold probabilities there."""
    if not is_learned(spec):
        return Fitted(spec, weights)
    Z = np.column_stack([logit(probs[m]) for m in weights])
    stacks = {None: _stack(Z, y)}
    if spec["method"] == "gate":
        for seg in np.unique(segment):
            rows = segment == seg
            if rows.sum() >= spec["min_rows"] and 0 < y[rows].sum() < rows.sum():
                stacks[seg] = _stack(Z[rows], y[rows])
    return Fitted(spec, weights, stacks)


def _stack(Z: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=STACK_C, max_iter=1000).fit(Z, y)


def fold_fits(spec: dict, weights: dict, oof: dict, y: np.ndarray, seeds: tuple,
              segment: np.ndarray | None = None) -> dict:
    """{(seed, fold): Fitted}: each meta-fit on the other folds' out-of-fold rows.
    `oof[m]` has one row per entry of `seeds`, in that order."""
    out = {}
    for s, seed in enumerate(seeds):
        for k, (tr, _) in enumerate(split.folds(y, seed)):
            out[seed, k] = fit(spec, weights, {m: oof[m][s][tr] for m in weights}, y[tr],
                               None if segment is None else segment[tr])
    return out


def oof_combine(spec: dict, weights: dict, oof: dict, y: np.ndarray, seeds: tuple = split.SEEDS,
                segment: np.ndarray | None = None) -> np.ndarray:
    """Combined out-of-fold probability, seeds x dev rows, each fold scored by a combiner that
    never saw it."""
    if not is_learned(spec):
        return fit(spec, weights, oof, y).apply(oof)
    fits = fold_fits(spec, weights, oof, y, seeds, segment)
    out = np.full((len(seeds), len(y)), np.nan)
    for s, seed in enumerate(seeds):
        for k, (_, va) in enumerate(split.folds(y, seed)):
            out[s, va] = fits[seed, k].apply({m: oof[m][s][va] for m in weights},
                                            None if segment is None else segment[va])
    return out


def combine_test(spec: dict, weights: dict, oof: dict, test: dict, y: np.ndarray,
                 seeds: tuple = split.SEEDS, segment: np.ndarray | None = None,
                 test_segment: np.ndarray | None = None) -> np.ndarray:
    """Combined test probability, seeds x test rows: one combiner per seed, fit on all dev
    out-of-fold rows, applied to that seed's fold-averaged test predictions."""
    if not is_learned(spec):
        return fit(spec, weights, test, y).apply(test)
    out = []
    for s in range(len(seeds)):
        f = fit(spec, weights, {m: oof[m][s] for m in weights}, y, segment)
        out.append(f.apply({m: test[m][s] for m in weights}, test_segment))
    return np.array(out)
