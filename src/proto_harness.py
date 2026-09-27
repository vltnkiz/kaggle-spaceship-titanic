"""PROTOTYPE (throwaway) experiment harness -- ticket #2.

Answers: how is an experiment declared and compared?

    uv run python -m src.proto_harness run base lgbm        # OOF 5-fold x 3 seeds, cached
    uv run python -m src.proto_harness blend base:lgbm=0.5 base:rf=0.5
    uv run python -m src.proto_harness log                  # experiment log vs main

Declaring an experiment = add a function to FEATURE_SETS or MODELS, then name it.
Everything is cached per (feature set, model) in data/processed/oof/, so a blend
never retrains and costs well under a second.
"""
import hashlib
import inspect
import json
import sys
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import PROCESSED, RAW, ROOT, TARGET
from src.features import build_features

SEEDS = [0, 1, 2]
N_SPLITS = 5
CACHE = PROCESSED / "oof"
LOG = ROOT / "experiments" / "log.jsonl"
MAIN = ROOT / "experiments" / "main.json"  # what "delta vs main" compares against


# ---------------------------------------------------------------- feature sets
def fs_base(full):
    return build_features(full)


FEATURE_SETS = {"base": fs_base}


# ---------------------------------------------------------------------- models
# A model = fn(X_tr, y_tr, X_va, y_va, seed) -> fitted object with predict_proba.
def m_lgbm(X_tr, y_tr, X_va, y_va, seed):
    """Baseline replica: early-stops on the scoring fold (optimistic CV)."""
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1)
    return m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                 callbacks=[lgb.early_stopping(100, verbose=False)])


def m_lgbm_fixed(X_tr, y_tr, X_va, y_va, seed):
    """Same, fixed 400 trees: scoring fold never seen during training."""
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1)
    return m.fit(X_tr, y_tr)


def _codes(X):
    X = X.copy()
    for c in X.select_dtypes("category"):
        X[c] = X[c].cat.codes
    return X


class _Codes:
    def __init__(self, m): self.m = m
    def predict_proba(self, X): return self.m.predict_proba(_codes(X))


def m_rf(X_tr, y_tr, X_va, y_va, seed):
    """sklearn stand-in for the TF-DF notebook RF (see issue #6)."""
    m = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=5,
                               n_jobs=-1, random_state=seed)
    return _Codes(m.fit(_codes(X_tr), y_tr))


MODELS = {"lgbm": m_lgbm, "lgbm_fixed": m_lgbm_fixed, "rf": m_rf}


# ---------------------------------------------------------------------- engine
def load():
    train, test = pd.read_csv(RAW / "train.csv"), pd.read_csv(RAW / "test.csv")
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    return full, len(train), train[TARGET].astype(int).values


def _fingerprint(fs, model):
    src = inspect.getsource(FEATURE_SETS[fs]) + inspect.getsource(MODELS[model])
    src += inspect.getsource(build_features) + repr((SEEDS, N_SPLITS))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def acc(y, p):
    return float(((p > 0.5) == y).mean())


def run(fs, model):
    key = f"{fs}__{model}"
    path = CACHE / f"{key}__{_fingerprint(fs, model)}.npz"
    if path.exists():
        print(f"[cache] {key}")
        return path
    full, n, y = load()
    feats = FEATURE_SETS[fs](full)
    X, X_test = feats.iloc[:n], feats.iloc[n:]
    oof = np.zeros((len(SEEDS), n))
    test = np.zeros((len(SEEDS), len(X_test)))
    t0 = time.time()
    for i, seed in enumerate(SEEDS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            m = MODELS[model](X.iloc[tr], y[tr], X.iloc[va], y[va], seed)
            oof[i, va] = m.predict_proba(X.iloc[va])[:, 1]
            test[i] += m.predict_proba(X_test)[:, 1] / N_SPLITS
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, oof=oof, test=test, y=y)
    print(f"[trained] {key} in {time.time() - t0:.1f}s")
    record(key, {key: 1.0})
    return path


def _latest(key):
    return max(CACHE.glob(f"{key}__*.npz"), key=lambda p: p.stat().st_mtime)


def score(members):
    """members: {"fs__model": weight}. Blends per seed, then averages accuracy."""
    parts = {k: np.load(_latest(k)) for k in members}
    y = next(iter(parts.values()))["y"]
    w = np.array(list(members.values()), dtype=float)
    w /= w.sum()
    oof = sum(wi * parts[k]["oof"] for wi, k in zip(w, members))
    per_seed = [acc(y, o) for o in oof]
    return float(np.mean(per_seed)), per_seed


def record(name, members):
    mean, per_seed = score(members)
    main = json.loads(MAIN.read_text())["cv"] if MAIN.exists() else None
    delta = None if main is None else mean - main
    row = dict(ts=datetime.now().isoformat(timespec="seconds"), name=name,
               members=members, cv=round(mean, 5), per_seed=[round(s, 5) for s in per_seed],
               delta_vs_main=None if delta is None else round(delta, 5),
               verdict=None if delta is None else ("KEEP" if delta >= 0.002 else "drop"))
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    d = "n/a (no main.json)" if delta is None else f"{delta:+.4f} -> {row['verdict']}"
    print(f"{name:40s} CV {mean:.4f}  seeds {row['per_seed']}  vs main {d}")


def main():
    cmd, *args = sys.argv[1:]
    if cmd == "run":
        run(*args)
    elif cmd == "blend":
        members = {}
        for a in args:
            k, w = a.split("=")
            fs, model = k.split(":")
            run(fs, model)
            members[f"{fs}__{model}"] = float(w)
        record("blend " + " ".join(args), members)
    elif cmd == "set-main":  # set-main <fs:model=w ...>: pin the reference score
        members = {k.replace(":", "__"): float(w) for k, w in (a.split("=") for a in args)}
        MAIN.parent.mkdir(exist_ok=True)
        MAIN.write_text(json.dumps(dict(members=members, cv=score(members)[0]), indent=2))
        print("main =", MAIN.read_text())
    elif cmd == "log":
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            print(f"{r['ts']}  {r['name']:40s} {r['cv']:.4f}  {r['delta_vs_main']}  {r['verdict']}")


if __name__ == "__main__":
    main()
