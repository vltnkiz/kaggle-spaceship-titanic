"""Experiment harness -- ticket #8, interface settled in issue #2.

Declaring an experiment is config: a model = a TOML file under
experiments/models/ naming a feature set (a Python function) and a kind
(a Python trainer, src/models.py) plus its params. Everything is cached
per (model file content) in data/processed/oof/, keyed by a content hash,
so blends and reruns never retrain.

    uv run python -m src.harness run experiments/models/lgbm.toml
    uv run python -m src.harness blend experiments/blends/my_blend.toml
    uv run python -m src.harness blend lgbm=0.5 rf=0.5
    uv run python -m src.harness blend lgbm=0.5 rf=0.5 --grid
    uv run python -m src.harness submit experiments/main.toml
    uv run python -m src.harness set-main experiments/main.toml
    uv run python -m src.harness log
"""
import argparse
import hashlib
import inspect
import itertools
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import ID, PROCESSED, RAW, ROOT, TARGET
from src.features import FEATURE_SETS, build_features
from src.models import MODEL_KINDS

SEEDS = [0, 1, 2]
N_SPLITS = 5
KEEP_THRESHOLD = 0.002

MODELS_DIR = ROOT / "experiments" / "models"
BLENDS_DIR = ROOT / "experiments" / "blends"
CACHE = PROCESSED / "oof"
LOG = ROOT / "experiments" / "log.jsonl"
MAIN = ROOT / "experiments" / "main.json"


# --------------------------------------------------------------------- data
def load_data():
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    return full, len(train), train[TARGET].astype(int).values, test[ID]


def acc(y, p):
    return float(((p > 0.5) == y).mean())


# ------------------------------------------------------------------- models
def load_model_config(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def fingerprint(config: dict) -> str:
    feats_src = inspect.getsource(FEATURE_SETS[config["features"]])
    if config["features"] == "base":
        feats_src += inspect.getsource(build_features)
    payload = json.dumps(
        {"features": feats_src, "kind": config["kind"], "params": config["params"],
         "seeds": SEEDS, "splits": N_SPLITS},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def cache_path(model_id: str, config: dict) -> Path:
    return CACHE / f"{model_id}__{fingerprint(config)}.npz"


def train_model(model_id: str) -> Path:
    """Run 5-fold x 3-seed CV for one model TOML, caching OOF + test predictions."""
    config = load_model_config(MODELS_DIR / f"{model_id}.toml")
    path = cache_path(model_id, config)
    if path.exists():
        print(f"[cache] {model_id}")
        return path

    full, n, y, _ = load_data()
    feats = FEATURE_SETS[config["features"]](full)
    X, X_test = feats.iloc[:n], feats.iloc[n:]
    trainer = MODEL_KINDS[config["kind"]]
    params = config["params"]

    oof = np.zeros((len(SEEDS), n))
    test_pred = np.zeros((len(SEEDS), len(X_test)))
    for i, seed in enumerate(SEEDS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            m = trainer(X.iloc[tr], y[tr], X.iloc[va], y[va], seed, params)
            oof[i, va] = m.predict_proba(X.iloc[va])[:, 1]
            test_pred[i] += m.predict_proba(X_test)[:, 1] / N_SPLITS

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, oof=oof, test=test_pred, y=y)
    return path


def latest_cache(model_id: str) -> Path:
    matches = list(CACHE.glob(f"{model_id}__*.npz"))
    if not matches:
        raise FileNotFoundError(f"no cached run for {model_id!r}; run it first")
    return max(matches, key=lambda p: p.stat().st_mtime)


# -------------------------------------------------------------------- blend
def score(members: dict[str, float]):
    """members: {model_id: weight}. Blends OOF per seed, averages accuracy."""
    parts = {k: np.load(latest_cache(k)) for k in members}
    y = next(iter(parts.values()))["y"]
    w = np.array(list(members.values()), dtype=float)
    w = w / w.sum()
    oof = sum(wi * parts[k]["oof"] for wi, k in zip(w, members))
    per_seed = [acc(y, oof[i]) for i in range(oof.shape[0])]
    return float(np.mean(per_seed)), per_seed


def blend_test(members: dict[str, float]):
    parts = {k: np.load(latest_cache(k)) for k in members}
    w = np.array(list(members.values()), dtype=float)
    w = w / w.sum()
    test = sum(wi * parts[k]["test"] for wi, k in zip(w, members))
    return test.mean(axis=0)


def grid_search(members: list[str], step: float = 0.1):
    """All weight combinations over `step`-sized simplex points, seeds averaged."""
    n = len(members)
    steps = round(1 / step)
    best = None
    for combo in itertools.product(range(steps + 1), repeat=n):
        if sum(combo) != steps:
            continue
        weights = {m: c * step for m, c in zip(members, combo) if c > 0}
        if len(weights) < 2:
            continue
        mean, per_seed = score(weights)
        if best is None or mean > best[1]:
            best = (weights, mean, per_seed)
    return best


# --------------------------------------------------------------------- log
def log_run(name: str, members: dict[str, float]):
    mean, per_seed = score(members)
    main_cv = json.loads(MAIN.read_text())["cv"] if MAIN.exists() else None
    delta = None if main_cv is None else mean - main_cv
    verdict = None if delta is None else ("KEEP" if delta >= KEEP_THRESHOLD else "drop")
    row = dict(
        ts=datetime.now().isoformat(timespec="seconds"), name=name, members=members,
        cv=round(mean, 5), per_seed=[round(s, 5) for s in per_seed],
        delta_vs_main=None if delta is None else round(delta, 5), verdict=verdict,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    d = "n/a (no main.json)" if delta is None else f"{delta:+.4f} -> {verdict}"
    print(f"{name:40s} CV {mean:.4f}  seeds {row['per_seed']}  vs main {d}")
    return row


def parse_weight_args(args: list[str]) -> dict[str, float]:
    members = {}
    for a in args:
        k, w = a.split("=")
        members[k] = float(w)
    return members


# --------------------------------------------------------------------- CLI
def cmd_run(args):
    for model_id in args.models:
        train_model(model_id)
        log_run(model_id, {model_id: 1.0})


def cmd_blend(args):
    if len(args.spec) == 1 and args.spec[0].endswith(".toml"):
        raw = args.spec[0]
        path = Path(raw) if Path(raw).exists() else BLENDS_DIR / raw
        members = tomllib.loads(path.read_text())["weights"]
        name = path.stem
    else:
        members = parse_weight_args(args.spec)
        name = "blend " + " ".join(args.spec)

    for model_id in members:
        train_model(model_id)

    if args.grid:
        weights, mean, per_seed = grid_search(list(members))
        print(f"[grid] best weights {weights} -> CV {mean:.4f}")
        log_run(name + " (grid)", weights)
    else:
        log_run(name, members)


def cmd_submit(args):
    config = tomllib.loads(Path(args.toml).read_text())
    members = config["weights"] if "weights" in config else {Path(args.toml).stem: 1.0}
    for model_id in members:
        train_model(model_id)
    _, n, _, test_ids = load_data()
    test_pred = blend_test(members)
    out = ROOT / "submissions" / f"{Path(args.toml).stem}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({ID: test_ids, TARGET: test_pred > 0.5}).to_csv(out, index=False)
    print("Wrote", out)


def cmd_set_main(args):
    config = tomllib.loads(Path(args.toml).read_text())
    members = config["weights"] if "weights" in config else {Path(args.toml).stem: 1.0}
    for model_id in members:
        train_model(model_id)
    mean, per_seed = score(members)
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    MAIN.write_text(json.dumps(
        dict(members=members, cv=round(mean, 5), per_seed=[round(s, 5) for s in per_seed]),
        indent=2,
    ) + "\n")
    print("Wrote", MAIN, "cv =", round(mean, 5))


def cmd_log(args):
    if not LOG.exists():
        print("no runs logged yet")
        return
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        print(f"{r['ts']}  {r['name']:40s} {r['cv']:.4f}  {r['delta_vs_main']}  {r['verdict']}")


def main():
    p = argparse.ArgumentParser(prog="harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="train+cache one or more models by id")
    p_run.add_argument("models", nargs="+")
    p_run.set_defaults(func=cmd_run)

    p_blend = sub.add_parser("blend", help="score a blend: a .toml filename or model=weight pairs")
    p_blend.add_argument("spec", nargs="+")
    p_blend.add_argument("--grid", action="store_true", help="search weights in 0.1 steps")
    p_blend.set_defaults(func=cmd_blend)

    p_submit = sub.add_parser("submit", help="write a submission CSV from a model/blend toml")
    p_submit.add_argument("toml")
    p_submit.set_defaults(func=cmd_submit)

    p_main = sub.add_parser("set-main", help="pin experiments/main.json from a model/blend toml")
    p_main.add_argument("toml")
    p_main.set_defaults(func=cmd_set_main)

    p_log = sub.add_parser("log", help="print the experiment log")
    p_log.set_defaults(func=cmd_log)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
