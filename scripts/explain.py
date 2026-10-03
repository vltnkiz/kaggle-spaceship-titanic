"""Explain what a config's models rely on: permutation importance on the validation folds.

    python -m scripts.explain CONFIG [--against CONFIG2] [--seeds 0]
    python -m scripts.explain [CONFIG] --smoke       CONFIG defaults to main (CI also passes a combiner)

Descriptive only, like the holdout reading: an explanation never labels, ranks or chooses a
config (CONTEXT.md's **Explanation**), so this script prints no verdict and writes nowhere a
decision reads from -- not `results/*.json`, `configs/` or `.batch/`, only
`results/explain/<config>-<digest>.json` (the digest is the one the config's run JSON carries).

Method, settled in kaggle-spaceship-titanic#67: for each model with weight > 0 (plus `blend`,
the weighted average the harness scores, when there are two or more), fit on each training
fold, then shuffle one group of validation-fold columns at a time and record how much the
log-loss rises and the accuracy falls. Groups are per **component** -- every column a feature
component adds, shuffled together with one shared row permutation so the group stays
internally consistent -- with raw columns credited to `base`, plus a per-column drill-down.

Uses only the component contract (`build(params, seed)` -> `fit` / `predict_proba`), so any
model added as an idea is explained with no extra code. Reuses the harness's own building
blocks (`config.load`, `registry.discover`, `data.load_dev`/`to_matrix`, `split.folds`) and
never edits `harness/`. Seed 0 by default: reusing a CV seed leaks nothing, since nothing here
decides anything, and it explains the very fold models the CV score came from.
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from harness import combiner, config, data, paths, registry, split
from harness.fit import fit_predict
from harness.score import cache_key, digest, predictions

BASE = "base"
BLEND = "blend"
SEEDS = (0,)
REPEATS = 5
EPS = 1e-15
SMOKE_ROWS = 300
OUT = paths.RESULTS / "explain"


def groups(cfg: config.Config, comps: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """The model matrix `build_matrix` would give, each of its columns credited to the
    component that first added it (raw columns to `base`), and a note per component that
    rewrote columns instead of adding them -- credit stays with the creator."""
    owner = {c: BASE for c in frame.columns}
    notes = []
    for name in cfg.features:
        out = comps[name].build(frame.copy())
        if not isinstance(out, pd.DataFrame) or len(out) != len(frame):
            raise ValueError(f"feature {name!r} must return a DataFrame with the same rows")
        rewritten = {}
        for c in out.columns:
            if c not in owner:
                owner[c] = name
            elif c in frame.columns and not out[c].equals(frame[c]):
                rewritten.setdefault(owner[c], []).append(c)
        notes += [f"{name} rewrites {', '.join(cols)}, credited to {creator}"
                  for creator, cols in rewritten.items()]
        frame = out
    X = data.to_matrix(frame)
    grouped = {}
    for c in X.columns:
        grouped.setdefault(owner[c], []).append(c)
    return X, grouped, notes


def explain(cfg: config.Config, comps: dict, X: pd.DataFrame, y: np.ndarray, grouped: dict,
            seeds: tuple = SEEDS, repeats: int = REPEATS, only_fold: int | None = None,
            oof: dict | None = None) -> dict:
    """{"component" | "column": {subject: {group: {"logloss": [mean, std], "accuracy": [mean, std]}}}}

    A subject is each model with weight > 0, plus `blend` when there are two or more. One
    reading = one shuffle of one group on one validation fold: the rise in log-loss and the
    fall in accuracy against that fold's unshuffled predictions. Every subject sees the same
    shuffles.

    `blend` is the config's whole pipeline, combiner included. A learned combiner is fit per
    validation fold on the other folds' out-of-fold predictions (`oof[m]`, one row per entry of
    `seeds`), exactly as scoring does, then applied to the shuffled fold's predictions; a `gate`
    combiner reads its segment column from the shuffled matrix too."""
    spec = cfg.combiner
    if combiner.is_learned(spec) and oof is None:
        raise ValueError(f"combiner {spec['method']!r} needs `oof`, the models' out-of-fold predictions")
    segment = combiner.segment_values(spec, X)
    fits = (combiner.fold_fits(spec, cfg.weights, oof, y, seeds, segment)
            if combiner.is_learned(spec) else {})
    fixed = None if combiner.is_learned(spec) else combiner.fit(spec, cfg.weights, {}, y)
    levels = {"component": grouped, "column": {c: [c] for cols in grouped.values() for c in cols}}
    subjects = [*cfg.weights, *([BLEND] if len(cfg.weights) > 1 else [])]
    readings = {lv: {s: {g: {"logloss": [], "accuracy": []} for g in gs} for s in subjects}
                for lv, gs in levels.items()}
    for seed in seeds:
        for k, (tr, va) in enumerate(split.folds(y, seed)):
            if only_fold is not None and k != only_fold:
                continue
            fitted = {}
            for m in cfg.weights:
                est = comps[m].build(dict(cfg.params[m]), seed)
                est.fit(X.iloc[tr], y[tr])
                fitted[m] = est
            Xv, yv = X.iloc[va], y[va]
            comb = fits.get((seed, k), fixed)
            before = _metrics(_predict(cfg, fitted, Xv, comb), yv)
            rng = np.random.default_rng([seed, k])
            for lv, gs in levels.items():
                for g, cols in gs.items():
                    for _ in range(repeats):
                        perm = rng.permutation(len(Xv))
                        shuffled = Xv.copy()
                        for c in cols:
                            shuffled[c] = Xv[c].iloc[perm].set_axis(Xv.index)
                        after = _metrics(_predict(cfg, fitted, shuffled, comb), yv)
                        for s in subjects:
                            r = readings[lv][s][g]
                            r["logloss"].append(after[s][0] - before[s][0])
                            r["accuracy"].append(before[s][1] - after[s][1])
    return {lv: {s: {g: {k: [float(np.mean(v)), float(np.std(v))] for k, v in r.items()}
                     for g, r in per_s.items()} for s, per_s in per_lv.items()}
            for lv, per_lv in readings.items()}


def _predict(cfg: config.Config, fitted: dict, X: pd.DataFrame, comb: combiner.Fitted) -> dict:
    probs = {m: est.predict_proba(X)[:, 1] for m, est in fitted.items()}
    if len(probs) > 1:  # the harness's own combiner (the weighted average, by default)
        probs[BLEND] = comb.apply(probs, combiner.segment_values(cfg.combiner, X))
    return probs


def _metrics(probs: dict, y: np.ndarray) -> dict:
    """{subject: (log-loss, accuracy)}, accuracy cut at 0.5 like the harness."""
    out = {}
    for s, p in probs.items():
        p = np.clip(p, EPS, 1 - EPS)
        out[s] = (float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                  float(np.mean((p > 0.5) == y)))
    return out


def render(result: dict, against: dict | None = None) -> str:
    """Markdown: per subject, a by-component table then a by-column drill-down, rows ranked by
    this config's log-loss rise. With `against`, its readings sit alongside; a component or
    column present in only one config shows `—` on the other side, never 0."""
    title = f"### Explanation: {result['config']} ({result['digest']})"
    if against:
        title += f" alongside {against['config']} ({against['digest']})"
    out = [title, "",
           "How much worse each model's validation-fold predictions get when a group of columns "
           "is shuffled. Descriptive only.", ""]
    out += [f"> note ({_stem(r)}): {n}" for r in (result, against) if r for n in r["notes"]]
    names = [_stem(result)] + ([_stem(against)] if against else [])
    subjects = list(dict.fromkeys([*result["component"], *(against["component"] if against else {})]))
    for s in subjects:
        for level in ("component", "column"):
            ours = result[level].get(s, {})
            theirs = against[level].get(s, {}) if against else {}
            rows = sorted(ours, key=lambda g: -ours[g]["logloss"][0])
            rows += [g for g in theirs if g not in ours]
            head = [level] + [f"{m} ({n})" for n in names for m in ("log-loss ↑", "accuracy ↓")]
            out += ["", f"#### {s}, by {level}", "",
                    "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for g in rows:
                cells = [g] + _cells(ours.get(g)) + (_cells(theirs.get(g)) if against else [])
                out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _cells(r: dict | None) -> list[str]:
    if r is None:
        return ["—", "—"]
    return [f"{r[k][0]:+.4f} ± {r[k][1]:.4f}" for k in ("logloss", "accuracy")]


def _stem(result: dict) -> str:
    return result["config"].replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".toml")


def run(cfg: config.Config, comps: dict, seeds: tuple = SEEDS) -> dict:
    """Explain `cfg` on the dev rows; the result carries the config's run digest."""
    dev, y, test = data.load_dev()
    X, grouped, notes = groups(cfg, comps, pd.concat([dev, test], ignore_index=True))
    data_fp = data.fingerprint()
    keys = {m: cache_key(cfg, comps, m, data_fp) for m in cfg.weights}
    oof = None
    if combiner.is_learned(cfg.combiner):  # cached out-of-fold predictions, trained on a miss
        if not set(seeds) <= set(split.SEEDS):
            raise ValueError(f"a {cfg.combiner['method']!r} combiner is explained on the scoring seeds {split.SEEDS}")
        preds, _, _ = predictions(cfg, comps)
        oof = {m: p[0][[split.SEEDS.index(s) for s in seeds]] for m, p in preds.items()}
    t0 = time.time()
    found = explain(cfg, comps, X.iloc[: len(dev)], y, grouped, seeds=seeds, oof=oof)
    return {"config": cfg.name, "digest": digest(keys, cfg.weights, cfg.combiner), "seeds": list(seeds),
            "repeats": REPEATS, "dev_rows": int(len(y)), "groups": grouped, "notes": notes,
            **found, "seconds": round(time.time() - t0, 1)}


def save(result: dict) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{_stem(result)}-{result['digest']}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    return path.relative_to(paths.ROOT).as_posix()


def smoke(comps: dict, path=None) -> None:
    cfg = config.load(path or paths.CONFIGS / "main.toml", comps)
    dev, y, test = data.load_dev()
    dev, y = dev.iloc[:SMOKE_ROWS], y[:SMOKE_ROWS]
    X, grouped, _ = groups(cfg, comps, pd.concat([dev, test], ignore_index=True))
    X = X.iloc[: len(dev)]
    oof = None
    if combiner.is_learned(cfg.combiner):  # needs every fold's out-of-fold predictions
        oof = {m: fit_predict(comps[m], cfg.params[m], X, y, X.iloc[:1], seeds=SEEDS)[0] for m in cfg.weights}
    t0 = time.time()
    explain(cfg, comps, X, y, grouped, repeats=1, only_fold=0, oof=oof)
    print(json.dumps({"smoke": "explain", "rows": len(y), "groups": sorted(grouped),
                      "seconds": round(time.time() - t0, 1)}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.explain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--against", default=None, help="a second config, explained the same way, shown alongside")
    ap.add_argument("--seeds", default="0", help="comma-separated fold seeds (default 0)")
    ap.add_argument("--smoke", action="store_true", help="seconds-long check that explanation runs (CI)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the table's ↑ ± — on a cp1252 console

    comps = registry.discover()
    if args.smoke:
        return smoke(comps, args.config)
    if not args.config:
        ap.error("config is required unless --smoke is given")

    seeds = tuple(int(s) for s in args.seeds.split(","))
    result = run(config.load(args.config, comps), comps, seeds)
    written = [save(result)]
    against = None
    if args.against:
        against = run(config.load(args.against, comps), comps, seeds)
        written.append(save(against))
    print(render(result, against))
    print("\nwritten: " + ", ".join(written))


if __name__ == "__main__":
    main()
