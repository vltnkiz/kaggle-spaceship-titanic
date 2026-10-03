"""The metric, and the only CLI that produces a score.

    python -m harness.score CONFIG              score it; write results/<run-id>.json
    python -m harness.score configs/main.toml --pin
                                                also pin it as the base (configs/main.json)
    python -m harness.score CONFIG --grid 0.1   sweep the config's blend weights over cached
                                                predictions; prints, records nothing
    python -m harness.score CONFIG --smoke      one fold of one seed on a subset; writes
                                                nothing (for CI: does the scorer still run?)

CV score: accuracy at a 0.5 threshold, 5-fold stratified CV, repeated over 3 seeds, over the
dev rows. Every config is scored on the same folds, so a delta against the pinned base is a
paired comparison. Each model's out-of-fold and test predictions are cached by a hash of
everything that produced them, so a combination of cached models (see harness/combiner.py)
costs no training.
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from harness import combiner, config, data, paths, registry, split
from harness.fit import build_matrix, fit_predict

KEEP_DELTA = 0.002
SMOKE_ROWS = 2000


# The code that produces predictions. The combiner, scoring and reporting are deliberately
# absent: editing them cannot change a cached prediction, so it must not invalidate one.
FINGERPRINTED = ("config.py", "data.py", "fit.py", "registry.py", "split.py")


def fingerprint() -> str:
    """Hash of the prediction-producing harness code: a changed fit loop, split or data
    loader must not reuse old predictions. Line endings are normalised so a CRLF checkout
    hashes the same as an LF one."""
    h = hashlib.sha256()
    for name in FINGERPRINTED:
        p = Path(__file__).parent / name
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def cache_key(cfg: config.Config, comps: dict, model: str, data_fp: str) -> str:
    recipe = {
        "features": [(n, comps[n].source_hash) for n in cfg.features],
        "model": (model, comps[model].source_hash),
        "params": cfg.params[model],
        "seeds": split.SEEDS, "folds": split.N_FOLDS,
        "harness": fingerprint(), "data": data_fp,
    }
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode()).hexdigest()[:20]


def predictions(cfg: config.Config, comps: dict) -> tuple[dict, dict, np.ndarray]:
    """{model: (oof, test)}, {model: cache key}, dev target. Trains only on a cache miss."""
    dev, y, test = data.load_dev()
    data_fp = data.fingerprint()
    keys = {m: cache_key(cfg, comps, m, data_fp) for m in cfg.weights}
    cache = paths.oof_dir()
    out, X, X_test = {}, None, None
    for model, key in keys.items():
        oof_path, test_path = cache / f"{key}.npy", cache / f"{key}.test.npy"
        if oof_path.exists() and test_path.exists():
            out[model] = (np.load(oof_path), np.load(test_path))
            print(f"  {model}: cached {key}")
            continue
        if X is None:
            full = build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True))
            X, X_test = full.iloc[: len(dev)], full.iloc[len(dev):]
        t0 = time.time()
        out[model] = fit_predict(comps[model], cfg.params[model], X, y, X_test)
        print(f"  {model}: trained in {time.time() - t0:.1f}s -> {key}")
        _save(oof_path, out[model][0])
        _save(test_path, out[model][1])
    return out, keys, y


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def blend(preds: dict, weights: dict, part: int = 0) -> np.ndarray:
    total = sum(weights.values())
    return sum(w * preds[m][part] for m, w in weights.items()) / total


def accuracy(prob: np.ndarray, y: np.ndarray) -> tuple[list[float], list[list[float]]]:
    """Per-seed accuracy over all dev rows, and per-seed per-fold accuracy."""
    per_seed, per_fold = [], []
    for s, seed in enumerate(split.SEEDS):
        hit = (prob[s] > 0.5) == y
        fold = split.fold_of(y, seed)
        per_seed.append(float(hit.mean()))
        per_fold.append([float(hit[fold == k].mean()) for k in range(split.N_FOLDS)])
    return per_seed, per_fold


def digest(cache_keys: dict, weights: dict, spec: dict | None = None) -> str:
    """Identity of a scored pipeline: what each model was trained from, and how they combine.
    The default `mean` combiner is left out, so a plain blend keeps the digest it always had."""
    parts = [cache_keys, weights]
    if spec is not None and spec["method"] != "mean":
        parts.append(spec)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def segment_columns(cfg: config.Config, comps: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """(dev, test) segment labels for a `gate` combiner, from the built feature matrix."""
    if cfg.combiner["method"] != "gate":
        return None, None
    dev, _, test = data.load_dev()
    seg = combiner.segment_values(cfg.combiner, build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True)))
    return seg[: len(dev)], seg[len(dev):]


def combined(cfg: config.Config, comps: dict) -> tuple[np.ndarray, np.ndarray, dict, dict, np.ndarray]:
    """(out-of-fold, test) combined probabilities by seed, the per-model predictions, their
    cache keys and the dev target."""
    preds, keys, y = predictions(cfg, comps)
    seg, test_seg = segment_columns(cfg, comps)
    oof = {m: p[0] for m, p in preds.items()}
    test = {m: p[1] for m, p in preds.items()}
    return (combiner.oof_combine(cfg.combiner, cfg.weights, oof, y, split.SEEDS, seg),
            combiner.combine_test(cfg.combiner, cfg.weights, oof, test, y, split.SEEDS, seg, test_seg),
            preds, keys, y)


def pinned_base(comps: dict) -> dict | None:
    """The pinned base, marked stale if configs/main.toml no longer produces what was pinned."""
    if not paths.PINNED.exists():
        return None
    base = json.loads(paths.PINNED.read_text())
    main = config.load(paths.CONFIGS / "main.toml", comps)
    data_fp = data.fingerprint()
    keys = {m: cache_key(main, comps, m, data_fp) for m in main.weights}
    base["stale"] = digest(keys, main.weights, main.combiner) != base["digest"]
    return base


def score(cfg: config.Config, comps: dict) -> dict:
    t0 = time.time()
    prob, _, _, keys, y = combined(cfg, comps)
    per_seed, per_fold = accuracy(prob, y)
    result = {
        "config": cfg.name,
        "features": list(cfg.features),
        "weights": cfg.weights,
        "combiner": cfg.combiner,
        "params": cfg.params,
        "cache_keys": keys,
        "digest": digest(keys, cfg.weights, cfg.combiner),
        "cv": float(np.mean(per_seed)),
        "per_seed": per_seed,
        "per_fold": per_fold,
        "dev_rows": int(len(y)),
    }
    result.update(_compare(result, pinned_base(comps)))
    result.update(harness=fingerprint(), data=data.fingerprint(), **_git(),
                  seconds=round(time.time() - t0, 1))
    return result


def _compare(result: dict, base: dict | None) -> dict:
    if base is None:
        return {"base": None, "delta": None, "verdict": None}
    diff = np.array(result["per_fold"]) - np.array(base["per_fold"])
    delta = result["cv"] - base["cv"]
    if base["stale"]:
        verdict = None  # main.toml or its inputs changed since the pin: re-pin first
    elif result["digest"] == base["digest"]:
        verdict = "base"
    else:
        verdict = "keep" if round(delta, 6) >= KEEP_DELTA else "drop"
    return {
        "base": {k: base[k] for k in ("cv", "digest", "commit", "stale")},
        "delta": delta,
        "folds_better": int((diff > 0).sum()),
        "folds_worse": int((diff < 0).sum()),
        "verdict": verdict,
    }


def _git() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True,
                              text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def record(result: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(result["config"]).stem
    paths.RESULTS.mkdir(exist_ok=True)
    path = paths.RESULTS / f"{stamp}-{stem}.json"
    n = 1
    while path.exists():
        n += 1
        path = paths.RESULTS / f"{stamp}-{stem}-{n}.json"
    result = {"run_id": path.stem, **result}
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def pin(result: dict, run_path: Path) -> None:
    keep = ("config", "digest", "cv", "per_seed", "per_fold", "cache_keys", "commit")
    pinned = {"run_id": run_path.stem, **{k: result[k] for k in keep}}
    paths.PINNED.write_text(json.dumps(pinned, indent=2) + "\n")


def grid(cfg: config.Config, comps: dict, step: float) -> None:
    models = list(cfg.weights)
    if cfg.combiner["method"] != "mean":
        sys.exit(f"--grid sweeps blend weights and needs combiner method 'mean', not {cfg.combiner['method']!r}")
    if len(models) < 2:
        sys.exit("--grid needs a config with two or more models")
    preds, _, y = predictions(cfg, comps)
    n = round(1 / step)
    rows = []
    for parts in itertools.product(range(n + 1), repeat=len(models)):
        if sum(parts) == n:
            w = {m: p / n for m, p in zip(models, parts)}
            rows.append((float(np.mean(accuracy(blend(preds, w), y)[0])), w))
    rows.sort(key=lambda r: -r[0])
    print(f"\nBlend sweep over {len(rows)} weightings (not recorded; set weights in a config and score it):")
    for cv, w in rows[:10]:
        print(f"  {cv:.5f}  " + "  ".join(f"{m}={v:.2f}" for m, v in w.items()))


def smoke(cfg: config.Config, comps: dict) -> None:
    dev, y, test = data.load_dev()
    dev, y = dev.iloc[:SMOKE_ROWS], y[:SMOKE_ROWS]
    full = build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True))
    X, X_test = full.iloc[: len(dev)], full.iloc[len(dev):]
    seg = combiner.segment_values(cfg.combiner, full)
    seg, test_seg = (None, None) if seg is None else (seg[: len(dev)], seg[len(dev):])
    seeds = split.SEEDS[:1]
    # a learned combiner is meta-fit on the other folds' predictions, so it needs every fold
    only_fold = None if combiner.is_learned(cfg.combiner) else 0
    va = split.folds(y, seeds[0])[0][1]
    oof, tst = {}, {}
    for m in cfg.weights:
        oof[m], tst[m] = fit_predict(comps[m], cfg.params[m], X, y, X_test, seeds=seeds, only_fold=only_fold)
    prob = combiner.oof_combine(cfg.combiner, cfg.weights, oof, y, seeds, seg)
    combiner.combine_test(cfg.combiner, cfg.weights, oof, tst, y, seeds, seg, test_seg)
    acc = float(((prob[0][va] > 0.5) == y[va]).mean())
    print(json.dumps({"smoke": cfg.name, "rows": len(y), "fold_rows": len(va), "accuracy": acc}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m harness.score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pin", action="store_true", help="pin configs/main.toml as the base")
    mode.add_argument("--grid", type=float, metavar="STEP", help="sweep blend weights")
    mode.add_argument("--smoke", action="store_true", help="fast check that scoring runs")
    args = ap.parse_args(argv)

    comps = registry.discover()
    cfg = config.load(args.config, comps)
    if args.smoke:
        return smoke(cfg, comps)
    if args.grid:
        return grid(cfg, comps, args.grid)
    if args.pin and cfg.name != "configs/main.toml":
        sys.exit("--pin only pins configs/main.toml")

    print(f"Scoring {cfg.name}: features={list(cfg.features)} models={cfg.weights}")
    result = score(cfg, comps)
    path = record(result)
    if args.pin:
        pin(result, path)
    _report(result, path, pinned=args.pin)


def _report(r: dict, path: Path, pinned: bool) -> None:
    seeds = " / ".join(f"{a:.4f}" for a in r["per_seed"])
    print(f"\nCV {r['cv']:.5f}   per seed {seeds}   ({r['dev_rows']} dev rows, {r['seconds']}s)")
    if pinned:
        print(f"Pinned as the base -> {paths.PINNED.relative_to(paths.ROOT).as_posix()}")
    elif r["base"] is None:
        print("No pinned base yet: run  python -m harness.score configs/main.toml --pin")
    elif r["base"]["stale"]:
        print(f"Base {r['base']['cv']:.5f} is STALE (main.toml or its inputs changed): re-pin; no verdict")
    else:
        print(f"Base {r['base']['cv']:.5f}   delta {r['delta']:+.5f}   folds better/worse "
              f"{r['folds_better']}/{r['folds_worse']} of {split.N_FOLDS * len(split.SEEDS)}"
              f"   verdict {r['verdict'].upper()}")
    print(f"Recorded {path.relative_to(paths.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
