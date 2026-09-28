"""Re-score a config on the confirmation seeds (200-202) -- disjoint from both the scorer's
own seeds (0-2, `harness/split.py`) and the tuner's (100-102, `scripts/tune.py`) -- and,
optionally, diff it against a second config scored the same way. This is what a batch's
matrix winner and the pinned base are both re-scored on before either label sticks.

    python -m scripts.confirm CONFIG [--against CONFIG2]
    python -m scripts.confirm CONFIG --smoke

Winner's-curse control: whichever matrix cell scored highest on the CV seeds is the noisiest
possible pick (it beat every rival on the seeds a batch also picks with). Re-scoring it and
the base on seeds neither selection touched tells the batch's real delta.

Reuses the harness's own building blocks (`config.load`, `registry.discover`,
`data.load_dev`, `harness.score.build_matrix`, `split.folds`/`fold_of`) but rolls its own
fit/predict loop over the confirmation seeds rather than calling `harness.score.score()` --
that function, and the OOF cache it reads and writes, are keyed to `split.SEEDS` (0-2) and
must stay that way (see `harness/score.py`'s `cache_key`), so confirmation runs are
uncached and never touch `results/` or `configs/main.json`. This script only adds to what
the harness runs, same as any other file under `scripts/`; it never edits `harness/`.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from harness import config, data, paths, registry, split
from harness.score import KEEP_DELTA, build_matrix

CONFIRM_SEEDS = (200, 201, 202)  # disjoint from split.SEEDS (0-2) and scripts.tune.TUNE_SEEDS (100-102)
SMOKE_ROWS = 300


def oof_accuracy(cfg: config.Config, comps: dict, X: pd.DataFrame, y: np.ndarray,
                  seeds: tuple, only_fold: int | None = None) -> tuple[list[float], list[list[float]]]:
    """Per-seed and per-seed-per-fold accuracy of `cfg`'s blend, fit fresh on `seeds`
    (never cached: these seeds are never reused across runs, so caching them would only
    grow `results/oof/` for predictions nothing else will ever read)."""
    oof = {m: np.full((len(seeds), len(X)), np.nan) for m in cfg.weights}
    for s, seed in enumerate(seeds):
        for k, (tr, va) in enumerate(split.folds(y, seed)):
            if only_fold is not None and k != only_fold:
                continue
            for m in cfg.weights:
                est = comps[m].build(dict(cfg.params[m]), seed)
                est.fit(X.iloc[tr], y[tr])
                oof[m][s, va] = est.predict_proba(X.iloc[va])[:, 1]
    total = sum(cfg.weights.values())
    blended = sum(w * oof[m] for m, w in cfg.weights.items()) / total
    per_seed, per_fold = [], []
    for s, seed in enumerate(seeds):
        fold = split.fold_of(y, seed)
        hit = (blended[s] > 0.5) == y
        if only_fold is not None:
            acc = float(hit[fold == only_fold].mean())
            per_seed.append(acc)
            per_fold.append([acc if only_fold == k else float("nan") for k in range(split.N_FOLDS)])
        else:
            per_seed.append(float(hit.mean()))
            per_fold.append([float(hit[fold == k].mean()) for k in range(split.N_FOLDS)])
    return per_seed, per_fold


def matrix_for(cfg: config.Config, comps: dict) -> tuple[pd.DataFrame, np.ndarray]:
    dev, y, test = data.load_dev()
    full = build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True))
    return full.iloc[: len(dev)], y


def confirm(cfg: config.Config, comps: dict, seeds: tuple = CONFIRM_SEEDS,
            only_fold: int | None = None) -> dict:
    X, y = matrix_for(cfg, comps)
    per_seed, per_fold = oof_accuracy(cfg, comps, X, y, seeds, only_fold=only_fold)
    return {"config": cfg.name, "seeds": list(seeds), "cv": float(np.mean(per_seed)),
            "per_seed": per_seed, "per_fold": per_fold}


def diff(result: dict, base: dict) -> dict:
    a, b = np.array(result["per_fold"]), np.array(base["per_fold"])
    delta = result["cv"] - base["cv"]
    d = a - b
    verdict = "keep" if round(delta, 6) >= KEEP_DELTA else ("near_miss" if delta >= 0 else "drop")
    return {
        "against": base["config"], "against_cv": base["cv"], "delta": delta,
        "folds_better": int((d > 0).sum()), "folds_worse": int((d < 0).sum()), "verdict": verdict,
    }


def smoke(comps: dict) -> None:
    cfg = config.load(paths.CONFIGS / "main.toml", comps)
    dev, y, test = data.load_dev()
    dev, y = dev.iloc[:SMOKE_ROWS], y[:SMOKE_ROWS]
    full = build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True))
    X = full.iloc[: len(dev)]
    t0 = time.time()
    per_seed, _ = oof_accuracy(cfg, comps, X, y, CONFIRM_SEEDS[:1], only_fold=0)
    print(json.dumps({"smoke": "confirm", "rows": len(y), "accuracy": per_seed[0],
                      "seconds": round(time.time() - t0, 1)}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.confirm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("--against", default=None, help="a second config, scored the same way, to diff against")
    ap.add_argument("--smoke", action="store_true", help="seconds-long check that confirmation scoring runs (CI)")
    args = ap.parse_args(argv)

    comps = registry.discover()
    if args.smoke:
        return smoke(comps)
    if not args.config:
        ap.error("config is required unless --smoke is given")

    cfg = config.load(args.config, comps)
    t0 = time.time()
    result = confirm(cfg, comps)
    result["seconds"] = round(time.time() - t0, 1)
    print(f"confirm {cfg.name}: CV {result['cv']:.5f}  seeds {CONFIRM_SEEDS}  ({result['seconds']}s)")

    if args.against:
        base_cfg = config.load(args.against, comps)
        t0 = time.time()
        base = confirm(base_cfg, comps)
        base["seconds"] = round(time.time() - t0, 1)
        result.update(diff(result, base))
        print(f"vs {base_cfg.name}: CV {base['cv']:.5f}  delta {result['delta']:+.5f}  "
              f"folds better/worse {result['folds_better']}/{result['folds_worse']} "
              f"of {split.N_FOLDS * len(CONFIRM_SEEDS)}  verdict {result['verdict'].upper()}")

    print(json.dumps(result))


if __name__ == "__main__":
    main()
