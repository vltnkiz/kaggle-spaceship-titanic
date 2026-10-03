"""The only code that reads the holdout. Prints; writes nothing.

    python -m harness.audit CONFIG [CONFIG ...]

Each model trains on all dev rows (one fit per seed, fixed params, nothing tuned) and
predicts the holdout; the config's combiner (harness/combiner.py) turns each seed's models into
one probability, and the seeds are averaged.

The number is a smoke detector for gross divergence, never a keep input: at ~1,300 rows
one standard error is about 0.011, five times the keep threshold. Run it at landing,
quote it in the landing recap, and do not copy it into results/, a config, or a map.
"""
import argparse

import numpy as np
import pandas as pd

from harness import combiner, config, data, registry, split
from harness.fit import build_matrix
from harness.score import predictions


def holdout_accuracy(cfg: config.Config, comps: dict) -> tuple[float, int]:
    train = data.read("train.csv")
    mask = split.in_holdout(train[data.ID])
    y = train.pop(data.TARGET).astype(int).to_numpy()
    dev, hold = train[~mask], train[mask]
    frame = pd.concat([dev, hold, data.read("test.csv")], ignore_index=True)
    X = build_matrix(cfg, comps, frame)
    X_dev, X_hold = X.iloc[: len(dev)], X.iloc[len(dev): len(dev) + len(hold)]
    held = {m: np.array([_fit_one(comps[m], cfg.params[m], seed, X_dev, y[~mask], X_hold)
                         for seed in split.SEEDS]) for m in cfg.weights}
    seg = combiner.segment_values(cfg.combiner, X)
    dev_seg, hold_seg = (None, None) if seg is None else (seg[: len(dev)], seg[len(dev): len(dev) + len(hold)])
    oof = None
    if combiner.is_learned(cfg.combiner):  # meta-fit on dev's cached out-of-fold predictions
        preds, _, _ = predictions(cfg, comps)
        oof = {m: p[0] for m, p in preds.items()}
    prob = combiner.combine_test(cfg.combiner, cfg.weights, oof, held, y[~mask], split.SEEDS,
                                 dev_seg, hold_seg).mean(axis=0)
    acc = float(((prob > 0.5) == y[mask]).mean())
    return acc, int(mask.sum())


def _fit_one(comp, params, seed, X_dev, y_dev, X_hold) -> np.ndarray:
    est = comp.build(dict(params), seed)
    est.fit(X_dev, y_dev)
    return est.predict_proba(X_hold)[:, 1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m harness.audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv)
    comps = registry.discover()
    for path in args.configs:
        cfg = config.load(path, comps)
        acc, n = holdout_accuracy(cfg, comps)
        se = np.sqrt(acc * (1 - acc) / n)
        print(f"holdout  {cfg.name}: accuracy {acc:.4f}  (n={n}, 1 s.e. = {se:.4f})")


if __name__ == "__main__":
    main()
