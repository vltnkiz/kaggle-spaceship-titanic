"""The only code that reads the holdout. Prints; writes nothing.

    python -m harness.audit CONFIG [CONFIG ...]

Each model trains on all dev rows (one fit per seed, fixed params, nothing tuned) and
predicts the holdout; seeds are averaged, models blended by the config's weights.

The number is a smoke detector for gross divergence, never a keep input: at ~1,300 rows
one standard error is about 0.011, five times the keep threshold. Run it at landing,
quote it in the landing recap, and do not copy it into results/, a config, or a map.
"""
import argparse

import numpy as np
import pandas as pd

from harness import config, data, registry, split
from harness.score import blend, build_matrix


def holdout_accuracy(cfg: config.Config, comps: dict) -> tuple[float, int]:
    train = data.read("train.csv")
    mask = split.in_holdout(train[data.ID])
    y = train.pop(data.TARGET).astype(int).to_numpy()
    dev, hold = train[~mask], train[mask]
    frame = pd.concat([dev, hold, data.read("test.csv")], ignore_index=True)
    X = build_matrix(cfg, comps, frame)
    X_dev, X_hold = X.iloc[: len(dev)], X.iloc[len(dev): len(dev) + len(hold)]
    preds = {}
    for m in cfg.weights:
        p = np.zeros(len(hold))
        for seed in split.SEEDS:
            est = comps[m].build(dict(cfg.params[m]), seed)
            est.fit(X_dev, y[~mask])
            p += est.predict_proba(X_hold)[:, 1] / len(split.SEEDS)
        preds[m] = (p,)
    acc = float(((blend(preds, cfg.weights) > 0.5) == y[mask]).mean())
    return acc, int(mask.sum())


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
