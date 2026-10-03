"""Write a Kaggle submission from a config's test predictions.

    python -m harness.submit CONFIG        -> submissions/<config name>.csv

Test predictions are the average of the 15 fold models the CV score came from (trained on
dev rows only), read from the cache; a model that is not cached yet is trained first. The
config's combiner (harness/combiner.py) turns each seed's models into one probability, and the
seeds are averaged.
"""
import argparse
from pathlib import Path

import pandas as pd

from harness import config, data, paths, registry
from harness.score import combined


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m harness.submit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args(argv)
    comps = registry.discover()
    cfg = config.load(args.config, comps)
    _, test_prob, *_ = combined(cfg, comps)
    prob = test_prob.mean(axis=0)
    test = data.read("test.csv")
    out = paths.ROOT / "submissions" / f"{Path(cfg.name).stem}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({data.ID: test[data.ID], data.TARGET: prob > 0.5}).to_csv(out, index=False)
    print(f"Wrote {out.relative_to(paths.ROOT).as_posix()} ({len(test)} rows)")


if __name__ == "__main__":
    main()
