"""Does row-level CV overstate accuracy because travel-group members sit on both sides of a fold?

    python -m scripts.group_cv [--seeds 0,1,2,200,201,202] [--workers 8] [--out FILE]
    python -m scripts.group_cv --smoke

For main, each screened feature idea added to main, and main with one feature dropped, scores
the same model on the same seeds under two fold schemes over the dev rows:
  row   -- `harness.split.folds` (StratifiedKFold on y), what every CV score uses;
  group -- StratifiedGroupKFold on the travel group, so no group spans train and validation,
           as in the real test set.
Reports mean accuracy per scheme, the paired gap (row - group) with its standard error over
seeds, and each variant's delta against main under both schemes (does the ranking change?).

Measurement only: adds to what the harness runs, never edits `harness/`, writes nothing to
`results/` (the optional --out file is yours to place). Reuses `config.load`,
`harness.score.build_matrix` and `split.folds`; rolls its own fit loop like `scripts/confirm.py`.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from harness import config, data, paths, registry, split
from harness.score import build_matrix

DEFAULT_SEEDS = (0, 1, 2, 200, 201, 202)  # the scorer's seeds plus the confirmation seeds
IDEAS = ("cabin_bins", "spend_logs", "family", "age_bins")
DROPS = ("Group", "CabinNum", "Deck", "Side", "GroupSize", "Solo")
THREADS = 4
SMOKE_ROWS = 600


def group_folds(y: np.ndarray, groups: np.ndarray, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=split.N_FOLDS, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def variants() -> dict[str, tuple[str, str | None]]:
    """{label: (config path, column to drop or None)}"""
    out = {"main": ("configs/main.toml", None)}
    for idea in IDEAS:
        path = paths.CONFIGS / "exp" / f"{idea}_cb.toml"  # the CatBoost-base screen where one exists
        if not path.exists():
            path = paths.CONFIGS / "exp" / f"{idea}.toml"
        out[f"+{idea}"] = (str(path), None)
    for col in DROPS:
        out[f"-{col}"] = ("configs/main.toml", col)
    return out


def run_task(task: tuple) -> tuple:
    label, cfg_path, drop, scheme, seed, rows = task
    comps = registry.discover()
    cfg = config.load(cfg_path, comps)
    dev, y, test = data.load_dev()
    full = build_matrix(cfg, comps, pd.concat([dev, test], ignore_index=True))
    X = full.iloc[: len(dev)]
    if drop:
        X = X.drop(columns=[drop])
    groups = dev["PassengerId"].str.split("_").str[0].to_numpy()
    if rows:
        X, y, groups = X.iloc[:rows], y[:rows], groups[:rows]
    folds = split.folds(y, seed) if scheme == "row" else group_folds(y, groups, seed)
    if rows:
        folds = folds[:1]
    prob = np.full(len(y), np.nan)
    for tr, va in folds:
        est = comps["catboost"].build({**cfg.params["catboost"], "thread_count": THREADS}, seed)
        est.fit(X.iloc[tr], y[tr])
        prob[va] = est.predict_proba(X.iloc[va])[:, 1]
    seen = ~np.isnan(prob)
    return label, scheme, seed, float(((prob[seen] > 0.5) == y[seen]).mean())


def se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarise(acc: dict, seeds: tuple) -> dict:
    """acc[label][scheme] -> per-seed accuracy array (ordered like `seeds`)."""
    rep = {}
    for label, a in acc.items():
        row, grp = np.array(a["row"]), np.array(a["group"])
        gap = row - grp
        rep[label] = {"row": float(row.mean()), "group": float(grp.mean()),
                      "gap": float(gap.mean()), "gap_se": se(gap)}
        if label != "main":
            for scheme, cur in (("row", row), ("group", grp)):
                d = cur - np.array(acc["main"][scheme])
                rep[label][f"d_{scheme}"] = float(d.mean())
                rep[label][f"d_{scheme}_se"] = se(d)
    return rep


def print_report(rep: dict, seeds: tuple) -> None:
    print(f"\nseeds {list(seeds)}; accuracy over dev rows; gap = row - group (paired by seed)\n")
    print(f"{'variant':<14}{'row':>9}{'group':>9}{'gap':>9}{'+-se':>8}   {'d_row':>9}{'+-se':>8}{'d_group':>10}{'+-se':>8}")
    for label, r in rep.items():
        tail = ""
        if label != "main":
            tail = (f"   {r['d_row']:+9.5f}{r['d_row_se']:8.5f}{r['d_group']:+10.5f}{r['d_group_se']:8.5f}")
        print(f"{label:<14}{r['row']:9.5f}{r['group']:9.5f}{r['gap']:+9.5f}{r['gap_se']:8.5f}{tail}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.group_cv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="write the full report as JSON here")
    ap.add_argument("--smoke", action="store_true", help="seconds-long check that it runs (one fold, subset)")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    vs = variants()
    if args.smoke:
        seeds, vs, rows = seeds[:1], {k: vs[k] for k in ("main", "-Group")}, SMOKE_ROWS
    else:
        rows = 0
    tasks = [(label, p, drop, scheme, seed, rows) for label, (p, drop) in vs.items()
             for scheme in ("row", "group") for seed in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        done = list(ex.map(run_task, tasks))
    acc = {l: {"row": [], "group": []} for l in vs}
    by = {(l, s, sd): a for l, s, sd, a in done}
    for l in vs:
        for scheme in ("row", "group"):
            acc[l][scheme] = [by[(l, scheme, sd)] for sd in seeds]
    if args.smoke:
        print(json.dumps({"smoke": "group_cv", "tasks": len(tasks), "seconds": round(time.time() - t0, 1)}))
        return
    rep = summarise(acc, seeds)
    print_report(rep, seeds)
    print(f"\n({round(time.time() - t0)}s)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seeds": list(seeds), "per_seed": acc, "summary": rep}, f, indent=2)


if __name__ == "__main__":
    main()
