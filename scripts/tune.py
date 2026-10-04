"""Search Optuna hyperparameters for one model already in the pinned base, and write a
tuned config -- a **tuning idea** -- for `harness.score` to screen like any other idea.

    python -m scripts.tune MODEL [--trials N] [--hours H] [--out NAME]
    python -m scripts.tune MODEL --smoke        seconds-long check that the search runs (CI)

Trials use disjoint fold-split/model seeds 100-102 -- never the scorer's 0-2 (`split.SEEDS`)
-- so a param set is never selected on the rows it will later be judged on. Each trial scores
1 seed x 5 folds (~3 min for CatBoost); once the budget (<=30 trials or <=2h, whichever comes
first) is spent, the top 3 trials by that single-seed score are re-checked on all 3 disjoint
seeds (15 folds), and the best of those three is written to
`configs/exp/<name>.toml` (`extends = "main"` + `[params.<model>]`). This script never runs
against the scorer's own seeds, never writes to `results/`, and never decides a verdict --
that happens when the written config is screened by `harness.score`, same as any other idea.

A trial's Optuna study is stored in `.tune/<model>.db` (sqlite, gitignored) so a crashed
search resumes with `--trials`/`--hours` picking up where it left off, same budget applying
to the trials still to run.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from harness import config, data, paths, registry, split
from harness.score import build_matrix

TUNE_SEEDS = (100, 101, 102)  # disjoint from split.SEEDS (0, 1, 2): never the scorer's rows
MAX_TRIALS = 30
MAX_HOURS = 2.0
TOP_K = 3
SMOKE_ROWS = 300
# --smoke still samples the real space (so the wiring is exercised) but clamps whichever
# param dominates fit cost, so CI stays seconds-long regardless of what Optuna draws.
SMOKE_CAPS = {"lgbm": {"n_estimators": 30}, "catboost": {"iterations": 30},
              "tabm": {"n_epochs": 5}, "tabpfn": {"n_estimators": 1}}

# One entry per tunable model: {param: (kind, low, high)}. kind is "int", "float" (uniform)
# or "float_log" (log-uniform). Unlisted params keep the component's own PARAMS default.
SPACES = {
    "lgbm": {
        "n_estimators": ("int", 200, 800),
        "learning_rate": ("float_log", 0.005, 0.1),
        "num_leaves": ("int", 15, 63),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
    "catboost": {
        "iterations": ("int", 400, 1500),
        "learning_rate": ("float_log", 0.01, 0.1),
        "depth": ("int", 4, 8),
        "l2_leaf_reg": ("float_log", 1.0, 10.0),
    },
    # tabm_k, num_emb_type and arch_type stay fixed at the component's defaults.
    "tabm": {
        "lr": ("float_log", 5e-4, 5e-3),
        "weight_decay": ("float", 0.0, 0.05),
        "dropout": ("float", 0.0, 0.4),
        "d_block": ("int", 128, 768),
        "n_blocks": ("int", 1, 4),
    },
    # TabPFN is pretrained; n_estimators is the one knob worth a search.
    "tabpfn": {
        "n_estimators": ("int", 4, 32),
    },
}


def sample(trial: optuna.Trial, space: dict, defaults: dict, caps: dict | None = None) -> dict:
    params = dict(defaults)
    for name, (kind, lo, hi) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi)
        elif kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            raise ValueError(f"unknown space kind {kind!r} for {name!r}")
    if caps:
        params.update(caps)
    return params


def cv_accuracy(comp, params: dict, seed: int, X, y, only_fold: int | None = None) -> float:
    """Fold-split and model seed are the same `seed`: the pairing the leak analysis relies on."""
    hits, total = 0, 0
    for k, (tr, va) in enumerate(split.folds(y, seed)):
        if only_fold is not None and k != only_fold:
            continue
        est = comp.build(dict(params), seed)
        est.fit(X.iloc[tr], y[tr])
        pred = est.predict_proba(X.iloc[va])[:, 1]
        hits += int(((pred > 0.5) == y[va]).sum())
        total += len(va)
    return hits / total


def run_search(model: str, comp, space: dict, X, y, trials: int, hours: float,
                seeds: tuple, storage: str | None, only_fold: int | None = None,
                caps: dict | None = None) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        seed = seeds[trial.number % len(seeds)]
        trial.set_user_attr("seed", seed)
        params = sample(trial, space, comp.params, caps)
        return cv_accuracy(comp, params, seed, X, y, only_fold=only_fold)

    kwargs = dict(direction="maximize", sampler=TPESampler(seed=seeds[0]))
    if storage:
        kwargs.update(study_name=f"tune-{model}", storage=storage, load_if_exists=True)
    study = optuna.create_study(**kwargs)
    study.optimize(objective, n_trials=trials, timeout=hours * 3600)
    return study


def recheck(comp, defaults: dict, study: optuna.Study, X, y, seeds: tuple,
            only_fold: int | None = None, caps: dict | None = None) -> tuple[dict, float]:
    """Re-score the top TOP_K trials on every seed given; return the best params and score."""
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:TOP_K]
    best_params, best_score = None, -1.0
    for t in top:
        params = {**defaults, **t.params, **(caps or {})}
        score = float(np.mean([cv_accuracy(comp, params, s, X, y, only_fold=only_fold)
                               for s in seeds]))
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score


def write_config(model: str, params: dict, name: str, note: str) -> Path:
    lines = [f"# {note}", 'extends = "main"', "", f"[params.{model}]"]
    lines += [f"{k} = {json.dumps(v)}" for k, v in params.items()]
    path = paths.CONFIGS / "exp" / f"{name}.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def load_matrix(comps: dict):
    cfg = config.load(paths.CONFIGS / "main.toml", comps)
    dev, y, _test = data.load_dev()
    return cfg, build_matrix(cfg, comps, dev), y


def smoke(model: str, comp, space: dict, comps: dict) -> None:
    """No sqlite storage, no config written, one fold per fit, cost params capped: only
    proves the search loop -- sampling, the seed pairing, top-k re-check -- runs end to
    end, fast, regardless of what Optuna draws from the real space."""
    _cfg, X, y = load_matrix(comps)
    X, y = X.iloc[:SMOKE_ROWS], y[:SMOKE_ROWS]
    caps = SMOKE_CAPS.get(model)
    t0 = time.time()
    study = run_search(model, comp, space, X, y, trials=2, hours=MAX_HOURS,
                       seeds=TUNE_SEEDS, storage=None, only_fold=0, caps=caps)
    _best_params, best_score = recheck(comp, comp.params, study, X, y, seeds=TUNE_SEEDS[:1],
                                       only_fold=0, caps=caps)
    print(json.dumps({"smoke": f"tune {model}", "trials": len(study.trials),
                      "best_score": best_score, "seconds": round(time.time() - t0, 1)}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.tune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", choices=sorted(SPACES),
                    help="a model already in the pinned base")
    ap.add_argument("--trials", type=int, default=MAX_TRIALS)
    ap.add_argument("--hours", type=float, default=MAX_HOURS)
    ap.add_argument("--out", default=None, help="configs/exp/<NAME>.toml (default tune-<model>)")
    ap.add_argument("--smoke", action="store_true", help="seconds-long check that the search runs (CI)")
    ap.add_argument("--list-models", action="store_true",
                    help="print every model with a defined search space, one per line, and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        print("\n".join(sorted(SPACES)))
        return
    if not args.model:
        ap.error("model is required unless --list-models is given")

    comps = registry.discover()
    comp, space = comps[args.model], SPACES[args.model]

    if args.smoke:
        return smoke(args.model, comp, space, comps)

    if args.trials > MAX_TRIALS or args.hours > MAX_HOURS:
        ap.error(f"a tuning idea is capped at {MAX_TRIALS} trials / {MAX_HOURS}h")

    _cfg, X, y = load_matrix(comps)
    storage_dir = paths.ROOT / ".tune"
    storage_dir.mkdir(exist_ok=True)

    t0 = time.time()
    study = run_search(args.model, comp, space, X, y, args.trials, args.hours, TUNE_SEEDS,
                       storage=f"sqlite:///{storage_dir / f'{args.model}.db'}")
    best_params, best_score = recheck(comp, comp.params, study, X, y, TUNE_SEEDS)
    elapsed = time.time() - t0

    name = args.out or f"tune-{args.model}"
    note = (f"Tuned {args.model} params (Optuna TPE, seeds {TUNE_SEEDS}, "
           f"{len(study.trials)} trials in {elapsed:.0f}s, "
           f"top-{TOP_K} re-check best fold-mean {best_score:.5f}).")
    path = write_config(args.model, best_params, name, note)
    print(f"{len(study.trials)} trials in {elapsed:.0f}s; best re-checked score {best_score:.5f}")
    print(f"Wrote {path.relative_to(paths.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
