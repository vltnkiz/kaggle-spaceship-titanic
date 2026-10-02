"""How much room is there for a better combiner over models we already have?

    python -m scripts.headroom [--models catboost lgbm xgboost extratrees logreg]
                               [--config configs/main.toml] [--subsets catboost+xgboost ...]
    python -m scripts.headroom --smoke

Measured over the models' cached out-of-fold predictions (dev rows, the harness's folds, seeds
0-2): each model is built from `--config`'s features and its own default parameters, so the
cache entries are the ones a batch already wrote; a miss trains the model as `harness.score`
would. The first model is the reference everything is compared against.

Four tables, all means over the three seeds:
  1. disagreement  pairwise disagreement at the 0.5 cut, and the share of rows where any
                   two models disagree
  2. oracle        accuracy if every row picked whichever model is right (the ceiling for any
                   per-row scheme), for all models and for every subset holding the reference
  3. segments      per-segment accuracy per model, the rows where the reference is wrong but
                   another model is right, and the in-sample accuracy of choosing the best
                   model per segment
  4. combiners     logit-mean, rank-mean, an IN-SAMPLE stacker (logistic regression on the
                   logits, fit on the very predictions it is scored on) and, for contrast, the
                   same stacker cross-validated over the predictions

Descriptive only: nothing here is a CV score, a verdict or a keep input. The oracle and the
in-sample stacker are upper bounds by construction; the segment-wise choice is in-sample. Rank-mean
is transductive (ranks and cut come from the validation fold's own predictions, label-free), so
it too is read as indicative. Reuses the harness's building blocks and never edits `harness/`.
"""
import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from harness import config, data, paths, registry, split
from harness.score import build_matrix, predictions

DEFAULT_MODELS = ("catboost", "lgbm", "xgboost", "extratrees", "logreg")
EPS = 1e-6
AGE_BINS = [-1, 12, 17, 25, 40, 60, np.inf]
AGE_LABELS = ["0-12", "13-17", "18-25", "26-40", "41-60", "61+"]


def load(models: tuple, config_path, comps: dict):
    """({model: oof (seeds x rows)}, dev frame with the features built, y)."""
    base = config.load(config_path, comps)
    cfg = config.Config(base.name, base.features, {m: 1.0 for m in models},
                        {m: {**comps[m].params, **base.params.get(m, {})} for m in models})
    preds, _, y = predictions(cfg, comps)
    dev, _, _ = data.load_dev()
    frame = dev
    for name in cfg.features:
        frame = comps[name].build(frame.copy())
    return {m: preds[m][0] for m in models}, frame, y


def hits(oof: dict, y: np.ndarray) -> dict:
    """{model: bool (seeds x rows)}: was the model right at the 0.5 cut."""
    return {m: (p > 0.5) == y for m, p in oof.items()}


def disagreement(oof: dict) -> tuple[pd.DataFrame, float]:
    models = list(oof)
    votes = {m: p > 0.5 for m, p in oof.items()}
    pair = pd.DataFrame(index=models, columns=models, dtype=float)
    for a, b in itertools.product(models, models):
        pair.loc[a, b] = float((votes[a] != votes[b]).mean())
    stack = np.stack(list(votes.values()))
    return pair, float((stack.any(axis=0) & ~stack.all(axis=0)).mean())


def oracle(hit: dict, models: list) -> float:
    return float(np.stack([hit[m] for m in models]).any(axis=0).mean())


def subsets(models: list, ref: str, explicit: list | None) -> list[list[str]]:
    if explicit:
        return [s.split("+") for s in explicit]
    others = [m for m in models if m != ref]
    return [[ref, *c] for r in range(1, len(others) + 1) for c in itertools.combinations(others, r)]


def oracle_table(hit: dict, models: list, ref: str, explicit: list | None) -> pd.DataFrame:
    rows = [{"models": m, "accuracy": float(hit[m].mean())} for m in models]
    rows += [{"models": "+".join(s), "oracle": oracle(hit, s)} for s in subsets(models, ref, explicit)]
    rows = pd.DataFrame(rows).set_index("models")
    rows["vs_" + ref] = rows[["accuracy", "oracle"]].max(axis=1) - float(hit[ref].mean())
    return rows.sort_values(["oracle", "accuracy"], ascending=False, na_position="last")


def segments(frame: pd.DataFrame) -> dict:
    """Segmentations of the dev rows, each a label per row."""
    spend = frame["TotalSpend"] if "TotalSpend" in frame else frame[
        ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
    group_size = frame["GroupSize"] if "GroupSize" in frame else frame.groupby(
        frame["PassengerId"].str.split("_").str[0])["PassengerId"].transform("size")
    seg = {
        "CryoSleep": frame["CryoSleep"].astype(str),
        "ZeroSpend": (spend == 0).map({True: "no spend", False: "spends"}),
        "Deck": frame["Deck"].astype(str) if "Deck" in frame else frame["Cabin"].str[0].astype(str),
        "Side": frame["Side"].astype(str) if "Side" in frame else frame["Cabin"].str[-1].astype(str),
        "HomePlanet": frame["HomePlanet"].astype(str),
        "AgeBand": pd.cut(frame["Age"], AGE_BINS, labels=AGE_LABELS).astype(str),
        "GroupSize": group_size.clip(upper=4).map(lambda n: f"{n}+" if n >= 4 else str(n)),
    }
    return {k: pd.Series(np.asarray(v, dtype=object)).fillna("missing").astype(str)
            for k, v in seg.items()}


def segment_table(hit: dict, models: list, ref: str, labels: pd.Series) -> pd.DataFrame:
    others = [m for m in models if m != ref]
    rows = []
    for seg in sorted(labels.unique()):
        idx = (labels == seg).to_numpy()
        row = {"segment": seg, "n": int(idx.sum())}
        row.update({m: float(hit[m][:, idx].mean()) for m in models})
        row["oracle"] = float(np.stack([hit[m][:, idx] for m in models]).any(axis=0).mean())
        row["any_disagree"] = float((np.stack([hit[m][:, idx] for m in models]).any(axis=0)
                                     & ~np.stack([hit[m][:, idx] for m in models]).all(axis=0)).mean())
        saved = ~hit[ref][:, idx] & np.stack([hit[m][:, idx] for m in others]).any(axis=0) \
            if others else np.zeros_like(hit[ref][:, idx])
        row["ref_wrong_other_right"] = float(saved.mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


def segmentwise_choice(table: pd.DataFrame, models: list, ref: str) -> tuple[float, float]:
    """(in-sample accuracy of picking the best model per segment, the reference's own)."""
    n = table["n"].to_numpy()
    best = table[models].max(axis=1).to_numpy()
    return float((best * n).sum() / n.sum()), float((table[ref].to_numpy() * n).sum() / n.sum())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logit_mean(oof: dict, models: list) -> np.ndarray:
    return np.mean([logit(oof[m]) for m in models], axis=0)


def rank_mean_votes(oof: dict, models: list, y: np.ndarray) -> np.ndarray:
    """Per validation fold: average the models' within-fold ranks, then cut so the share
    predicted positive equals the models' mean share at 0.5. Label-free but transductive."""
    out = np.zeros((len(split.SEEDS), len(y)), dtype=bool)
    for s, seed in enumerate(split.SEEDS):
        for _, va in split.folds(y, seed):
            ranks = np.mean([pd.Series(oof[m][s, va]).rank(pct=True).to_numpy() for m in models], axis=0)
            share = np.mean([(oof[m][s, va] > 0.5).mean() for m in models])
            k = int(round(share * len(va)))
            order = np.argsort(-ranks, kind="stable")
            vote = np.zeros(len(va), dtype=bool)
            vote[order[:k]] = True
            out[s, va] = vote
    return out


def stacker(oof: dict, models: list, y: np.ndarray, cross_validated: bool) -> np.ndarray:
    """Logistic regression on the models' logits, fit on the out-of-fold predictions. In-sample
    scores the rows it was fit on (an upper bound); cross-validated scores each fold with a
    stacker fit on the other four."""
    out = np.zeros((len(split.SEEDS), len(y)), dtype=bool)
    for s, seed in enumerate(split.SEEDS):
        Z = np.column_stack([logit(oof[m][s]) for m in models])
        if cross_validated:
            for tr, va in split.folds(y, seed):
                out[s, va] = LogisticRegression(C=1e6, max_iter=1000).fit(Z[tr], y[tr]).predict(Z[va])
        else:
            out[s] = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y).predict(Z)
    return out


def acc(vote: np.ndarray, y: np.ndarray) -> float:
    return float((vote == y).mean())


def combiner_table(oof: dict, hit: dict, models: list, ref: str, y: np.ndarray,
                   explicit: list | None) -> pd.DataFrame:
    rows = []
    for s in subsets(models, ref, explicit) + ([models] if not explicit and len(models) > 2 else []):
        rows.append({
            "models": "+".join(s),
            "prob_mean": acc(np.mean([oof[m] for m in s], axis=0) > 0.5, y),
            "logit_mean": acc(logit_mean(oof, s) > 0, y),
            "rank_mean": acc(rank_mean_votes(oof, s, y), y),
            "stacker_CV": acc(stacker(oof, s, y, True), y),
            "stacker_IN_SAMPLE_UPPER_BOUND": acc(stacker(oof, s, y, False), y),
        })
    table = pd.DataFrame(rows).set_index("models").drop_duplicates()
    return table.sort_values("stacker_IN_SAMPLE_UPPER_BOUND", ascending=False)


def report(oof: dict, frame: pd.DataFrame, y: np.ndarray, models: list, explicit: list | None) -> dict:
    ref = models[0]
    hit = hits(oof, y)
    fmt = lambda t: t.to_string(float_format=lambda v: f"{v:.4f}")  # noqa: E731
    out = {}

    pair, any_dis = disagreement(oof)
    print(f"\n== 1. Disagreement at 0.5 (mean over {len(split.SEEDS)} seeds, {len(y)} dev rows) ==")
    print(fmt(pair))
    print(f"share of rows where any two models disagree: {any_dis:.4f}")
    out["disagreement"] = {"pairwise": pair.to_dict(), "any_two": any_dis}

    otab = oracle_table(hit, models, ref, explicit)
    print(f"\n== 2. Oracle accuracy (upper bound for any per-row scheme); reference {ref} "
          f"= {hit[ref].mean():.5f} ==")
    full = oracle(hit, models)
    print(fmt(otab))
    print(f"all {len(models)} models: oracle {full:.4f}  (+{full - hit[ref].mean():.4f} over {ref})")
    out["oracle"] = {"table": otab.to_dict("index"), "all": full, "reference": float(hit[ref].mean())}

    print("\n== 3. Segments ==")
    out["segments"] = {}
    for name, labels in segments(frame).items():
        t = segment_table(hit, models, ref, labels)
        chosen, own = segmentwise_choice(t, models, ref)
        print(f"\n-- {name}: segment-wise best-model choice {chosen:.4f} (in-sample) vs {ref} {own:.4f}"
              f"  [{chosen - own:+.4f}]")
        print(fmt(t))
        out["segments"][name] = {"table": t.to_dict("index"), "segmentwise_in_sample": chosen,
                                 "reference": own}

    ctab = combiner_table(oof, hit, models, ref, y, explicit)
    print(f"\n== 4. Fixed and learned combiners (accuracy; reference {ref} = {hit[ref].mean():.5f}) ==")
    print("stacker_IN_SAMPLE_UPPER_BOUND is fit on the predictions it is scored on: never a CV score.")
    print(fmt(ctab))
    out["combiners"] = ctab.to_dict("index")

    top = ctab["stacker_IN_SAMPLE_UPPER_BOUND"].max()
    print(f"\nGAP vs {ref} ({hit[ref].mean():.5f}): oracle(all) {full:.5f} "
          f"[{full - hit[ref].mean():+.5f}], in-sample stacker (best subset) {top:.5f} "
          f"[{top - hit[ref].mean():+.5f}], cross-validated stacker (best subset) "
          f"{ctab['stacker_CV'].max():.5f} [{ctab['stacker_CV'].max() - hit[ref].mean():+.5f}]")
    return out


def smoke(comps: dict) -> None:
    """Seconds-long check on synthetic predictions: the tables compute, nothing is read from disk."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    oof = {m: np.clip(y * 0.3 + rng.random((len(split.SEEDS), n)) * 0.7, 0, 1)
           for m in ("a", "b", "c")}
    hit = hits(oof, y)
    pair, any_dis = disagreement(oof)
    ctab = combiner_table(oof, hit, list(oof), "a", y, None)
    print(json.dumps({"smoke": "headroom", "any_disagree": any_dis,
                      "oracle": oracle(hit, list(oof)), "stacker": float(
                          ctab["stacker_IN_SAMPLE_UPPER_BOUND"].max())}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.headroom", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    help="model components; the first is the reference")
    ap.add_argument("--config", default=str(paths.CONFIGS / "main.toml"),
                    help="features come from this config (default: main)")
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="model sets to report in tables 2 and 4, e.g. catboost+xgboost")
    ap.add_argument("--json", default=None, help="also write the numbers to this file")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    comps = registry.discover()
    if args.smoke:
        return smoke(comps)
    models = list(dict.fromkeys(args.models))
    if len(models) < 2:
        sys.exit("--models needs two or more models")
    oof, frame, y = load(tuple(models), args.config, comps)
    out = report(oof, frame, y, models, args.subsets)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
