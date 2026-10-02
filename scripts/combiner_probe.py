"""Can a selector pick the rows where another model is right and the reference is wrong?

    python -m scripts.combiner_probe [--models catboost lgbm xgboost extratrees logreg]
                                     [--config configs/main.toml] [--null-draws 2] [--jobs 4]
    python -m scripts.combiner_probe --smoke

Measured over the models' cached out-of-fold predictions (dev rows, the harness's folds, seeds
0-2), the same inputs as `scripts.headroom`; the first model is the reference. Every rule is
NESTED: its threshold, k, weights, C or tree is fit on the four training folds' predictions only
(hyper-parameters by an inner 4-fold CV over those rows) and scored on the held-out fold. Rules:

  1. gated        flip the reference's vote when its confidence |p-0.5| < t and at least k other
                  models disagree with it; (t, k) maximises training accuracy ("none" is a candidate)
  2. stacker      on the models' logits plus the reference's confidence, the number of disagreeing
                  models and segment dummies (HomePlanet, Deck, CryoSleep, age band, group size,
                  zero spend): a plain logistic regression, one with pairwise interactions, a
                  depth-limited tree on the label, and a depth-limited tree that predicts "the
                  reference is wrong" and flips when it says so (the selector proper)
  3. per_segment  a logistic regression on the logits, fit only inside one of the segments where
                  the reference is weakest (Earth, Deck G, Deck E, age 0-12); outside it the
                  reference's vote stands. One row per segment and `per_segment_all` combining them
  4. conf_weights per-model weights by that model's own confidence bin: a logistic regression on
                  logit x bin indicators, and a vote weighted by each (model, bin)'s training accuracy

Reported per rule: accuracy (mean over seeds), the paired difference to the reference, folds
better/worse (of seeds x 5), its standard error over folds, and a NULL reading: the same
pipeline over fake models (the reference's vote flipped at each model's observed disagreement
rate within the reference's confidence bins, independent of the label, keeping the model's own
confidence), so `null-adjusted` is what the rule gains beyond what fitting noise gives.

Descriptive only: these are measurements, never CV scores or keep inputs. The training rows'
predictions are themselves out-of-fold (as in `headroom`'s cross-validated stacker). Reuses the
harness's building blocks and never edits `harness/`.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from harness import paths, registry, split
from scripts.headroom import DEFAULT_MODELS, load, logit, segments

T_GRID = tuple(np.round(np.arange(0.05, 0.501, 0.05), 2))
SEGMENTS = {"Earth": ("HomePlanet", "Earth"), "DeckG": ("Deck", "G"),
            "DeckE": ("Deck", "E"), "Age0-12": ("AgeBand", "0-12")}
STACK_SEGMENTS = ("HomePlanet", "Deck", "CryoSleep", "AgeBand", "GroupSize", "ZeroSpend")
CONF_BINS = 4


def lr(C, poly=False):
    steps = ([PolynomialFeatures(2, interaction_only=True, include_bias=False)] if poly else []) + [
        StandardScaler(), LogisticRegression(C=C, max_iter=300)]
    return make_pipeline(*steps)


def tree(depth_leaf):
    return DecisionTreeClassifier(max_depth=depth_leaf[0], min_samples_leaf=depth_leaf[1], random_state=0)


def fit_best(make, grid, X, y, tr):
    """The grid point with the best inner-CV accuracy over the training rows (first wins ties),
    refit on all of them."""
    inner = StratifiedKFold(4, shuffle=True, random_state=0)
    best, best_acc = grid[0], -1.0
    for g in grid:
        acc = np.mean([(make(g).fit(X[tr[a]], y[tr[a]]).predict(X[tr[b]]) == y[tr[b]]).mean()
                       for a, b in inner.split(tr, y[tr])])
        if acc > best_acc + 1e-12:
            best, best_acc = g, acc
    return make(best).fit(X[tr], y[tr])


def gated(ref_vote, conf, ndis, y, tr, va, n_others):
    """Flip rule (t, k) maximising training accuracy; no flip unless it strictly beats leaving the
    reference alone."""
    best, best_acc = None, float((ref_vote[tr] == y[tr]).mean())
    for t in T_GRID:
        for k in range(1, n_others + 1):
            flip = (conf[tr] < t) & (ndis[tr] >= k)
            acc = float(((ref_vote[tr] ^ flip) == y[tr]).mean())
            if acc > best_acc + 1e-12:
                best, best_acc = (t, k), acc
    if best is None:
        return ref_vote[va].copy()
    t, k = best
    return ref_vote[va] ^ ((conf[va] < t) & (ndis[va] >= k))


def bin_edges(p, tr):
    return np.quantile(np.abs(p[tr] - 0.5), np.linspace(0, 1, CONF_BINS + 1)[1:-1])


def bin_features(Z, P, tr, rows):
    """logit x own-confidence-bin indicators (one block per model)."""
    cols, bins = [], []
    for m in range(P.shape[1]):
        which = np.searchsorted(bin_edges(P[:, m], tr), np.abs(P[rows, m] - 0.5), side="right")
        bins.append(which)
        for b in range(CONF_BINS):
            cols.append(Z[rows, m] * (which == b))
    return np.column_stack(cols), np.column_stack(bins)


def fold_votes(Z, P, y, dummies, seg, tr, va):
    """{rule: bool votes for the va rows}, everything fit on the tr rows."""
    n_models = P.shape[1]
    ref_vote = P[:, 0] > 0.5
    conf = np.abs(P[:, 0] - 0.5)
    ndis = ((P[:, 1:] > 0.5) != ref_vote[:, None]).sum(axis=1)
    numeric = np.column_stack([Z, conf, ndis])
    X = np.column_stack([numeric, dummies])
    out = {"reference": ref_vote[va], "logit_mean": Z[va].mean(axis=1) > 0}
    out["gated"] = gated(ref_vote, conf, ndis, y, tr, va, n_models - 1)

    out["stack_logits"] = fit_best(lr, (0.01, 0.1, 1, 10), Z, y, tr).predict(Z[va]).astype(bool)
    out["stack_meta"] = fit_best(lr, (0.01, 0.1, 1), X, y, tr).predict(X[va]).astype(bool)
    out["stack_meta_interact"] = fit_best(lambda C: lr(C, poly=True), (0.001, 0.003, 0.01), X, y, tr
                                          ).predict(X[va]).astype(bool)
    grid = [(d, leaf) for d in (2, 3, 4) for leaf in (50, 150)]
    out["tree_label"] = fit_best(tree, grid, X, y, tr).predict(X[va]).astype(bool)
    wrong = (ref_vote != y).astype(int)
    sel = fit_best(tree, grid, X, wrong, tr)
    out["tree_selector"] = ref_vote[va] ^ (sel.predict_proba(X[va])[:, -1] > 0.5)
    if wrong[tr].min() == wrong[tr].max():  # degenerate training fold: tree has one class
        out["tree_selector"] = ref_vote[va].copy()

    per = ref_vote.copy()
    chosen = {}
    for name, (key, value) in SEGMENTS.items():
        inside = (seg[key] == value).to_numpy()
        rows = tr[inside[tr]]
        vote = ref_vote[va].copy()
        if len(rows) > 50 and y[rows].min() != y[rows].max():
            Xs = np.column_stack([Z, conf, ndis])
            model = fit_best(lr, (0.01, 0.1, 1, 10), Xs, y, rows)
            hit = inside[va]
            vote[hit] = model.predict(Xs[va][hit]).astype(bool)
        out[f"per_segment_{name}"] = vote
        chosen[name] = (inside[va], vote)
    combined = ref_vote[va].copy()
    for name, (inside, vote) in sorted(chosen.items(), key=lambda kv: kv[1][0].sum()):
        combined[inside] = vote[inside]
    out["per_segment_all"] = combined

    Fb_tr, _ = bin_features(Z, P, tr, np.arange(len(y)))
    out["conf_weights_lr"] = fit_best(lr, (0.01, 0.1, 1), Fb_tr, y, tr).predict(Fb_tr[va]).astype(bool)
    _, bins = bin_features(Z, P, tr, np.arange(len(y)))
    score = np.zeros(len(va))
    for m in range(n_models):
        for b in range(CONF_BINS):
            cell = tr[bins[tr, m] == b]
            right = float(((P[cell, m] > 0.5) == y[cell]).mean()) if len(cell) else 0.5
            w = max(float(logit(np.array(min(max(right, 1e-3), 1 - 1e-3)))), 0.0)
            at = bins[va, m] == b
            score[at] += w * np.where(P[va, m][at] > 0.5, 1.0, -1.0)
    out["conf_weights_acc"] = np.where(score == 0, ref_vote[va], score > 0)
    return out


def run(oof, y, models, dummies, seg, jobs):
    """{rule: votes (seeds x rows)} for the given predictions."""
    P_all = [np.column_stack([oof[m][s] for m in models]) for s in range(len(split.SEEDS))]
    Z_all = [logit(P) for P in P_all]
    tasks = [(s, tr, va) for s, seed in enumerate(split.SEEDS) for tr, va in split.folds(y, seed)]
    res = Parallel(n_jobs=jobs)(delayed(fold_votes)(Z_all[s], P_all[s], y, dummies, seg, tr, va)
                                for s, tr, va in tasks)
    votes = {}
    for (s, _, va), got in zip(tasks, res):
        for rule, v in got.items():
            votes.setdefault(rule, np.zeros((len(split.SEEDS), len(y)), dtype=bool))[s, va] = v
    return votes, tasks


def paired(votes, tasks, y):
    """{rule: (acc, delta vs reference, better folds, worse folds, n folds, SE of fold deltas)}."""
    ref = votes["reference"]
    base = float((ref == y).mean())
    out = {}
    for rule, v in votes.items():
        diffs = np.array([((v[s, va] == y[va]).mean() - (ref[s, va] == y[va]).mean()) * len(va)
                          for s, _, va in tasks]) / np.array([len(va) for _, _, va in tasks])
        out[rule] = {"acc": float((v == y).mean()), "delta": float((v == y).mean()) - base,
                     "better": int((diffs > 1e-12).sum()), "worse": int((diffs < -1e-12).sum()),
                     "folds": len(diffs), "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs)))}
    return out


def fake_models(oof, models, rng, bins=8):
    """Models with no information beyond the reference: its vote flipped at the rate each real model
    disagrees with it inside the reference's confidence bin, independent of the label; each
    keeps the real model's own confidence."""
    ref = oof[models[0]]
    ref_logit = logit(ref)
    conf = np.abs(ref - 0.5)
    edges = np.quantile(conf, np.linspace(0, 1, bins + 1)[1:-1])
    which = np.searchsorted(edges, conf, side="right")
    out = {models[0]: ref}
    for m in models[1:]:
        disagree = (oof[m] > 0.5) != (ref > 0.5)
        rate = np.array([disagree[which == b].mean() for b in range(bins)])[which]
        sign = np.where(rng.random(ref.shape) < rate, -np.sign(ref_logit), np.sign(ref_logit))
        out[m] = 1 / (1 + np.exp(-sign * np.abs(logit(oof[m]))))
    return out


def probe(oof, y, frame, models, null_draws, jobs):
    seg = segments(frame)
    dummies = pd.get_dummies(pd.DataFrame({k: seg[k] for k in STACK_SEGMENTS})).to_numpy(float)
    votes, tasks = run(oof, y, models, dummies, seg, jobs)
    real = paired(votes, tasks, y)
    nulls = []
    for d in range(null_draws):
        fake = fake_models(oof, models, np.random.default_rng(d))
        nv, nt = run(fake, y, models, dummies, seg, jobs)
        nulls.append({r: v["delta"] for r, v in paired(nv, nt, y).items()})
        print(f"  null draw {d + 1}/{null_draws} done", file=sys.stderr)
    rows = []
    for rule, v in real.items():
        null = float(np.mean([n[rule] for n in nulls])) if nulls else float("nan")
        rows.append({"rule": rule, **v, "null_delta": null, "null_adjusted": v["delta"] - null})
    return pd.DataFrame(rows).set_index("rule")


def report(table: pd.DataFrame, ref: str) -> dict:
    base = float(table.loc["reference", "acc"])
    show = table.copy()
    show["better/worse"] = [f"{b}/{w} of {f}" for b, w, f in zip(show.better, show.worse, show.folds)]
    cols = ["acc", "delta", "better/worse", "se", "null_delta", "null_adjusted"]
    print(f"\n== Nested-CV rules vs {ref} alone ({base:.5f}); accuracy = mean over seeds ==")
    print(show[cols].to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.5f}"))
    rules = table.drop(index=["reference", "logit_mean", "stack_logits"])
    best = rules["delta"].idxmax()
    top = rules.loc[best]
    print(f"\nBEST rule: {best}  {top.acc:.5f}  [{top.delta:+.5f} vs {ref}; better/worse "
          f"{int(top.better)}/{int(top.worse)} of {int(top.folds)}; null-adjusted {top.null_adjusted:+.5f}]")
    print("(logit_mean and stack_logits are the fixed / linear baselines, not among the four "
          "families; stack_*, tree_* are family 2. The verdict reads the raw delta: the null models "
          "carry no signal, so a negative null_delta is fitting cost and null_adjusted only says how "
          "much of that cost the real predictions avoid.)")
    return {"reference": base, "best": best, "table": table.to_dict("index")}


def smoke() -> None:
    """Seconds-long check on synthetic predictions: every rule runs, nothing is read from disk."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n).astype(bool)
    oof = {m: np.clip(y * 0.25 + rng.random((len(split.SEEDS), n)) * 0.75, 0.01, 0.99)
           for m in ("a", "b", "c")}
    frame = pd.DataFrame({
        "CryoSleep": rng.choice([True, False], n), "TotalSpend": rng.integers(0, 3, n) * 100,
        "GroupSize": rng.integers(1, 6, n), "Deck": rng.choice(list("EFG"), n),
        "Side": rng.choice(list("PS"), n), "HomePlanet": rng.choice(["Earth", "Mars"], n),
        "Age": rng.integers(1, 70, n).astype(float)})
    table = probe(oof, y, frame, ["a", "b", "c"], null_draws=1, jobs=1)
    print(json.dumps({"smoke": "combiner_probe", "rules": len(table), "best": float(
        table.drop(index=["reference"])["delta"].max())}))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.combiner_probe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                    help="model components; the first is the reference")
    ap.add_argument("--config", default=str(paths.CONFIGS / "main.toml"),
                    help="features come from this config (default: main)")
    ap.add_argument("--null-draws", type=int, default=2, help="fake-model repeats for the null reading")
    ap.add_argument("--jobs", type=int, default=1, help="parallel (seed, fold) tasks")
    ap.add_argument("--json", default=None, help="also write the numbers to this file")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    if args.smoke:
        return smoke()
    models = list(dict.fromkeys(args.models))
    if len(models) < 2:
        sys.exit("--models needs two or more models")
    oof, frame, y = load(tuple(models), args.config, registry.discover())
    out = report(probe(oof, y, frame, models, args.null_draws, args.jobs), models[0])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
