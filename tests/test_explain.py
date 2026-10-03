"""Checks on scripts/explain.py -- synthetic frames and fake components only, no competition
data needed. python -m unittest discover -s tests -t ."""
import unittest

import numpy as np
import pandas as pd

from harness.config import Config
from harness.registry import Component
from scripts import explain


def feature(name, build):
    return Component(name, "feature", build)


def cfg(features=(), weights=None, combiner=None):
    weights = weights or {"m": 1.0}
    spec = {"method": "mean"} if combiner is None else combiner
    return Config("configs/exp/x.toml", tuple(features), weights, {m: {} for m in weights}, spec)


class GroupsTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"PassengerId": ["0001_01", "0002_01"],
                                   "Age": [30.0, 40.0], "Spa": [0.0, 5.0]})

    def test_raw_columns_go_to_base_and_added_columns_to_their_component(self):
        comps = {"older": feature("older", lambda f: f.assign(Old=f["Age"] > 35)),
                 "spa_log": feature("spa_log", lambda f: f.assign(SpaLog=np.log1p(f["Spa"])))}
        X, groups, notes = explain.groups(cfg(["older", "spa_log"]), comps, self.frame)
        self.assertEqual(groups, {"base": ["Age", "Spa"], "older": ["Old"], "spa_log": ["SpaLog"]})
        self.assertEqual(list(X.columns), ["Age", "Spa", "Old", "SpaLog"])
        self.assertEqual(notes, [])

    def test_a_rewritten_column_stays_with_its_creator_and_is_noted(self):
        comps = {"clip": feature("clip", lambda f: f.assign(Age=f["Age"].clip(upper=35),
                                                            Spa=f["Spa"].clip(upper=1)))}
        _, groups, notes = explain.groups(cfg(["clip"]), comps, self.frame)
        self.assertEqual(groups, {"base": ["Age", "Spa"]})
        self.assertEqual(notes, ["clip rewrites Age, Spa, credited to base"])


class Rule:
    """A fake fitted model: P(transported) is a fixed function of the matrix."""
    def __init__(self, rule):
        self.rule = rule

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(self.rule(X), dtype=float), 0.05, 0.95)
        return np.column_stack([1 - p, p])


def model(name, rule):
    return Component(name, "model", lambda params, seed: Rule(rule))


class ExplainTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 200
        self.y = rng.integers(0, 2, n)
        self.X = pd.DataFrame({"A": self.y.astype(float), "B": rng.normal(size=n),
                               "C": rng.normal(size=n)})

    def test_only_the_group_a_model_reads_gets_importance(self):
        comps = {"m": model("m", lambda X: X["A"])}
        out = explain.explain(cfg(), comps, self.X, self.y, {"base": ["A"], "noise": ["B", "C"]})
        self.assertGreater(out["component"]["m"]["base"]["logloss"][0], 0)
        self.assertGreater(out["component"]["m"]["base"]["accuracy"][0], 0)
        self.assertEqual(out["component"]["m"]["noise"]["logloss"], [0.0, 0.0])
        self.assertEqual(out["column"]["m"]["B"]["accuracy"], [0.0, 0.0])

    def test_a_group_is_shuffled_with_one_shared_permutation(self):
        X = self.X.assign(B=self.X["A"])  # the model reads only whether A and B agree
        comps = {"m": model("m", lambda X: X["A"] == X["B"])}
        out = explain.explain(cfg(), comps, X, np.ones(len(X), dtype=int), {"pair": ["A", "B"]})
        self.assertEqual(out["component"]["m"]["pair"]["logloss"], [0.0, 0.0])
        self.assertGreater(out["column"]["m"]["A"]["logloss"][0], 0)

    def test_blend_is_explained_only_when_two_models_are_on(self):
        one = {"m": model("m", lambda X: X["A"])}
        two = {**one, "n": model("n", lambda X: X["B"] > 0)}
        groups = {"base": ["A", "B", "C"]}
        self.assertEqual(set(explain.explain(cfg(), one, self.X, self.y, groups)["component"]), {"m"})
        out = explain.explain(cfg(weights={"m": 1.0, "n": 1.0}), two, self.X, self.y, groups)
        self.assertEqual(set(out["component"]), {"m", "n", "blend"})

    def test_blend_runs_through_a_learned_combiner_fit_on_the_other_folds(self):
        comps = {"m": model("m", lambda X: X["A"]), "n": model("n", lambda X: X["B"] > 0)}
        weights = {"m": 1.0, "n": 1.0}
        spec = {"method": "stack"}
        with self.assertRaisesRegex(ValueError, "needs `oof`"):
            explain.explain(cfg(weights=weights, combiner=spec), comps, self.X, self.y,
                            {"base": ["A", "B", "C"]})
        reads = {"m": self.X["A"].to_numpy(), "n": (self.X["B"] > 0).to_numpy(dtype=float)}
        oof = {m: np.clip(reads[m], 0.05, 0.95)[None, :] for m in weights}
        out = explain.explain(cfg(weights=weights, combiner=spec), comps, self.X, self.y,
                              {"base": ["A", "B", "C"]}, oof=oof)
        self.assertGreater(out["component"]["blend"]["base"]["logloss"][0], 0)


def reading(ll, acc):
    return {"logloss": [ll, 0.001], "accuracy": [acc, 0.002]}


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.result = {"config": "configs/exp/x.toml", "digest": "d1", "notes": [],
                       "component": {"m": {"base": reading(0.2, 0.05), "new": reading(0.01, 0.0)}},
                       "column": {"m": {"Age": reading(0.2, 0.05), "Spa": reading(0.0, 0.0),
                                        "New": reading(0.01, 0.0)}}}
        self.base = {"config": "configs/main.toml", "digest": "d0", "notes": [],
                     "component": {"m": {"base": reading(0.25, 0.06)}},
                     "column": {"m": {"Age": reading(0.2, 0.05), "Spa": reading(0.05, 0.01)}}}

    def test_a_component_missing_from_one_config_shows_a_dash_not_zero(self):
        out = explain.render(self.result, self.base)
        row = next(line for line in out.splitlines() if line.startswith("| new "))
        self.assertIn("—", row)
        self.assertNotIn("+0.0000 ±", row.split("|")[-2])

    def test_notes_are_labelled_with_their_config(self):
        self.base["notes"] = ["spend rewrites Spa, credited to base"]
        out = explain.render(self.result, self.base)
        self.assertIn("> note (main): spend rewrites Spa, credited to base", out.splitlines())

    def test_output_carries_no_verdict_wording(self):
        for out in (explain.render(self.result), explain.render(self.result, self.base)):
            for word in ("keep", "near-miss", "near_miss", "drop"):
                self.assertNotIn(word, out.lower())


if __name__ == "__main__":
    unittest.main()
