"""Checks on harness/combiner.py and its wiring -- synthetic probabilities only, no competition
data needed. python -m unittest discover -s tests -t ."""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from harness import combiner, config, paths, registry, score, split
from harness.score import blend
from scripts import matrix

WEIGHTS = {"a": 1.0, "b": 3.0}


def synthetic(n=600, seed=0):
    """Two models' out-of-fold probabilities for one seed, a target, and a 3-way segment."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    seg = rng.choice(["x", "y", "z"], n, p=[0.5, 0.45, 0.05])
    noise = lambda s: rng.normal(0, s, n)
    pa = 1 / (1 + np.exp(-((2 * y - 1) * 0.8 + noise(1.5))))
    pb = 1 / (1 + np.exp(-((2 * y - 1) * 0.5 + noise(1.0))))
    return {"a": pa, "b": pb}, y, seg


class SpecTest(unittest.TestCase):
    def test_default_is_mean(self):
        self.assertEqual(combiner.validate(None), {"method": "mean"})
        self.assertEqual(combiner.validate({}), {"method": "mean"})

    def test_gate_fills_its_defaults(self):
        self.assertEqual(combiner.validate({"method": "gate", "segment": "Deck"}),
                         {"method": "gate", "segment": "Deck", "min_rows": 200})

    def test_bad_specs_are_errors(self):
        for bad in ({"method": "rank_mean"}, {"method": "mean", "C": 1}, {"method": "gate"},
                    {"method": "gate", "segment": "Deck", "min_rows": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                combiner.validate(bad)

    def test_gate_segment_must_be_a_matrix_column(self):
        spec = combiner.validate({"method": "gate", "segment": "Deck"})
        with self.assertRaisesRegex(ValueError, "not a column"):
            combiner.segment_values(spec, pd.DataFrame({"Age": [1.0]}))
        self.assertEqual(list(combiner.segment_values(spec, pd.DataFrame({"Deck": ["A", "B"]}))), ["A", "B"])
        self.assertIsNone(combiner.segment_values({"method": "stack"}, pd.DataFrame()))


class FixedMethodsTest(unittest.TestCase):
    def setUp(self):
        self.probs, self.y, _ = synthetic()

    def test_mean_is_exactly_the_harness_blend(self):
        got = combiner.fit({"method": "mean"}, WEIGHTS, self.probs, self.y).apply(self.probs)
        want = blend({m: (p,) for m, p in self.probs.items()}, WEIGHTS)
        self.assertTrue(np.array_equal(got, want))

    def test_logit_mean_averages_logits_by_weight(self):
        got = combiner.fit({"method": "logit_mean"}, WEIGHTS, self.probs, self.y).apply(self.probs)
        z = (1 * combiner.logit(self.probs["a"]) + 3 * combiner.logit(self.probs["b"])) / 4
        np.testing.assert_allclose(got, 1 / (1 + np.exp(-z)))

    def test_a_fixed_combiner_ignores_the_target(self):
        a = combiner.fit({"method": "logit_mean"}, WEIGHTS, self.probs, self.y).apply(self.probs)
        b = combiner.fit({"method": "logit_mean"}, WEIGHTS, self.probs, 1 - self.y).apply(self.probs)
        self.assertTrue(np.array_equal(a, b))


class LearnedMethodsTest(unittest.TestCase):
    def setUp(self):
        self.probs, self.y, self.seg = synthetic()
        self.stack = {"method": "stack"}
        self.gate = combiner.validate({"method": "gate", "segment": "s", "min_rows": 100})

    def test_stack_fit_learns_from_the_target_and_apply_does_not(self):
        f = combiner.fit(self.stack, WEIGHTS, self.probs, self.y)
        again = combiner.fit(self.stack, WEIGHTS, self.probs, self.y)
        np.testing.assert_allclose(f.apply(self.probs), again.apply(self.probs))
        flipped = combiner.fit(self.stack, WEIGHTS, self.probs, 1 - self.y)
        self.assertGreater(np.abs(f.apply(self.probs) - flipped.apply(self.probs)).max(), 0.1)
        # apply on fresh rows reads probabilities only: no target argument exists
        fresh, _, _ = synthetic(n=50, seed=9)
        self.assertEqual(f.apply(fresh).shape, (50,))

    def test_stack_beats_chance_and_uses_the_logits(self):
        f = combiner.fit(self.stack, WEIGHTS, self.probs, self.y)
        self.assertGreater(((f.apply(self.probs) > 0.5) == self.y).mean(), 0.6)
        self.assertEqual(f.stacks[None].coef_.shape, (1, 2))
        self.assertEqual(f.stacks[None].C, 1.0)

    def test_weights_are_on_off_only_for_learned_methods(self):
        a = combiner.fit(self.stack, {"a": 1.0, "b": 1.0}, self.probs, self.y).apply(self.probs)
        b = combiner.fit(self.stack, {"a": 5.0, "b": 0.2}, self.probs, self.y).apply(self.probs)
        np.testing.assert_allclose(a, b)

    def test_gate_fits_a_stack_per_big_segment_and_falls_back_for_small_ones(self):
        f = combiner.fit(self.gate, WEIGHTS, self.probs, self.y, self.seg)
        self.assertEqual({k for k in f.stacks if k is not None}, {"x", "y"})  # z has < 100 rows
        out = f.apply(self.probs, self.seg)
        glob = f.stacks[None].predict_proba(np.column_stack([combiner.logit(self.probs[m]) for m in WEIGHTS]))[:, 1]
        small = self.seg == "z"
        np.testing.assert_allclose(out[small], glob[small])              # small segment: the global stack
        self.assertGreater(np.abs(out[~small] - glob[~small]).max(), 0)  # big ones: their own

    def test_gate_sends_an_unseen_segment_to_the_global_stack(self):
        f = combiner.fit(self.gate, WEIGHTS, self.probs, self.y, self.seg)
        fresh = {m: p[:5] for m, p in self.probs.items()}
        np.testing.assert_allclose(f.apply(fresh, np.array(["new"] * 5)),
                                   f.apply(fresh, np.array(["z"] * 5)))

    def test_gate_skips_a_single_class_segment(self):
        y = self.y.copy()
        y[self.seg == "y"] = 1
        f = combiner.fit(self.gate, WEIGHTS, self.probs, y, self.seg)
        self.assertNotIn("y", f.stacks)


class NestingTest(unittest.TestCase):
    """oof_combine meta-fits each outer fold on the other folds only."""
    def setUp(self):
        self.probs, self.y, self.seg = synthetic(n=500)
        self.oof = {m: np.array([p, p, p]) for m, p in self.probs.items()}  # 3 seeds, same rows

    def test_a_held_out_folds_labels_do_not_reach_its_own_predictions(self):
        gate = combiner.validate({"method": "gate", "segment": "s", "min_rows": 50})
        for spec, seg in (({"method": "stack"}, None), (gate, self.seg)):
            base = combiner.oof_combine(spec, WEIGHTS, self.oof, self.y, split.SEEDS, seg)
            va = split.folds(self.y, split.SEEDS[0])[0][1]
            y2 = self.y.copy()
            y2[va] = 1 - y2[va]
            fixed = {seed: split.folds(self.y, seed) for seed in split.SEEDS}  # flipping y would re-stratify
            with mock.patch.object(split, "folds", lambda y, seed: fixed[seed]):
                moved = combiner.oof_combine(spec, WEIGHTS, self.oof, y2, split.SEEDS, seg)
            np.testing.assert_array_equal(base[0][va], moved[0][va], err_msg=spec["method"])
            rest = ~np.isin(np.arange(500), va)
            self.assertGreater(np.abs(base[0][rest] - moved[0][rest]).max(), 0)

    def test_every_dev_row_is_scored_once_per_seed(self):
        out = combiner.oof_combine({"method": "stack"}, WEIGHTS, self.oof, self.y)
        self.assertEqual(out.shape, (3, 500))
        self.assertFalse(np.isnan(out).any())

    def test_fixed_methods_skip_the_fold_loop(self):
        out = combiner.oof_combine({"method": "mean"}, WEIGHTS, self.oof, self.y)
        np.testing.assert_array_equal(out, blend({m: (p,) for m, p in self.oof.items()}, WEIGHTS))

    def test_test_predictions_use_one_meta_fit_per_seed_on_all_dev_rows(self):
        rng = np.random.default_rng(3)
        test = {m: rng.uniform(0.05, 0.95, (3, 40)) for m in WEIGHTS}
        out = combiner.combine_test({"method": "stack"}, WEIGHTS, self.oof, test, self.y)
        self.assertEqual(out.shape, (3, 40))
        f = combiner.fit({"method": "stack"}, WEIGHTS, {m: self.oof[m][0] for m in WEIGHTS}, self.y)
        np.testing.assert_allclose(out[0], f.apply({m: test[m][0] for m in WEIGHTS}))

    def test_fixed_test_predictions_are_the_blend(self):
        test = {m: np.random.default_rng(4).uniform(size=(3, 7)) for m in WEIGHTS}
        out = combiner.combine_test({"method": "mean"}, WEIGHTS, self.oof, test, self.y)
        np.testing.assert_array_equal(out, blend({m: (p,) for m, p in test.items()}, WEIGHTS))


class IdentityTest(unittest.TestCase):
    keys, weights = {"a": "k1"}, {"a": 1.0}

    def test_mean_keeps_the_digest_it_always_had(self):
        self.assertEqual(score.digest(self.keys, self.weights),
                         score.digest(self.keys, self.weights, {"method": "mean"}))

    def test_other_methods_and_params_change_the_digest(self):
        seen = {score.digest(self.keys, self.weights, s) for s in (
            {"method": "mean"}, {"method": "logit_mean"}, {"method": "stack"},
            {"method": "gate", "segment": "Deck", "min_rows": 200},
            {"method": "gate", "segment": "Side", "min_rows": 200})}
        self.assertEqual(len(seen), 5)

    def test_fingerprint_covers_prediction_code_only(self):
        hashed = set(score.FINGERPRINTED)
        self.assertTrue({"data.py", "split.py", "registry.py", "config.py", "fit.py"} <= hashed)
        for name in ("combiner.py", "score.py", "submit.py", "audit.py", "paths.py"):
            self.assertNotIn(name, hashed)
        for name in hashed:
            self.assertTrue((paths.ROOT / "harness" / name).exists(), name)


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comps = registry.discover()

    def load(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=paths.CONFIGS, delete=False) as f:
            f.write(text)
        try:
            return config.load(f.name, self.comps)
        finally:
            os.unlink(f.name)

    def test_main_is_mean(self):
        self.assertEqual(config.load(paths.CONFIGS / "main.toml", self.comps).combiner, {"method": "mean"})

    def test_a_combiner_table_extends_main(self):
        cfg = self.load('extends = "main"\n[combiner]\nmethod = "gate"\nsegment = "Deck"\n')
        self.assertEqual(cfg.combiner, {"method": "gate", "segment": "Deck", "min_rows": 200})
        self.assertEqual(cfg.weights, config.load(paths.CONFIGS / "main.toml", self.comps).weights)

    def test_a_child_combiner_replaces_rather_than_merges(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=paths.CONFIGS, delete=False) as f:
            f.write('extends = "main"\n[combiner]\nmethod = "gate"\nsegment = "Deck"\nmin_rows = 50\n')
        try:
            cfg = self.load(f'extends = "{os.path.basename(f.name)}"\n[combiner]\nmethod = "stack"\n')
        finally:
            os.unlink(f.name)
        self.assertEqual(cfg.combiner, {"method": "stack"})

    def test_unknown_method_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "combiner.method"):
            self.load('extends = "main"\n[combiner]\nmethod = "vote"\n')


class MatrixTest(unittest.TestCase):
    def test_combiner_ideas_merge_into_one_table_and_exclude_each_other(self):
        merged = matrix.merge({"f": {"features": {"x": True}},
                               "g": {"combiner": {"method": "stack"}}})
        self.assertEqual(merged["combiner"], {"method": "stack"})
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            matrix.merge({"g": {"combiner": {"method": "stack"}}, "h": {"combiner": {"method": "logit_mean"}}})

    def test_written_cell_carries_the_combiner_table(self):
        path = matrix.write_cell(("zz_test",), matrix.merge(
            {"g": {"combiner": {"method": "gate", "segment": "Deck"}}}))
        try:
            self.assertIn("[combiner]", path.read_text())
            self.assertEqual(config.load(path, registry.discover()).combiner["segment"], "Deck")
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
