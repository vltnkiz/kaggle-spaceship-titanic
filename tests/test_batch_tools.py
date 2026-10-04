"""Checks on the batch tooling built atop the frozen harness (scripts/matrix.py,
scripts/confirm.py, scripts/base_lb.py) -- pure logic only, no competition data needed.
python -m unittest discover -s tests -t ."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import paths
from scripts import base_lb, confirm, matrix, tune


class MatrixCellsTest(unittest.TestCase):
    def test_all_off_cell_is_first(self):
        self.assertEqual(matrix.cells(["a", "b"], [])[0], ())

    def test_every_subset_present_without_exclusions(self):
        got = set(matrix.cells(["a", "b", "c"], []))
        want = {(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")}
        self.assertEqual(got, want)

    def test_excluded_pair_and_supersets_are_skipped(self):
        got = set(matrix.cells(["a", "b", "c"], [frozenset({"a", "b"})]))
        self.assertNotIn(("a", "b"), got)
        self.assertNotIn(("a", "b", "c"), got)
        self.assertIn(("a", "c"), got)
        self.assertIn(("b", "c"), got)


class MatrixMergeTest(unittest.TestCase):
    def test_merges_features_and_models_across_ideas(self):
        merged = matrix.merge({
            "a": {"features": {"age_bins": True}, "models": {}, "params": {}},
            "b": {"features": {"family": True}, "models": {"catboost": 1.0}, "params": {}},
        })
        self.assertEqual(merged["features"], {"age_bins": True, "family": True})
        self.assertEqual(merged["models"], {"catboost": 1.0})

    def test_params_overrides_merge_per_model(self):
        merged = matrix.merge({
            "a": {"features": {}, "models": {}, "params": {"catboost": {"depth": 6}}},
            "b": {"features": {}, "models": {}, "params": {"catboost": {"iterations": 500}}},
        })
        self.assertEqual(merged["params"], {"catboost": {"depth": 6, "iterations": 500}})

    def test_read_idea_rejects_a_config_not_extending_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            exp.mkdir()
            (exp / "bad.toml").write_text('extends = "batch-1"\n[features]\nx = true\n')
            with mock.patch.object(matrix.paths, "CONFIGS", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "extends"):
                    matrix.read_idea("bad")


class ConfirmDiffTest(unittest.TestCase):
    def _result(self, cv, per_fold):
        return {"config": "x", "cv": cv, "per_fold": per_fold}

    def test_keep_needs_the_same_bar_as_a_single_screen(self):
        base = self._result(0.800, [[0.8] * 5] * 3)
        result = self._result(0.803, [[0.803] * 5] * 3)
        out = confirm.diff(result, base)
        self.assertEqual(out["verdict"], "keep")

    def test_small_positive_delta_is_a_near_miss(self):
        base = self._result(0.800, [[0.8] * 5] * 3)
        result = self._result(0.8005, [[0.8005] * 5] * 3)
        out = confirm.diff(result, base)
        self.assertEqual(out["verdict"], "near_miss")

    def test_negative_delta_is_a_drop(self):
        base = self._result(0.800, [[0.8] * 5] * 3)
        result = self._result(0.795, [[0.795] * 5] * 3)
        out = confirm.diff(result, base)
        self.assertEqual(out["verdict"], "drop")


class BaseLbTest(unittest.TestCase):
    def test_write_then_read_round_trips_and_matches_the_current_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pinned = root / "main.json"
            pinned.write_text(json.dumps({"commit": "abc123", "cv": 0.812}))
            sidecar = root / "main.lb.json"
            with mock.patch.object(base_lb.paths, "PINNED", pinned), \
                 mock.patch.object(base_lb, "PATH", sidecar):
                self.assertTrue(base_lb.stale())  # nothing recorded yet
                base_lb.write(0.805, "test")
                self.assertFalse(base_lb.stale())
                self.assertEqual(base_lb.read()["public_score"], 0.805)

    def test_a_moved_pin_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pinned = root / "main.json"
            sidecar = root / "main.lb.json"
            with mock.patch.object(base_lb.paths, "PINNED", pinned), \
                 mock.patch.object(base_lb, "PATH", sidecar):
                pinned.write_text(json.dumps({"commit": "old", "cv": 0.80}))
                base_lb.write(0.80, "test")
                pinned.write_text(json.dumps({"commit": "new", "cv": 0.81}))
                self.assertTrue(base_lb.stale())


class TuneSpaceOverrideTest(unittest.TestCase):
    def test_override_keeps_kind_and_leaves_the_default_space_untouched(self):
        space = tune.override_space(tune.SPACES["catboost"], ["depth=3:5", "learning_rate=0.01:0.05"])
        self.assertEqual(space["depth"], ("int", 3, 5))
        self.assertEqual(space["learning_rate"], ("float_log", 0.01, 0.05))
        self.assertEqual(tune.SPACES["catboost"]["depth"], ("int", 4, 8))

    def test_unknown_param_or_malformed_range_is_rejected(self):
        for bad in ("nope=1:2", "depth=3", "depth=:5"):
            with self.assertRaises(ValueError):
                tune.override_space(tune.SPACES["catboost"], [bad])


if __name__ == "__main__":
    unittest.main()
