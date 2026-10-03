"""Checks on the harness's guarantees.  python -m unittest discover -s tests -t ."""
import ast
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness import config, data, paths, registry, split

ROOT = paths.ROOT
HAS_DATA = (paths.raw_dir() / "train.csv").exists()


@unittest.skipUnless(HAS_DATA, "no competition data")
class HoldoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = data.read("train.csv")
        cls.mask = split.in_holdout(cls.train[data.ID])
        cls.dev, cls.y, _ = data.load_dev()

    def test_holdout_is_about_fifteen_percent(self):
        self.assertTrue(0.12 < self.mask.mean() < 0.18, self.mask.mean())

    def test_dev_rows_exclude_every_holdout_passenger(self):
        held = set(self.train.loc[self.mask, data.ID])
        self.assertEqual(len(self.dev), (~self.mask).sum())
        self.assertFalse(held & set(self.dev[data.ID]))

    def test_no_travel_group_straddles_the_carve(self):
        group = self.train[data.ID].str[:4]
        self.assertFalse(set(group[self.mask]) & set(group[~self.mask]))

    def test_every_dev_row_is_validated_exactly_once_per_seed(self):
        for seed in split.SEEDS:
            seen = np.concatenate([va for _, va in split.folds(self.y, seed)])
            self.assertEqual(sorted(seen), list(range(len(self.y))))

    def test_carve_is_fixed(self):
        again = split.in_holdout(self.train[data.ID].sample(frac=1, random_state=1))
        self.assertEqual(again.sum(), self.mask.sum())


class HoldoutAbsenceTest(unittest.TestCase):
    def test_only_the_loader_and_audit_select_holdout_rows(self):
        users = {p.name for p in (ROOT / "harness").glob("*.py")
                 if "in_holdout" in p.read_text() and p.name != "split.py"}
        self.assertEqual(users, {"data.py", "audit.py"})

    def test_nothing_imports_audit(self):
        for p in [*(ROOT / "harness").glob("*.py"), *(ROOT / "components").glob("*.py")]:
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                else:
                    continue
                self.assertFalse([n for n in names if "audit" in n], p.name)


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comps = registry.discover()

    def load(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", dir=paths.CONFIGS,
                                         delete=False) as f:
            f.write(text)
        try:
            return config.load(f.name, self.comps)
        finally:
            os.unlink(f.name)

    def test_main_switches_on_a_model(self):
        self.assertTrue(config.load(paths.CONFIGS / "main.toml", self.comps).weights)

    def test_example_is_off_in_main_and_on_in_one_line(self):
        main = config.load(paths.CONFIGS / "main.toml", self.comps)
        exp = config.load(paths.CONFIGS / "exp" / "age_bins.toml", self.comps)
        self.assertNotIn("age_bins", main.features)
        self.assertEqual(exp.features, (*main.features, "age_bins"))
        self.assertEqual(exp.weights, main.weights)

    def test_nothing_is_on_by_default(self):
        with self.assertRaisesRegex(ValueError, "no model"):
            self.load("")

    def test_unknown_component_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no component"):
            self.load('extends = "main"\n[features]\nnot_a_thing = true\n')

    def test_a_weight_of_zero_switches_a_model_off(self):
        # Not `extends = "main"`: this must hold regardless of which model main pins.
        with self.assertRaisesRegex(ValueError, "no model"):
            self.load('[models]\nlgbm = 0\n')


@unittest.skipUnless(shutil.which("git"), "no git")
class FreezeHookTest(unittest.TestCase):
    def test_hook_blocks_harness_and_allows_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            (repo / "hooks").mkdir()
            shutil.copy(ROOT / "hooks" / "pre-commit", repo / "hooks" / "pre-commit")
            os.chmod(repo / "hooks" / "pre-commit", 0o755)
            git("config", "core.hooksPath", "hooks")
            for d in ("harness", "components"):
                (repo / d).mkdir()
                (repo / d / "x.py").write_text("x = 1\n")

            git("add", "components/x.py")
            self.assertEqual(git("commit", "-qm", "component").returncode, 0)
            git("add", "harness/x.py")
            blocked = git("commit", "-qm", "harness")
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("harness/ is frozen", blocked.stderr)

    def test_harness_change_label_lets_a_harness_commit_through(self):
        # A stub `gh` stands in for GitHub: issue 7 carries the label, issue 8 does not.
        with tempfile.TemporaryDirectory() as tmp:
            repo, bin_dir = Path(tmp) / "repo", Path(tmp) / "bin"
            for d in (repo / "hooks", repo / "harness", bin_dir):
                d.mkdir(parents=True)
            gh = bin_dir / "gh"
            gh.write_text("#!/bin/sh\n"
                          '[ "$3" = "7" ] && printf "harness-change\nwayfinder:task\n"\n'
                          "exit 0\n", newline="\n")
            os.chmod(gh, 0o755)
            shutil.copy(ROOT / "hooks" / "pre-commit", repo / "hooks" / "pre-commit")
            os.chmod(repo / "hooks" / "pre-commit", 0o755)
            env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
            env.pop("HARNESS_CHANGE", None)

            def git(*a, **extra):
                return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True,
                                      env={**env, **extra})
            git("init", "-q")
            git("config", "user.email", "t@t")
            git("config", "user.name", "t")
            git("config", "core.hooksPath", "hooks")
            (repo / "harness" / "x.py").write_text("x = 1" + chr(10))
            git("add", "harness/x.py")

            self.assertNotEqual(git("commit", "-qm", "no env").returncode, 0)
            labelled_not = git("commit", "-qm", "unlabelled", HARNESS_CHANGE="8")
            self.assertNotEqual(labelled_not.returncode, 0)
            self.assertIn("does not carry", labelled_not.stderr)
            self.assertNotEqual(git("commit", "-qm", "junk", HARNESS_CHANGE="7; true").returncode, 0)
            self.assertEqual(git("commit", "-qm", "sanctioned", HARNESS_CHANGE="7").returncode, 0)


if __name__ == "__main__":
    unittest.main()
