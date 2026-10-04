"""Checks on the wrapper logic of components/tabpfn.py and components/tabm.py -- the parts
the harness makes necessary (chunked predict, fixed vocabulary, numeric imputation). No
model is trained and no weights or token are needed.
python -m unittest discover -s tests -t ."""
import unittest

import numpy as np
import pandas as pd

from components import tabm, tabpfn


def frame(deck, age):
    return pd.DataFrame({"Deck": pd.Categorical(deck), "Age": age})


class ChunkedModel:
    """Stands in for a fitted TabPFNClassifier; records the size of every call."""

    def __init__(self):
        self.calls = []

    def predict_proba(self, X):
        self.calls.append(len(X))
        return np.tile([0.4, 0.6], (len(X), 1))


class TabPFNWrapperTest(unittest.TestCase):
    def fitted(self):
        est = tabpfn.build(dict(tabpfn.PARAMS), seed=0)
        est.categories = {"Deck": ["A", "B", "nan"]}
        est.model = ChunkedModel()
        return est

    def test_predict_is_chunked_and_complete(self):
        est = self.fitted()
        X = frame(["A"] * 2500, [30.0] * 2500)
        out = est.predict_proba(X)
        self.assertEqual(out.shape, (2500, 2))
        self.assertEqual(est.model.calls, [1000, 1000, 500])

    def test_unseen_level_falls_back_to_nan_level(self):
        est = self.fitted()
        prepared = est._prep(frame(["A", "Z", None], [1.0, 2.0, 3.0]))
        self.assertEqual(list(prepared["Deck"].astype(str)), ["A", "nan", "nan"])
        self.assertEqual(list(prepared["Deck"].cat.categories), ["A", "B", "nan"])


class TabMWrapperTest(unittest.TestCase):
    def test_numeric_nan_is_filled_with_training_median(self):
        est = tabm.build(dict(tabm.PARAMS), seed=0)
        est.categories = {"Deck": ["A", "nan"]}
        est.medians = {"Age": 30.0}
        prepared = est._prep(frame(["A", "Z"], [np.nan, 10.0]))
        self.assertEqual(list(prepared["Age"]), [30.0, 10.0])
        self.assertEqual(list(prepared["Deck"].astype(str)), ["A", "nan"])

    def test_defaults_are_toml_writable(self):
        # scripts/tune.py writes every PARAMS value into a TOML config, so no None / nested values
        for comp in (tabm, tabpfn):
            for key, value in comp.PARAMS.items():
                self.assertIsInstance(value, (int, float, str), f"{comp.NAME}.{key}")


if __name__ == "__main__":
    unittest.main()
