"""Load the competition data. No fitting, no statistics, nothing learned from any row.

The only transformation here is `to_matrix`, which drops identifier columns and marks
text columns as categorical. Everything else a model sees comes from components.
"""
import hashlib

import numpy as np
import pandas as pd

from harness import paths, split

TARGET = "Transported"
ID = "PassengerId"
IDENTIFIERS = ["PassengerId", "Name", "Cabin"]  # read by components, never by a model


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.raw_dir() / name)


def load_dev() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(dev rows without the target, dev target, test rows). Holdout rows are gone before this returns."""
    train = read("train.csv")
    train = train[~split.in_holdout(train[ID])].reset_index(drop=True)
    y = train.pop(TARGET).astype(int).to_numpy()
    return train, y, read("test.csv")


def to_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers; every non-numeric column becomes a category."""
    frame = frame.drop(columns=[c for c in IDENTIFIERS if c in frame.columns])
    for c in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[c]):
            frame[c] = frame[c].astype(str).astype("category")
    return frame


def fingerprint() -> str:
    """Hash of the raw files, so a cached prediction never outlives the data it came from."""
    h = hashlib.sha256()
    for name in ("train.csv", "test.csv"):
        h.update((paths.raw_dir() / name).read_bytes())
    return h.hexdigest()[:16]
