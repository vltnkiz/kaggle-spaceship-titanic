"""The holdout carve and the CV folds. Both are fixed: changing either invalidates every score."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HOLDOUT_SALT = "spaceship-holdout-2026-09-27"
HOLDOUT_PERMILLE = 150  # ~15% of travel groups

SEEDS = (0, 1, 2)
N_FOLDS = 5


def in_holdout(passenger_ids: pd.Series) -> np.ndarray:
    """True for passengers in the holdout.

    Whole travel groups go in or out together, because no group spans train and test:
    the holdout then resembles the test set, and group-level features (group size) come
    out the same whether or not the holdout rows are present.
    Membership depends only on the group id, never on row order or the target.
    """
    groups = passenger_ids.str.split("_").str[0]
    return np.array([_bucket(g) < HOLDOUT_PERMILLE for g in groups])


def _bucket(group: str) -> int:
    digest = hashlib.sha256(f"{HOLDOUT_SALT}:{group}".encode()).hexdigest()
    return int(digest, 16) % 1000


def folds(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, validation) index pairs over whatever rows it is given, i.e. the dev rows."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def fold_of(y: np.ndarray, seed: int) -> np.ndarray:
    """For each row, the index of the fold that validates it."""
    out = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(folds(y, seed)):
        out[va] = k
    return out
