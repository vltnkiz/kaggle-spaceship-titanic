"""Synthetic train/test CSVs shaped like the competition data, for the CI smoke run only.

No real competition data is committed to the repo (data/ is gitignored, and Kaggle's terms
do not clearly allow redistributing it). This generates a small, deterministic fixture with
the same columns and dtypes, sized just large enough for one stratified fold: score.py's
--smoke mode never checks accuracy, only that the pipeline runs end to end.

    python -m scripts.gen_ci_fixture DEST_DIR [--rows N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOME_PLANETS = ["Earth", "Europa", "Mars"]
DESTINATIONS = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]
DECKS = list("ABCDEFG")
SIDES = ["P", "S"]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def make_frame(n: int, rng: np.random.Generator, id_offset: int, with_target: bool) -> pd.DataFrame:
    # Group passengers 1-3 to a group, like real travel groups, so `group`/holdout carving
    # has something to key on.
    group_sizes = []
    remaining = n
    while remaining > 0:
        size = min(remaining, int(rng.integers(1, 4)))
        group_sizes.append(size)
        remaining -= size

    passenger_id, group_col = [], []
    for i, size in enumerate(group_sizes):
        gid = id_offset + i
        for p in range(1, size + 1):
            passenger_id.append(f"{gid:04d}_{p:02d}")
            group_col.append(gid)

    n = len(passenger_id)
    cryo = rng.choice([True, False], size=n)
    spend = rng.integers(0, 500, size=(n, len(SPEND_COLS))).astype(float)
    spend[cryo] = 0.0
    # A little real-world messiness: some missing values, like the actual data has.
    for col in range(len(SPEND_COLS)):
        missing = rng.random(n) < 0.02
        spend[missing, col] = np.nan

    frame = pd.DataFrame({
        "PassengerId": passenger_id,
        "HomePlanet": rng.choice(HOME_PLANETS, size=n),
        "CryoSleep": cryo,
        "Cabin": [f"{rng.choice(DECKS)}/{rng.integers(0, 2000)}/{rng.choice(SIDES)}" for _ in range(n)],
        "Destination": rng.choice(DESTINATIONS, size=n),
        "Age": rng.integers(0, 80, size=n).astype(float),
        "VIP": rng.choice([True, False], size=n, p=[0.05, 0.95]),
        **{col: spend[:, i] for i, col in enumerate(SPEND_COLS)},
        "Name": [f"Person{i}" for i in range(n)],
    })
    if with_target:
        # Some real signal so both classes are well represented, needed for stratified folds.
        score = frame["CryoSleep"].astype(int) + rng.normal(0, 1, size=n)
        frame["Transported"] = score > np.median(score)
    return frame


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--rows", type=int, default=400, help="dev+holdout train rows")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    args.dest.mkdir(parents=True, exist_ok=True)
    train = make_frame(args.rows, rng, id_offset=0, with_target=True)
    test = make_frame(max(args.rows // 4, 50), rng, id_offset=100_000, with_target=False)
    train.to_csv(args.dest / "train.csv", index=False)
    test.to_csv(args.dest / "test.csv", index=False)
    print(f"wrote {len(train)} train / {len(test)} test rows to {args.dest}")


if __name__ == "__main__":
    main()
