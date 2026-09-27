import numpy as np
import pandas as pd

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PassengerId = gggg_pp -> travel group
    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)

    # Cabin = deck/num/side
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    # Passengers in cryosleep can't spend money
    df.loc[df["CryoSleep"] == True, SPEND_COLS] = df.loc[df["CryoSleep"] == True, SPEND_COLS].fillna(0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype("category")

    return df.drop(columns=["PassengerId", "Cabin", "Name"])


def build_features_spend_log(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    for c in SPEND_COLS + ["TotalSpend"]:
        df[f"Log{c}"] = np.log1p(df[c])
    return df


def build_features_spend_count(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    df["SpendCount"] = (df[SPEND_COLS] > 0).sum(axis=1)
    return df


def build_features_spend_luxury_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    df["LuxurySpend"] = df[["Spa", "VRDeck", "RoomService"]].sum(axis=1)
    df["BasicSpend"] = df[["FoodCourt", "ShoppingMall"]].sum(axis=1)
    return df


FEATURE_SETS = {
    "base": build_features,
    "spend_log": build_features_spend_log,
    "spend_count": build_features_spend_count,
    "spend_luxury_basic": build_features_spend_luxury_basic,
}
