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


def build_features_group_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Like build_features, but fills HomePlanet/Deck/Side from travel-group
    mates before the missing-marker categorical pass (group's own mode; group
    has no known value -> left NaN, same as build_features)."""
    df = df.copy()

    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    for c in ["HomePlanet", "Deck", "Side"]:
        group_mode = df.groupby("Group")[c].transform(
            lambda s: s.mode().iat[0] if s.notna().any() else pd.NA
        )
        df[c] = df[c].fillna(group_mode)

    df.loc[df["CryoSleep"] == True, SPEND_COLS] = df.loc[df["CryoSleep"] == True, SPEND_COLS].fillna(0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype("category")

    return df.drop(columns=["PassengerId", "Cabin", "Name"])


def build_features_group_impute_hs(df: pd.DataFrame) -> pd.DataFrame:
    """Like build_features_group_impute, but only HomePlanet/Side (both 100%
    within-group agreement in train+test); Deck agrees only ~69% of the time
    and is left out here to isolate its effect."""
    df = df.copy()

    df["Group"] = df["PassengerId"].str.split("_").str[0].astype(int)
    df["GroupSize"] = df.groupby("Group")["Group"].transform("size")
    df["Solo"] = (df["GroupSize"] == 1).astype(int)

    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    for c in ["HomePlanet", "Side"]:
        group_mode = df.groupby("Group")[c].transform(
            lambda s: s.mode().iat[0] if s.notna().any() else pd.NA
        )
        df[c] = df[c].fillna(group_mode)

    df.loc[df["CryoSleep"] == True, SPEND_COLS] = df.loc[df["CryoSleep"] == True, SPEND_COLS].fillna(0)
    df["TotalSpend"] = df[SPEND_COLS].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    for c in CAT_COLS:
        df[c] = df[c].astype(str).astype("category")

    return df.drop(columns=["PassengerId", "Cabin", "Name"])


FEATURE_SETS = {
    "base": build_features,
    "group_impute": build_features_group_impute,
    "group_impute_hs": build_features_group_impute_hs,
}
