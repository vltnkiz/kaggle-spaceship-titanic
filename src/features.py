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
