"""Consistency imputation: fill NaNs in place from travel-group mates, surname-mates and hard rules.

Fixed rule, chosen up front: a NaN is filled from mates only when every known value among
them agrees (unanimous); disagreeing or all-missing mates leave it NaN. No global fallbacks,
no was-missing flags, no target. Group and surname are read from PassengerId / Name directly.

1. HomePlanet, Deck, Side, VIP  <- unanimous value among travel-group mates (gggg prefix).
2. HomePlanet                   <- unanimous value among surname-mates (last token of Name).
3. CryoSleep                    <- False when any bill is positive.
4. Bills                        <- 0 when CryoSleep is True.
TotalSpend / NoSpend are recomputed as in `spend` when present.
"""
NAME = "impute"
KIND = "feature"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
GROUP_COLS = ["HomePlanet", "Deck", "Side", "VIP"]


def _fill_unanimous(frame, col, key):
    """Fill NaNs in `col` with the single known value of their `key` mates, if exactly one."""
    known = frame[col].notna() & key.notna()
    values = frame.loc[known, col].groupby(key[known])
    single = values.first()[values.nunique() == 1]
    missing = frame[col].isna() & key.isin(single.index)
    frame.loc[missing, col] = key[missing].map(single)


def build(frame):
    group = frame["PassengerId"].str.split("_").str[0]
    surname = frame["Name"].str.split().str[-1]
    for col in GROUP_COLS:
        if col in frame.columns:
            _fill_unanimous(frame, col, group)
    _fill_unanimous(frame, "HomePlanet", surname)

    spent = (frame[SPEND_COLS] > 0).any(axis=1)
    frame.loc[spent & frame["CryoSleep"].isna(), "CryoSleep"] = False
    asleep = frame["CryoSleep"] == True  # noqa: E712 - object column of True/False/NaN
    frame.loc[asleep, SPEND_COLS] = frame.loc[asleep, SPEND_COLS].fillna(0)

    if "TotalSpend" in frame.columns:
        frame["TotalSpend"] = frame[SPEND_COLS].sum(axis=1)
    if "NoSpend" in frame.columns:
        frame["NoSpend"] = (frame[SPEND_COLS].sum(axis=1) == 0).astype(int)
    return frame
