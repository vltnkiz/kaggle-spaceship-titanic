"""Spend: cryosleepers' missing bills are zero; total spend and a no-spend flag."""
NAME = "spend"
KIND = "feature"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build(frame):
    asleep = frame["CryoSleep"] == True  # noqa: E712 - object column of True/False/NaN
    frame.loc[asleep, SPEND_COLS] = frame.loc[asleep, SPEND_COLS].fillna(0)
    frame["TotalSpend"] = frame[SPEND_COLS].sum(axis=1)
    frame["NoSpend"] = (frame["TotalSpend"] == 0).astype(int)
    return frame
