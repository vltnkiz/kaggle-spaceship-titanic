"""Spend reshaping: log1p of each bill, count of billed categories, luxury vs basic totals."""
import numpy as np

NAME = "spend_logs"
KIND = "feature"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY_COLS = ["RoomService", "Spa", "VRDeck"]
BASIC_COLS = ["FoodCourt", "ShoppingMall"]


def build(frame):
    for col in SPEND_COLS:
        frame[f"{col}_log"] = np.log1p(frame[col])
    frame["SpendCount"] = (frame[SPEND_COLS] > 0).sum(axis=1)
    frame["LuxurySpend"] = frame[LUXURY_COLS].sum(axis=1)
    frame["BasicSpend"] = frame[BASIC_COLS].sum(axis=1)
    return frame
