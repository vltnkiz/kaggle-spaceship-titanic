"""Cabin-number region bands (fixed width 300) and deck+side as one categorical.

Runs after `cabin`, which supplies Deck, CabinNum and Side.
"""
import numpy as np

NAME = "cabin_bins"
KIND = "feature"

REGION_WIDTH = 300  # fixed up front, not tuned


def build(frame):
    frame["CabinRegion"] = np.floor(frame["CabinNum"] / REGION_WIDTH)
    missing = frame["Deck"].isna() | frame["Side"].isna()
    deck_side = frame["Deck"].astype(str) + "/" + frame["Side"].astype(str)
    frame["DeckSide"] = deck_side.where(~missing)
    return frame
