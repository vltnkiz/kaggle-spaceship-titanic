"""Cabin (deck/num/side) split into its three parts."""
import pandas as pd

NAME = "cabin"
KIND = "feature"


def build(frame):
    cabin = frame["Cabin"].str.split("/", expand=True)
    frame["Deck"] = cabin[0]
    frame["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    frame["Side"] = cabin[2]
    return frame
