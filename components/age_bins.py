"""Example component: age bands. Copy this file's shape for a new idea.

Off unless a config switches it on; configs/exp/age_bins.toml does, in one line.
"""
import pandas as pd

NAME = "age_bins"
KIND = "feature"


def build(frame):
    bands = pd.cut(frame["Age"], bins=[-1, 12, 17, 25, 40, 200],
                   labels=["child", "teen", "young", "adult", "older"])
    frame["AgeBand"] = bands.astype(str)
    return frame
