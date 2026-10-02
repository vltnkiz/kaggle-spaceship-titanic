"""Lookup keys for the mates_cb model: the travel group and the surname, as plain strings.

MateGroup:   the gggg prefix of PassengerId (gggg_pp).
MateSurname: the last token of Name (missing when Name is missing).
Nothing is counted or learned here; these are identifiers that only mates_cb reads (it pops
them out of X before CatBoost sees anything).
"""
NAME = "mate_keys"
KIND = "feature"


def build(frame):
    frame["MateGroup"] = frame["PassengerId"].str.split("_").str[0]
    frame["MateSurname"] = frame["Name"].str.split().str[-1]
    return frame
