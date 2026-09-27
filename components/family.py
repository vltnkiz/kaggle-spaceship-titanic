"""Surname family features from Name, counted over the frame (dev + test rows, no target).

FamilySize:   rows in the frame sharing the surname (0 when Name is missing).
FamilyGroups: distinct travel groups (PassengerId prefix) the surname appears in (0 when missing).
The surname string itself is not emitted (it would be a huge-cardinality category).
"""
NAME = "family"
KIND = "feature"


def build(frame):
    surname = frame["Name"].str.split().str[-1]
    group = frame["PassengerId"].str.split("_").str[0]
    known = surname.notna()
    size = surname[known].map(surname[known].value_counts())
    groups = surname[known].map(group[known].groupby(surname[known]).nunique())
    frame["FamilySize"] = size.reindex(frame.index).fillna(0).astype(int)
    frame["FamilyGroups"] = groups.reindex(frame.index).fillna(0).astype(int)
    return frame
