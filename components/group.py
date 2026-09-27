"""Travel group from PassengerId (gggg_pp): the group id, its size, and whether travelling alone."""
NAME = "group"
KIND = "feature"


def build(frame):
    frame["Group"] = frame["PassengerId"].str.split("_").str[0].astype(int)
    frame["GroupSize"] = frame.groupby("Group")["Group"].transform("size")
    frame["Solo"] = (frame["GroupSize"] == 1).astype(int)
    return frame
