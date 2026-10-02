"""Drop the raw Group integer id (GroupSize and Solo stay); a no-op if Group is absent."""
NAME = "drop_group_id"
KIND = "feature"


def build(frame):
    return frame.drop(columns=["Group"], errors="ignore")
