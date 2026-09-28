"""The pinned base's own public leaderboard score -- what every batch's leaderboard gate
compares its final config against (see `CONTEXT.md`'s **Leaderboard gate**). Kept separate
from `configs/main.json` (the CV pin `harness/score.py` writes) because writing it is not
the scorer's job and `harness/` stays frozen.

    python -m scripts.base_lb                      print the recorded base LB, or that none is recorded
    python -m scripts.base_lb SCORE -m MESSAGE      record a new base LB (landing, after a promote)

A batch's leaderboard gate never calls this script to *submit*: the base's LB score is
already known before the gate runs -- either from here, or, when a batch's own final config
is what gets promoted to the new base, from the very gate submission that just scored it, so
recording the new base's LB costs no extra quota. The one exception is a base that has never
been submitted at all (a fresh pin with no LB yet); that is a one-off, quota-permitting,
`scripts.submit_leaderboard` call followed by recording its score here -- not this script's
job either.
"""
import argparse
import json
from datetime import datetime, timezone

from harness import paths

PATH = paths.CONFIGS / "main.lb.json"


def read() -> dict | None:
    return json.loads(PATH.read_text()) if PATH.exists() else None


def stale() -> bool:
    """True if nothing is recorded, or what's recorded is for a commit that isn't the
    current pin (`configs/main.json` moved since the LB was last recorded)."""
    current = read()
    if current is None:
        return True
    pinned = json.loads(paths.PINNED.read_text())
    return current["commit"] != pinned["commit"]


def write(public_score: float, message: str) -> dict:
    pinned = json.loads(paths.PINNED.read_text())
    record = {
        "public_score": public_score,
        "commit": pinned["commit"],
        "cv": pinned["cv"],
        "message": message,
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PATH.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.base_lb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("score", nargs="?", type=float, help="the base's public LB score to record")
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args(argv)

    if args.score is None:
        current = read()
        if current is None:
            print("no base LB recorded")
            return
        flag = " (STALE: commit has moved since this was recorded)" if stale() else ""
        print(json.dumps(current, indent=2) + flag)
        return

    if not args.message:
        ap.error("-m MESSAGE is required when recording a score")
    rec = write(args.score, args.message)
    print(f"Recorded base LB {rec['public_score']} for commit {rec['commit'][:12]} "
          f"({PATH.relative_to(paths.ROOT).as_posix()})")


if __name__ == "__main__":
    main()
