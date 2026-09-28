"""Submit a config's predictions to the Kaggle leaderboard and report the score.

    python -m scripts.submit_leaderboard CONFIG [-m MESSAGE]
    python -m scripts.submit_leaderboard --quota

Called once per batch, in phase 5, on the batch's final config -- never on a drop, and never
per idea (see `.claude/skills/experiment-batch/SKILL.md`'s Phase 5 and `CONTEXT.md`'s
**Leaderboard gate**: the gate now drops a bad keep *and* can promote a good near-miss, so it
is a comparison against the pinned base's own LB, not a CV-vs-LB veto). This script only
prints the score; the planner applies the gate and records the outcome. It never writes
anything to `results/`, a config, or anywhere a planner reads -- `scripts/base_lb.py` is
where a base's LB gets recorded once a batch's landing promotes it.

`--quota` prints how many of the 10/day submissions remain, without submitting, so a planner
can check headroom before calling this. Running out stops the submit, not the batch: an
un-submitted keep lands anyway, flagged unverified; an un-submitted near-miss waits.
"""
import argparse
import csv
import io
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from harness import paths

COMPETITION = "spaceship-titanic"
DAILY_LIMIT = 10


def remaining_quota() -> int:
    """How many of today's 10 submissions are unused, per the CLI's own submissions list."""
    out = subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
        check=True, capture_output=True, text=True, cwd=paths.ROOT,
    ).stdout
    today = datetime.now(timezone.utc).date()
    used = 0
    for row in csv.DictReader(io.StringIO(out)):
        try:
            submitted = datetime.strptime(row.get("date", ""), "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            continue
        if submitted == today:
            used += 1
    return DAILY_LIMIT - used


def submit(config: str, message: str) -> Path:
    subprocess.run([sys.executable, "-m", "harness.submit", config], check=True, cwd=paths.ROOT)
    csv_path = paths.ROOT / "submissions" / f"{Path(config).stem}.csv"
    subprocess.run(
        ["uv", "run", "kaggle", "competitions", "submit", "-c", COMPETITION,
         "-f", str(csv_path), "-m", message],
        check=True, cwd=paths.ROOT,
    )
    return csv_path


def latest_score(file_name: str, timeout: float = 600.0, interval: float = 15.0) -> dict:
    """Poll the submissions list for `file_name`'s newest entry until it has scored."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            ["uv", "run", "kaggle", "competitions", "submissions", "-c", COMPETITION, "--csv"],
            check=True, capture_output=True, text=True, cwd=paths.ROOT,
        ).stdout
        rows = [r for r in csv.DictReader(io.StringIO(out)) if r.get("fileName") == file_name]
        if rows and rows[0].get("publicScore"):
            return rows[0]
        time.sleep(interval)
    raise TimeoutError(f"{file_name} did not score within {timeout:.0f}s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.submit_leaderboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?")
    ap.add_argument("-m", "--message", default=None)
    ap.add_argument("--quota", action="store_true", help="print remaining submissions today and exit")
    args = ap.parse_args(argv)

    if args.quota:
        print(f"{remaining_quota()} submissions remaining today")
        return
    if not args.config:
        ap.error("config is required unless --quota is given")

    remaining = remaining_quota()
    if remaining <= 0:
        print(f"0 of {DAILY_LIMIT} submissions remaining today; skipping submit.", file=sys.stderr)
        sys.exit(1)
    print(f"{remaining} of {DAILY_LIMIT} submissions remaining today (before this one)")

    message = args.message or Path(args.config).stem
    csv_path = submit(args.config, message)
    print(f"Uploaded {csv_path.relative_to(paths.ROOT).as_posix()}")
    row = latest_score(csv_path.name)
    print(f"Public score: {row['publicScore']}  (status={row['status']}, file={row['fileName']})")


if __name__ == "__main__":
    main()
