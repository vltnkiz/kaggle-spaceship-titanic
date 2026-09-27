"""Submit a config's predictions to the Kaggle leaderboard and report the score.

    python -m scripts.submit_leaderboard CONFIG [-m MESSAGE]

Writes the submission CSV via harness.submit, uploads it with the kaggle CLI, then polls
`kaggle competitions submissions` until the new entry has scored and prints its public
score.

The leaderboard is a milestone fact, not a keep input (see the map's Notes): this script
only prints the score. It never writes it to results/, a config, or anywhere a planner
reads.
"""
import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path

from harness import paths

COMPETITION = "spaceship-titanic"


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
    ap.add_argument("config")
    ap.add_argument("-m", "--message", default=None)
    args = ap.parse_args(argv)
    message = args.message or f"wayfinder ticket-35: {Path(args.config).stem}"

    csv_path = submit(args.config, message)
    print(f"Uploaded {csv_path.relative_to(paths.ROOT).as_posix()}")
    row = latest_score(csv_path.name)
    print(f"Public score: {row['publicScore']}  (status={row['status']}, file={row['fileName']})")


if __name__ == "__main__":
    main()
