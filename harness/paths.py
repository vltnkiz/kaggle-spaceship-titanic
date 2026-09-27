"""Where things live. Worktrees share the main clone's raw data and OOF cache."""
import os
import subprocess
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
PINNED = CONFIGS / "main.json"


@cache
def shared_root() -> Path:
    """The main clone, even when running inside a worktree of it."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return ROOT


def raw_dir() -> Path:
    """$SPACESHIP_DATA, else this checkout's data/raw, else the main clone's."""
    if env := os.environ.get("SPACESHIP_DATA"):
        return Path(env)
    local = ROOT / "data" / "raw"
    return local if (local / "train.csv").exists() else shared_root() / "data" / "raw"


def oof_dir() -> Path:
    """Content-addressed, so every worktree can safely share one cache."""
    return shared_root() / "results" / "oof"
