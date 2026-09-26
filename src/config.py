from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"

COMPETITION = "spaceship-titanic"
TARGET = "Transported"
ID = "PassengerId"
SEED = 42
