"""Download and unzip competition data into data/raw.

Requires Kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
and that you've accepted the competition rules on the website.
"""
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import COMPETITION, RAW


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION, path=RAW, quiet=False)
    zip_path = RAW / f"{COMPETITION}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    zip_path.unlink()
    print("Files:", sorted(p.name for p in RAW.iterdir()))


if __name__ == "__main__":
    main()
