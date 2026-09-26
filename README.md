# Spaceship Titanic

https://www.kaggle.com/competitions/spaceship-titanic — binary classification (`Transported`), metric: accuracy.

## Setup

1. Accept the competition rules on the Kaggle website (Join Competition).
2. Create an API token (kaggle.com → Settings → API → Create New Token) and save the
   `KGAT_...` string to `C:\Users\<you>\.kaggle\access_token` (or set `KAGGLE_API_TOKEN`).
3. Download data:
   ```
   uv run python -m src.download_data
   ```

## Usage

```
uv run python -m src.train                  # CV + submissions/lgbm_baseline.csv
uv run jupyter lab                          # EDA in notebooks/
uv run kaggle competitions submit -c spaceship-titanic -f submissions/lgbm_baseline.csv -m "baseline"
```

## Layout

```
data/raw/        competition CSVs (gitignored)
data/processed/  intermediate features
notebooks/       EDA / experiments
src/             config, features, training
models/          saved models
submissions/     submission CSVs
```
