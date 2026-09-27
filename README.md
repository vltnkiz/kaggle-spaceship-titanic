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
uv run python -m src.harness run lgbm                          # train+cache a model (experiments/models/lgbm.toml)
uv run python -m src.harness blend lgbm=0.7 rf=0.3              # score a manual-weight blend
uv run python -m src.harness blend lgbm=1 rf=1 --grid           # grid-search blend weights (0.1 steps)
uv run python -m src.harness submit experiments/main.toml       # write a submission from cached predictions
uv run python -m src.harness set-main experiments/main.toml     # pin experiments/main.json after a keep
uv run python -m src.harness log                                # print experiments/log.jsonl
uv run jupyter lab                                              # EDA in notebooks/
uv run kaggle competitions submit -c spaceship-titanic -f submissions/main.csv -m "main"
```

An experiment is a model TOML (`experiments/models/<id>.toml`: `features`, `kind`, `[params]`)
or a blend TOML (`experiments/blends/<id>.toml`: `[weights]`). OOF and test predictions are
cached by content hash under `data/processed/oof/`, so a blend never retrains. `experiments/main.toml`
+ pinned `experiments/main.json` define the pipeline on `main` and its CV, for "delta vs main".
See `CONTEXT.md` for vocabulary and the keep/drop rule.

## Layout

```
data/raw/          competition CSVs (gitignored)
data/processed/    OOF/test prediction cache (gitignored)
experiments/       model + blend TOML configs, main.json, log.jsonl
notebooks/         EDA / experiments
src/               config, features, model kinds, harness
models/            saved models
submissions/       submission CSVs (gitignored)
```
