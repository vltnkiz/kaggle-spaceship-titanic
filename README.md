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
4. Install the harness freeze hook (once per clone; worktrees share it):
   ```
   sh scripts/install-hooks.sh
   ```

## Experiments

Every score comes from the frozen harness. Research only **adds** files to `components/`
and `configs/`; it never edits `harness/`.

```
uv run python -m harness.score configs/main.toml          # score what main is
uv run python -m harness.score configs/exp/age_bins.toml  # score an idea; prints delta vs the pinned base
uv run python -m harness.score configs/main.toml --pin    # re-pin the base (part of landing a keep)
uv run python -m harness.score CONFIG --grid 0.1          # sweep blend weights over cached predictions
uv run python -m harness.score CONFIG --smoke             # seconds-long check that scoring runs (CI)
uv run python -m harness.audit CONFIG [CONFIG ...]        # holdout reading, landing time only
uv run python -m harness.submit CONFIG                    # submissions/<config>.csv
uv run python -m unittest discover -s tests -t .
```

**CV score**: accuracy, 5-fold stratified CV × seeds 0/1/2, over the dev rows (training
rows outside the holdout). Every config is scored on the same folds, so the delta against
`configs/main.json` is paired. Keep an idea if it improves the pinned base by ≥ +0.002.

**Adding an idea** is one file and one line:

```python
# components/my_idea.py
NAME = "my_idea"      # must equal the file name
KIND = "feature"      # or "model"

def build(frame):     # model: build(params, seed) -> estimator; optional PARAMS = {...}
    frame["NewCol"] = ...
    return frame
```

```toml
# configs/exp/my_idea.toml
extends = "main"
[features]
my_idea = true
```

A feature component receives dev and test rows with no target column, and must not learn
from labels. A model component's `fit` only ever receives training-fold rows; any early
stopping must split those, never the scored fold. `harness/registry.py` has the contract.

**Outputs.** Each run writes its own `results/<run-id>.json` (committed; never a shared
log). Out-of-fold and test predictions are cached by content hash in `results/oof/`
(ignored), in the main clone even when running from a worktree, so worktrees share it and
a blend of cached models costs no training.

**The holdout** is ~15% of training passengers, whole travel groups at a time (train and
test share no groups). `harness/data.py` drops it before anything else sees the data; only
`harness/audit.py` reads it. Its number is a smoke detector (one s.e. ≈ 0.01), never a keep
input: do not write it into `results/`, a config or a map.

### How strong the freeze is

- `hooks/pre-commit` refuses a commit that stages anything under `harness/`. It stops
  carelessness, nothing more: `git commit --no-verify` walks past it, and an agent in
  `auto` mode can run that. It is also only active once `scripts/install-hooks.sh` ran.
- The unbypassable layer is the CI gate, which fails any PR into `main` that changes
  `harness/`.
- Neither can see leakage inside `harness/data.py` or `harness/split.py`, because both
  sides of every comparison share them. Those two files are short on purpose: read them.

## Layout

```
harness/         frozen: data, split, score, registry, config, audit, submit
components/      add-only: one feature or model per file
configs/         main.toml (what main is), main.json (its pinned score), exp/*.toml (ideas)
results/         one JSON per run; oof/ is the prediction cache
hooks/, scripts/ the freeze hook and its installer
tests/           checks on the harness's guarantees
data/raw/        competition CSVs (gitignored; worktrees fall back to the main clone's)
notebooks/       EDA
src/             download_data
```
