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
uv run python -m harness.score CONFIG --grid 0.1          # sweep blend weights over cached predictions (method = "mean" only)
uv run python -m harness.score CONFIG --smoke             # seconds-long check that scoring runs (CI)
uv run python -m harness.audit CONFIG [CONFIG ...]        # holdout reading, landing time only
uv run python -m harness.submit CONFIG                    # submissions/<config>.csv
uv run python -m scripts.tune MODEL                        # tuning idea: writes configs/exp/tune-<model>.toml
uv run python -m scripts.matrix IDEA [IDEA ...]            # every on/off combination of a batch's ideas
uv run python -m scripts.confirm CONFIG [--against CONFIG] # re-score on the confirmation seeds (200-202)
uv run python -m scripts.submit_leaderboard CONFIG -m MSG  # submit to Kaggle, print the public score
uv run python -m scripts.base_lb [SCORE -m MSG]             # read or record the pinned base's own LB
uv run python -m scripts.explain CONFIG [--against CONFIG] # what the models rely on (descriptive only)
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
a blend of cached models costs no training. `scripts/explain.py` writes its permutation
importances to `results/explain/<config>-<digest>.json` (committed), apart from the run
files: an explanation is descriptive, never a keep input.

**The holdout** is ~15% of training passengers, whole travel groups at a time (train and
test share no groups). `harness/data.py` drops it before anything else sees the data; only
`harness/audit.py` reads it. Its number is a smoke detector (one s.e. ≈ 0.01), never a keep
input: do not write it into `results/`, a config or a map.

### How strong the freeze is

- `hooks/pre-commit` refuses a commit that stages anything under `harness/`. It stops
  carelessness, nothing more: `git commit --no-verify` walks past it, and an agent in
  `auto` mode can run that. It is also only active once `scripts/install-hooks.sh` ran.
- The unbypassable layer is the CI gate (`.github/workflows/ci.yml`), which fails any PR
  into `main` (and any push to a `worktree-ticket-*` branch) that changes `harness/`, and
  separately smoke-scores every config against a synthetic fixture
  (`scripts/gen_ci_fixture.py`, no competition data or Kaggle credentials needed) to catch a
  scorer that no longer runs. It is not yet a *required* status check on `main`: this repo
  is private and GitHub's required-status-check protection needs a paid plan or a public
  repo, so nothing today stops a direct push or a `--no-verify` merge from landing red. See
  [issue #22](https://github.com/vltnkiz/kaggle-spaceship-titanic/issues/22) for the
  follow-up decision.
- **The one sanctioned way past both** is a *harness change*: its own issue and PR, never part
  of a batch. The user applies the `harness-change` label to the issue; the commit sets
  `HARNESS_CHANGE=<issue number>` and the hook checks that label with `gh`. The user applies
  the same label to the PR, which makes CI's freeze check skip itself. Nobody else applies it.
- Neither can see leakage inside `harness/data.py` or `harness/split.py`, because both
  sides of every comparison share them. Those two files are short on purpose: read them.

### The combiner

A config turns its models' probabilities into one with a `[combiner]` table
(`harness/combiner.py`). Absent, it is `mean`, the weighted blend.

```toml
[combiner]
method = "mean"        # mean | logit_mean | stack | gate
# gate only:
segment = "Deck"       # a column of the built feature matrix (errors if its feature is off)
min_rows = 200         # a segment with fewer training rows uses the global stack
```

`stack` and `gate` are learned: meta-fit per outer fold on the other folds' cached
out-of-fold predictions, so they cost seconds over the cache and retrain nothing. They read a
model's weight as on/off only. A combiner is a config-only idea (`extends = "main"` plus the
table), screened, matrixed and confirmed like any other; the methods exclude each other in the
matrix (`--exclude`). `scripts/tune.py` never tunes combiner parameters.

## Layout

```
harness/         frozen: data, split, fit, score, combiner, registry, config, audit, submit
                 (only data, split, fit, registry and config feed the cache fingerprint)
components/      add-only: one feature or model per file
configs/         main.toml (what main is), main.json (its pinned CV), main.lb.json (its pinned LB),
                 exp/*.toml (ideas), exp/_matrix/ (scratch matrix cells, gitignored)
results/         one JSON per run; oof/ is the prediction cache; explain/ the explanations
hooks/, scripts/ the freeze hook, its installer, the CI fixture generator, tune/matrix/confirm/
                 submit_leaderboard/base_lb/explain (batch tooling, built on the frozen harness)
.github/         the CI gate (freeze check + smoke run)
tests/           checks on the harness's guarantees
data/raw/        competition CSVs (gitignored; worktrees fall back to the main clone's)
notebooks/       EDA
src/             download_data
```
