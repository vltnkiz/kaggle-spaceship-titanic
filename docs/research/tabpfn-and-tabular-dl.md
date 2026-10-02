# TabPFN and one modern tabular DL model as harness components

Research for issue #76 (map #59). Written 2026-10-02. Nothing here is implemented; no repo file other than this one was touched.

Labels used throughout: **[measured]** = I ran it on this machine (RTX 3080 Ti 12 GB, 32 logical CPUs, Windows 11, driver 591.86 / CUDA 13.1) in a throwaway venv under `$CLAUDE_JOB_DIR/tmp`; **[doc]** = stated by a cited primary source; **[estimate]** = my extrapolation.

**Measurement caveat.** The benchmark data was a crude stand-in for the harness matrix: the first 85% of `train.csv` rows (7,389 rows), 16 columns (raw columns, Cabin split to Deck/Num/Side, Group, GroupSize, total spend), one seed, 5 stratified folds (5,911 training rows per fold, in line with the real ~5.8k). Categorical columns were pandas `category` as `harness.data.to_matrix` produces them. Accuracy figures below are therefore only a sanity check, not CV scores comparable with 0.81269. Timings are what matter.

---

## 1. TabPFN

### Version: use TabPFN-3.5 (package default); TabPFN-2 is the ungated fallback

- The `tabpfn` package on PyPI is at 9.1.0 (9.0.0 released 2026-06-11); it needs Python >= 3.10 and `torch>=2.5` [PyPI JSON, https://pypi.org/pypi/tabpfn/json].
- The README says "To use our default TabPFN-3.5 model: `TabPFNClassifier()`", with `TabPFN-3.5-Fast` selectable via `ModelVersion.V3_5_FAST`, and TabPFN-3, 2.6, 2.5 and 2 as older checkpoints [https://github.com/PriorLabs/TabPFN, README].
- Model table [https://docs.priorlabs.ai/models]: TabPFN-3.5 (Sep 2026, up to 1,000,000 rows / 20,000 features, "balanced"); 3.5-Fast ("up to 6x faster"); 3.5-Thinking (200k rows / 2,000 features, "most accurate", more inference compute); TabPFN-3 (May 2026); 2.6 (Apr 2026); 2.5 (Nov 2025); TabPFNv2 (2025, 10,000 rows / 500 features / 10 classes).
- The 3.5 technical report (arXiv 2609.17895) claims it "significantly outperforms its predecessor, TabPFN-3, and all existing baselines"; Fast is "up to 3x faster" than TabPFN-3 [https://arxiv.org/abs/2609.17895]. That is the vendor's claim. TabPFN-2.5's report claims it leads TabArena and matches AutoGluon 1.4 [https://arxiv.org/abs/2511.08667]. I could not verify either independently.
- So "v2 vs 2.5" in the issue is stale: v2.5 and v2.6 are two generations back. Recommendation: **TabPFN-3.5** (default). Keep `v3.5-fast` (334 MB vs 876 MB checkpoint) as the time-budget fallback and TabPFN-2 as the zero-friction fallback (below).

### Licence and gated weights: a HITL task IS needed for 3.5

- Code: Apache-2.0. Weights are licensed separately; TabPFN-2.5/2.6/3/3.5 weights are non-commercial licences, TabPFN-2 weights are the "Prior Labs License" (Apache-2.0 plus an attribution requirement) [README, License section].
- TabPFN-3.5 licence text (https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE, v1.0, 2026-09-09): free for "Non-Commercial Purpose", defined to include "internal benchmarking, academic research ... as well as Data Science Competitions" on "established platforms (such as Kaggle ...)", provided results are not used in commercial decisions or paid products. License is personal and non-transferable; no hosting as a service; Outputs may only be used for non-commercial purposes; you may not use Outputs to train a model competitive with TabPFN-3.5. Kaggle use is therefore explicitly allowed. Redistribution of weights needs the licence and an attribution notice (do not commit weights to the repo).
- The HF repo itself reports `gated: false` (https://huggingface.co/api/models/Prior-Labs/tabpfn_3_5), but the **library enforces its own licence-acceptance gate**: `model_loading._download_model` calls `ensure_license_accepted` for V2.5/V2.6/V3/V3.5 before downloading [https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model_loading.py]. README FAQ: first use opens a browser to log in at https://ux.priorlabs.ai and accept the licence, token cached locally; for "headless / CI environments" accept the licence on the **License** tab and set `TABPFN_TOKEN`.
- **[measured]** with `TABPFN_NO_BROWSER=1` and no token, `TabPFNClassifier.create_default_for_version(V3_5)` raised `TabPFNLicenseError: TabPFN requires a one-time license acceptance ... Set the TABPFN_TOKEN environment variable with a valid API key obtained from https://ux.priorlabs.ai`. So I could not run 3.5 itself; its timings below are estimates.
- **TabPFN-2 needs no login**: weights `Prior-Labs/TabPFN-v2-clf` are ungated on HF and downloaded without a token **[measured]**; its limits are 10,000 rows / 500 features [models page], enough for a 5.9k-row fold.

**HITL setup task (needed before the 3.5 batch):**
1. The user creates/logs into a PriorLabs account at https://ux.priorlabs.ai, accepts the TabPFN-3.5 licence on the License tab, and copies the API token.
2. Locally: either let the first `fit` open the browser once (token cached), or set `TABPFN_TOKEN` in the user env.
3. CI: add repo secret `TABPFN_TOKEN` and pass it to the smoke job's env (`TABPFN_NO_BROWSER=1` too). A fork PR would not receive it, which does not matter for a single-owner repo, but note that agents' worktree branches pushed to `origin` do get repo secrets only for pushes/PRs from the same repo (the CI triggers are `pull_request` into main and pushes to `worktree-ticket-*`, both same-repo).
4. Cache weights in CI (`TABPFN_MODEL_CACHE_DIR` + `actions/cache`): the 3.5 checkpoint is 876 MB (`tabpfn-v3.5-20260909.safetensors`, 876,027,932 bytes; Fast is 334 MB) [HF API with blobs=true]. Downloading 876 MB on every smoke run is avoidable.
If the user will not do this, the fallback is TabPFN-2 (`ModelVersion.V2`), which needs nothing.

### Limits against our data

| | Limit | Our fold |
|---|---|---|
| Rows (GPU), 3.5 | 1,000,000 [doc] | ~5.9k train, ~1.5k val, ~4.3k test per fold |
| Features, 3.5 | 20,000 [doc] | ~20-30 |
| Rows (CPU) | 5,000 for 3/3.5/3.5-Fast, 1,000 for older [README] | 5.9k > 5,000: needs `TABPFN_ALLOW_CPU_LARGE_DATASET=true` |
| Classes | unlimited (3.5), 10 (v2) | 2 |

README on GPU: "even older ones with ~8GB VRAM work well; 16GB needed for some large datasets".

### Categoricals and NaNs

- NaNs: handled natively; README FAQ "Can TabPFN handle missing values? Yes!".
- Categoricals: README says do **not** scale or one-hot encode; the classifier infers categorical columns, and `categorical_features_indices` is a hint; "A column with pandas' `category` dtype counts as listed here" [classifier.py docstring, https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/classifier.py]. This matches what `to_matrix` produces, so the harness frame can go straight in. **[measured]** v2 fit/predict on the `category`-dtype frame with NaNs worked, no preprocessing.

### Ensembling knobs and seeding

From the classifier signature/docstring (classifier.py): `n_estimators` (default `"auto"` = checkpoint default, raised on wide data; each member is a differently-prepared "prompt" of the training set; averaged), `softmax_temperature` ("auto" = checkpoint), `balance_probabilities`, `average_before_softmax`, `inference_precision`, `fit_mode`, `memory_saving_mode`, `ignore_pretraining_limits`, `device` ("auto"), and `random_state` (**default 0, not None**: "we depart from the usual scikit-learn behavior in that by default we provide a fixed seed of 0"). So the component must pass the harness `seed` as `random_state` or every seed would give the same ensemble.

Determinism: the docstring says reproducibility is not always guaranteed (non-deterministic torch ops, numerical instability; higher precision helps across hardware). **[measured]** two fits with `random_state=0` on the 3080 Ti (v2): max |dp| = 0.0, 0 label flips. Same-machine reproducibility held; cross-hardware (local GPU vs CI CPU) is not promised, and predictions are cached by the harness anyway.

### Time per fold (v3.5 = estimate, v2 = measured)

[measured] TabPFN-2, GPU, default `n_estimators`, 5,911 train rows:
- fit 0.3 s after the first (weights are cached in-process; first fit in a fresh process 0.5-9.7 s incl. download).
- predict 1.5k val rows: ~7 s. Predict ~4.3k test rows in chunks of 1,000: 23 s for 4,278 rows, peak GPU memory 6.9 GB.
- `n_estimators=1`: predict 1.6 s on 1.5k rows (vs 6.7-7.3 s at default).
- **Predicting the 4.3k test rows in a single call on the 12 GB card did not finish in 18 min** (GPU memory pegged at 11.9 GB, 100% util; I stopped it). README: "If the test set is very large, split it into chunks of 1000 samples each." The harness calls `predict_proba(X_test)` with the full test set every fold (`harness/score.py:fit_predict`), so the component's `predict_proba` must chunk (<= 1,000 rows) itself.
- Per fold ~30 s (val + test), per config run (5 folds x 3 seeds) ~7.5 min for v2 on this GPU. Accuracy, one seed on the crude features, v2: 0.8315/0.8031/0.7896/0.8329/0.8192, mean 0.8153.
- [estimate] 3.5 default: same order as v2 to ~3x slower (the model is larger, 876 MB; "balanced" per the models page; Fast is claimed "up to 6x faster"). Plan for ~0.5-1.5 min/fold, ~10-25 min per 15-fold run; measure on first use. Use Fast if too slow.

CPU [measured, v2, 32 threads, 1 fold, `TABPFN_ALLOW_CPU_LARGE_DATASET=true`]: fit 0.3 s, predict 1.5k val rows 105.5 s. A full run (15 folds, plus the test rows) would be ~hours; on a 4-vCPU CI runner worse still. **CPU is viable only for smoke-size fixtures** (v2, 1.6k train / 0.4k val rows: 21 s on this machine **[measured]**; CI runner [estimate] 1-2 min).

---

## 2. TabM vs RealMLP: pick TabM, via pytabkit

### Packages

- `pytabkit` 1.7.3 (2026-01-06), Apache-2.0; scikit-learn-compatible `RealMLP_TD_Classifier`, `TabM_D_Classifier`, etc.; TabM and RealMLP work **without** the `[models]` extra; core requires numpy, pandas, psutil, pytorch-lightning, scikit-learn, torch, torchmetrics [PyPI JSON https://pypi.org/pypi/pytabkit/json; README https://github.com/dholzmueller/pytabkit]. README: "Please install torch separately if you want to control the version (CPU/GPU etc.)".
- `tabm` 0.0.3 (PyPI; deps torch, `rtdl_num_embeddings`, Apache-2.0) is the authors' (Yandex) building-block package: `TabM.make(n_num_features=..., cat_cardinalities=..., d_out=...)` returns a torch module with output `(batch, k, d_out)`; **no sklearn API, no training loop, no preprocessing** [https://github.com/yandex-research/tabm]. We would have to write the optimizer/early-stopping/embedding/quantile preprocessing ourselves. pytabkit already wraps it (its README lists TabM D/HPO, "updated TabM code to a newer version").
- Decision: use `pytabkit` for both so RealMLP is a free fallback.

### Why TabM

- TabM paper (Gorishniy et al., ICLR 2025, https://arxiv.org/abs/2410.24210): "demonstrates the best performance among tabular DL models" while efficient; one model imitates an ensemble of k MLPs with shared parameters.
- RealMLP paper (Holzmüller, Grinsztajn, Steinwart, NeurIPS 2024, https://arxiv.org/abs/2407.04491): meta-tuned defaults, "competitive with GBDTs in terms of benchmark scores", "favorable time-accuracy tradeoff"; tuned on 118 datasets of 1k-500k samples.
- TabArena (NeurIPS 2025 D&B, https://arxiv.org/abs/2506.16791): deep learning "substantially closed the gap" with trees under larger budgets with ensembling. A web search snippet of the TabArena leaderboard (secondary, as of May 2026) listed RealMLP tuned+ensembled at Elo 1514 vs TabM 1448; I could not load the live leaderboard (HF Space), so treat that as unverified. On the published numbers RealMLP is not worse than TabM at full tuning. My reasons for TabM are therefore practical, not benchmark-rank:
  1. **Speed [measured]**: TabM_D 2.2-2.9 s/fold vs RealMLP_TD 34-39 s/fold on the 3080 Ti (5.9k rows, GPU). RealMLP's 256 epochs of a fixed schedule make a tune trial (5 folds) ~3 min vs ~12 s; a 30-trial RealMLP tune (~90 min) would hit `tune.py`'s 2 h cap, TabM's takes minutes.
  2. **Quality [measured, one seed, crude features, mean over 5 folds]**: TabM_D default 0.8070; TabM with `arch_type='tabm-mini', num_emb_type='pbld'` 0.8112; RealMLP_TD 0.7875; TabPFN-2 0.8153. One seed, so differences < ~0.005 are noise; the point is TabM is not weaker here.
  3. TabM is also the more different-in-kind partner (ensemble of MLPs with quantile preprocessing and optional PLR embeddings); either is an MLP family, so diversity from the GBDTs is similar. This is a judgement, not a measurement.
- pytabkit's `TabM_D_Classifier` defaults (`DefaultParams.TABM_D_CLASS`, read from the installed package): `arch_type='tabm'`, `tabm_k=32`, `num_emb_type='none'`, `lr=0.002`, `weight_decay=0`, `dropout=0.1`, `d_block=512`, `n_blocks='auto'`, `patience=16`, `batch_size=256`, `tfms=['quantile_tabr']`. These are "library defaults", not meta-tuned; pytabkit's README says its HPO spaces use TabM-mini with numerical embeddings. Recommend the component default to `arch_type='tabm-mini', num_emb_type='pbld'` [measured slightly better, same cost].

### Preprocessing, seeding, determinism

- Categoricals: auto-detected from pandas `category`/`object` dtypes (README) [measured: works on the harness frame].
- **NaNs in numeric columns are an error**: `ValueError: NaN values in continuous columns are currently not allowed!` **[measured, both TabM and RealMLP]**; README: "Missing numerical values are currently not allowed and need to be imputed beforehand." The Spaceship data has numeric NaNs (Age, spend, CabinNum...). The component must impute inside `fit` (training-fold median, stored and re-applied in `predict_proba`; optionally add missing-indicator columns). Learned from the training fold only, so no leak. Categorical NaNs are already the string `"nan"` after `to_matrix`.
- Scaling: none needed (the TD/D pipelines apply their own: RealMLP robust-scale + smooth clip, TabM quantile transform).
- Early stopping: pytabkit holds out `val_fraction=0.2` of whatever `fit` receives, which satisfies the registry contract ("`fit` only ever receives training-fold rows; any early stopping must split those"). Effective training set is 80% of the fold.
- Seeding: `random_state` (int) in the constructor; the component passes the harness `seed`. **[measured]** two fits with the same `random_state` on GPU gave bit-identical probabilities for TabM_D (GPU and CPU) and RealMLP_TD (GPU), 0 label flips. Cross-hardware not tested.
- GPU memory is small for these (MLPs on 5.9k rows).
- `device=None` auto-selects a GPU if present; on CI (no CUDA) it falls back to CPU without code changes.
- CPU smoke cost [measured, 32 threads, 1.6k train rows, after a ~32 s cold `import torch/pytabkit/tabpfn`]: TabM_D 41.5 s (early stopping not engaging as quickly on 1.6k rows), RealMLP_TD 11.8 s. On a 4-vCPU runner [estimate] 2-3x. Cap with `n_epochs` in smoke (see section 5).

---

## 3. Dependencies and CI

### This repo today

- `pyproject.toml`: requires-python >= 3.11; deps catboost, kaggle, lightgbm, matplotlib, numpy, optuna, pandas, scikit-learn, seaborn, xgboost. No torch. `uv.lock` is 609,073 bytes, universal (resolution-markers for win32/emscripten/other, py 3.11/3.12/3.14); the only CUDA-ish content is `nvidia-nccl-cu12` pulled by xgboost.
- `.github/workflows/ci.yml` job `smoke` (ubuntu-latest, `astral-sh/setup-uv@v3` with `enable-cache: true`, `uv sync`): unit tests; generate the synthetic fixture (`python -m scripts.gen_ci_fixture`, `$RUNNER_TEMP/ci-fixture`, default sizes); `harness.score <cfg> --smoke` for `configs/main.toml` and every `configs/exp/*.toml` (one fold of one seed on the first 2,000 rows, `harness/score.py` `SMOKE_ROWS = 2000`; only models a config enables are fit); `scripts.tune <model> --smoke` for every key of `tune.SPACES` (2 trials + recheck = ~3 fits, 300 rows, `SMOKE_CAPS` per model); `scripts.confirm --smoke`, `scripts.matrix --smoke`, `scripts.explain --smoke` (these use `configs/main.toml` = CatBoost only, and matrix's `SMOKE_CAPS`).
- Consequence: a new model component costs CI time only through (a) any `configs/exp/*.toml` that enables it (batch configs, `tune-<model>.toml`), (b) its `SPACES` entry (CI auto-loops `--list-models`), (c) `unittest` if tests import the registry (`registry.discover()` imports every component, so `import pytabkit`/`import tabpfn` must succeed at import time, or be deferred to `build`/`fit`; the existing components import their libs at module top, but that makes every script and test pay the ~30 s cold torch import). **Recommend importing torch-based libraries lazily inside `build()`/`fit()`** so discovery stays cheap.
- With "no skip path", the smoke run must really fit TabPFN and TabM. For TabPFN 3.5 that needs the `TABPFN_TOKEN` secret and weight download in CI (see HITL).

### Install size / time

[measured, Windows 11, fresh uv cache, `uv pip install` into a throwaway venv]:
- CPU torch from `https://download.pytorch.org/whl/cpu` (torch 2.14.1+cpu): **43 s**. Then `tabpfn 9.1.0 pytabkit 1.7.3 tabm 0.0.3` and their dependencies (pytorch-lightning, torchmetrics, skrub, safetensors, einops, huggingface-hub, pydantic-settings, rtdl-num-embeddings, ...): **31 s**. Whole venv: **852 MB**.
- CUDA torch from `.../whl/cu128` (torch 2.11.0+cu128): **790 s** (~13 min, download-bound, cold cache), venv **4.2 GB** (before adding the tabpfn/pytabkit deps). One-time cost locally; `torch.cuda.is_available()` was True with driver CUDA 13.1.
- Linux CI [estimate, not measured]: CPU wheel ~200 MB; installing PyPI's default Linux torch would instead pull the CUDA wheel plus `nvidia-*` libraries (GB-scale; uv docs: "PyPI provides GPU-accelerated wheels on Linux", https://docs.astral.sh/uv/guides/integration/pytorch/), so CI **must** use the CPU index. `setup-uv` caching turns reinstall into a cache restore.
- New direct deps: `torch`, `tabpfn`, `pytabkit` (the `tabm` PyPI package is not needed).

### uv configuration (verified by resolving a copy in a temp dir)

Single universal `uv.lock`, torch chosen per platform by markers, no extras and no CI workflow change beyond secrets:

```toml
[project]
dependencies = [
    # ...existing...
    "tabpfn>=9.1.0",
    "pytabkit>=1.7.3",
    "torch>=2.5",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
  { index = "pytorch-cpu",   marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

**[measured]** I copied this repo's `pyproject.toml` + `uv.lock` into a temp dir, added the above and ran `uv lock`: it resolved in 2.3 s (warm cache), lock grew 609 KB -> 887 KB, and the lock holds three torch entries selected by marker: `2.11.0+cu128` (win32), `2.14.1+cpu` (linux), `2.14.1` (PyPI, macOS/other). This is the marker pattern from the uv docs (section "CPU on Linux/Windows, GPU on ...", adapted so Windows gets CUDA and Linux CPU). Notes:
- The single-lock-across-platforms problem is solved by markers, not extras: uv locks all three, each platform installs its own. The extras + `[tool.uv] conflicts` pattern (uv docs, option 2) would force `uv sync --extra cpu` in CI and `--extra cu128` locally and split the lock's resolution; avoid it unless a Linux GPU box is added.
- Different torch versions per platform (2.11 vs 2.14) is a consequence of the CUDA index lagging; OOF caches are keyed by harness/data/component hashes (`score.cache_key`), not by torch version, so a cache computed on Windows would be reused on Linux for a different torch. Predictions are cached so this only matters if the cache were shared, which it is not (CI has no cache).
- Platform GPU: a Linux user with a GPU would get CPU torch; irrelevant here (local = Windows).
- `uv sync` locally on Windows then downloads ~3 GB of CUDA torch once (measured 13 min cold).
- Python: pytabkit/tabpfn support 3.10-3.14; the repo's >=3.11 is fine.
- Do not add the `uv.lock` change in this PR; it belongs to the batch ticket (the lock must be regenerated with the new deps).

### Smoke-run cost estimate

[estimates; anchors are the measured numbers above] CI `uv sync` adds ~1-3 min cold (cached after). Each new model's smoke fits on the CI runner (4 vCPU, 2k rows): TabM 1-2 min (set `n_epochs` cap), RealMLP not used, TabPFN-3.5 CPU 1-3 min plus a one-off weight download (cache it). `tune --smoke` for each new model fits 3 times: ~1-3 min each with caps. Total added CI time ~6-12 min per run of the `smoke` job (currently seconds-to-minutes). Not prohibitive. If it later is, the lever is smaller `SMOKE_ROWS`/caps, not a skip.

---

## 4. Tuning (`scripts/tune.py` `SPACES`)

Format today: `{param: (kind, low, high)}`, kinds `int`, `float`, `float_log`; unlisted params keep the component's `PARAMS`; trials are 1 seed x 5 folds (seeds 100-102), cap 30 trials / 2 h; top 3 re-checked on 3 seeds. CI requires an entry for every model that appears (and fits it in `--smoke`). `SMOKE_CAPS` in **both** `scripts/tune.py` and `scripts/matrix.py` (not `harness/`, so editable) need entries for expensive params.

Proposed spaces (my proposals; ranges are informed by the pytabkit defaults above and the usual TabM/MLP ranges, **not** validated by a search, **not** copied from the TabM paper's appendix, which I did not read):

```python
"tabm": {                      # fit ~2.5 s/fold, so 30 trials x 5 folds ~ 6-10 min
    "lr": ("float_log", 5e-4, 5e-3),
    "weight_decay": ("float", 0.0, 0.05),
    "dropout": ("float", 0.0, 0.4),
    "d_block": ("int", 128, 768),
    "n_blocks": ("int", 1, 4),
},
"tabpfn": {                    # few knobs; a minimal space only to keep phase 4 uniform
    "n_estimators": ("int", 4, 32),
},
```

- `tabm`: leave `tabm_k=32`, `num_emb_type`, `arch_type` fixed. `SMOKE_CAPS["tabm"] = {"n_epochs": 5}` (note `n_epochs` default is effectively unbounded with `patience=16`; the cap makes smoke seconds-long).
- `tabpfn`: TabPFN is a pre-trained foundation model; the vendor's own guidance is that tuning knobs is a small effect and the model is ensembling-limited by `n_estimators`. A one-knob space (`n_estimators`) is cheap to write and keeps phase 4 "unconditional" and CI's `--list-models` loop uniform. Argument for **no** richer space: `softmax_temperature`/`balance_probabilities` default to checkpoint-tuned values and are categorical/booleans that the `SPACES` kinds cannot express (adding a `categorical` kind to `tune.py` is possible but not worth it for a model with ~no gain). Cost: trial = 5 folds x (fit + predict 1.5k rows) ~ 40-100 s [estimate] (the test-set predict that `fit_predict` does is skipped by `cv_accuracy`, which only predicts the validation fold), 30 trials ~ 20-50 min [estimate]. `SMOKE_CAPS["tabpfn"] = {"n_estimators": 1}`.
- Pre-condition in the tune cost: the search saw the dev rows, so the keep rule's "majority of better folds" already guards it; nothing to change.
- If the batch's phase 4 is allowed to skip a model with "no space needed", say so in the ticket; today the skill and CI loop assume an entry exists.

---

## 5. Component fit: what strains the harness

How a model component is defined (`harness/registry.py`, `components/catboost.py`, `components/xgboost.py`): a file `components/<name>.py` with `NAME` (== file stem), `KIND = "model"`, optional `PARAMS` dict (defaults a config's `[params.<name>]` can override), and `build(params, seed)` returning an object with `fit(X, y)` and `predict_proba(X)` returning an `(n, 2)` array (harness takes column 1). `X` is a pandas DataFrame whose non-numeric columns are `category` dtype (`data.to_matrix`); NaN categories were stringified to `"nan"`, numeric NaNs remain. A new instance is built per fold (`score.fit_predict`: `comp.build(dict(params), seed)` per seed x fold) with `seed` == split seed (0,1,2 for scoring, 100-102 tune, 200-202 confirm). Predictions (OOF + test) are cached under `cache_key` = features + component source hash + params + seeds + harness hash + data hash; the cache lives in the shared main clone, so each model trains once per distinct key. Wrappers carry categorical bookkeeping inside the class (catboost converts to str; xgboost fixes categories at fit time).

Strain points, in order of severity:

1. **TabPFN `predict_proba` must chunk.** The harness predicts the whole test set (4.3k rows) each fold; single-call on the 12 GB card hung > 18 min [measured, v2]. Chunk to <= 1,000 rows inside the component. (README guidance: chunks of 1,000.)
2. **Predicting test rows every fold multiplies TabPFN cost.** `fit_predict` predicts val (~1.5k) **and** the full test set each fold, so ~5.8k rows/fold at ~5.4 ms/row [measured, v2] ~30 s/fold. `explain.py` also calls `predict_proba` many times (once per component group x repeats on the validation fold) with no caching; budget accordingly or lower `n_estimators` there.
3. **Run-time budget.** v2 measured ~7.5 min per full 3-seed score on GPU; 3.5 estimated 10-25 min. A batch multiplies this: singles (each new model once), matrix cells that change features retrain (cache key includes features), confirmation seeds 200-202 for base **and** best cell (another 2 x 15 folds per model, uncached for new seeds), phase-4 auto-tune of every model in the result (TabM ~10 min; TabPFN 20-50 min; RealMLP would exceed the 2 h cap). Rule of thumb: a TabPFN-3.5 model in a result costs ~1-2 h of GPU across a batch [estimate]. Not prohibitive, but the 3.5-Fast checkpoint is the lever.
4. **NaN handling differs**: pytabkit rejects numeric NaN; the component must impute from the training fold. TabPFN takes NaN as is.
5. **Categoricals**: the `category` dtype with unseen levels at predict time (a level absent from the training fold): xgboost's wrapper pins categories at fit. TabPFN and pytabkit auto-detect from dtype; TabM's one-hot/embedding map built at fit; I did not test an unseen level for either. The wrapper should convert categoricals to a fixed vocabulary like the xgboost component does.
6. **Seeding**: TabPFN `random_state` defaults to **0**, not None; pass the harness seed. pytabkit `random_state` defaults to None; pass the harness seed. Both were bit-reproducible same-machine on GPU [measured]; the docs warn about cross-hardware drift.
7. **Per-fold weight load**: the harness builds a fresh estimator per fold. v2 weights loaded once per process (later fits 0.3 s) [measured]; 3.5's 876 MB checkpoint load is unmeasured. If per-fold load shows, cache the loaded model in a module-level dict. Also free GPU memory between folds (`del est; torch.cuda.empty_cache()` is not triggered by the harness; the component can do it at end of `predict_proba` if memory creeps).
8. **GPU memory across folds**: v2 peaked at 6.9 GB with 1,000-row chunks [measured]; TabM small. Local runs are sequential (one machine), so no contention, but `nvidia-smi` showed ~0.8 GB already used by the desktop.
9. **Device handling**: nothing in the harness passes a device. Components should use `device=None/"auto"` (pytabkit, TabPFN both auto-select CUDA) so the same file runs locally and in CPU CI. On CPU, TabPFN needs `TABPFN_ALLOW_CPU_LARGE_DATASET=true` for folds > 5,000 rows (full runs on CPU are not practical); in the smoke (<= 2,000 rows -> 1.6k train rows) 3.5's 5,000-row limit is not hit (the v2 1,000-row limit would be).
10. **Import cost / registry discovery**: `registry.discover()` imports every component module; a top-level `import torch` costs ~30 s cold on this machine [measured, imports of torch + pytabkit + tabpfn]. Import inside `build()`.
11. **Parallelism / threads**: pytabkit and torch use all cores by default; harness is sequential, so no oversubscription. `n_threads` exists as a pytabkit parameter if CI needs it.
12. **Frozen harness**: all of this is doable as `components/` + `configs/` + `scripts/` + `pyproject.toml`/`uv.lock` + `ci.yml` changes; `harness/` stays untouched. `scripts/tune.py` and `scripts/matrix.py` (the `SPACES` and `SMOKE_CAPS`) are outside `harness/`. The freeze-check job only inspects `harness/`.

---

## Recommendations for the batch

- **TabPFN version:** TabPFN-3.5 (`create_default_for_version(ModelVersion.V3_5)`); 3.5-Fast if per-run time exceeds ~25 min; TabPFN-2 only as the ungated fallback.
- **HITL setup first (blocks the 3.5 smoke and any run):** user accepts the TabPFN-3.5 licence at https://ux.priorlabs.ai (Kaggle use is expressly permitted), obtains a token, sets `TABPFN_TOKEN` locally (or runs once with a browser) and adds it as a GitHub Actions secret; add model-cache `actions/cache` for the 876 MB checkpoint. No HITL needed for TabM or for TabPFN-2.
- **DL model:** TabM via `pytabkit.TabM_D_Classifier` (`arch_type='tabm-mini'`, `num_emb_type='pbld'`), not RealMLP: ~15x faster per fold [measured] and not weaker here; RealMLP is available from the same package if wanted. Impute numeric NaNs in `fit`; pass `random_state=seed`; `device=None`.
- **Deps plan:** add `torch>=2.5`, `tabpfn>=9.1.0`, `pytabkit>=1.7.3` to main `dependencies`; `[tool.uv.sources]` markers: `sys_platform == 'win32'` -> cu128 index, `'linux'` -> CPU index, both `explicit = true`; single universal lock (verified resolves). Local first install ~13 min / ~4 GB; CI install ~1-2 min cold [estimate], cached after. Import torch libs lazily in `build()`.
- **Tuning spaces:** `tabm` {lr float_log 5e-4..5e-3, weight_decay 0..0.05, dropout 0..0.4, d_block int 128..768, n_blocks int 1..4}; `tabpfn` {n_estimators int 4..32}. Add `SMOKE_CAPS` entries in both `tune.py` and `matrix.py` (`tabm: n_epochs`, `tabpfn: n_estimators=1`).
- **Harness caveats to carry into the component specs:** chunk TabPFN `predict_proba` at <= 1,000 rows (unchunked 4.3k test rows hung the 12 GB GPU); fix categorical vocabulary at `fit`; seed both libraries explicitly (TabPFN defaults to `random_state=0`); expect per-run cost of ~7.5 min (TabPFN-2, measured) to ~25 min (3.5, estimated) per 15 folds and ~1-2 h of GPU for a TabPFN-3.5 model across screen + matrix + confirm + tune; CPU only for smoke.
- **Ordering:** run the HITL task, then a component PR (components, SPACES, SMOKE_CAPS, pyproject/uv.lock, ci.yml with the secret and cache), then the blend batch (#77). Measure 3.5's real fold time on the first run and revisit Fast vs default.

### Source index

- TabPFN: https://github.com/PriorLabs/TabPFN (README, `src/tabpfn/classifier.py`, `src/tabpfn/model_loading.py`); https://pypi.org/pypi/tabpfn/json; https://docs.priorlabs.ai/models; https://huggingface.co/Prior-Labs/tabpfn_3_5 (+ `/raw/main/LICENSE`, HF API); arXiv 2609.17895 (3.5), 2511.08667 (2.5).
- pytabkit: https://github.com/dholzmueller/pytabkit (README, `sklearn_interfaces.py`); https://pypi.org/pypi/pytabkit/json. TabM: https://github.com/yandex-research/tabm, https://pypi.org/pypi/tabm/json, arXiv 2410.24210. RealMLP: arXiv 2407.04491. TabArena: arXiv 2506.16791, https://github.com/autogluon/tabarena.
- uv + PyTorch: https://docs.astral.sh/uv/guides/integration/pytorch/.
- This repo: `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `harness/registry.py`, `harness/score.py`, `harness/data.py`, `harness/config.py`, `components/catboost.py`, `components/xgboost.py`, `scripts/tune.py`, `scripts/matrix.py`, `scripts/gen_ci_fixture.py`, `CONTEXT.md`.
