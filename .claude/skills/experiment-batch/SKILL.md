---
name: experiment-batch
description: Run a batch of screened ideas against the frozen harness (screen against a pinned base, matrix every combination, confirm the winner on fresh seeds, auto-tune, then a leaderboard-gated landing) and resolve the result. Use when a wayfinder:experiment ticket on the Spaceship Titanic map is claimed — the ticket's own session is the planner this skill drives.
---

An experiment ticket no longer chases one idea; it dispatches a **batch**. This session, the one that claimed the `wayfinder:experiment` ticket, **is the planner**. It runs a screen-then-matrix search over `components/` and `configs/` against `harness/` (frozen — see `README.md` and `harness/registry.py` for the component contract, `CONTEXT.md` for vocabulary), using `Agent`-tool subagents as workers and a reviewer for phase 1's free-form ideas, and the scripts under `scripts/` (`matrix.py`, `confirm.py`, `tune.py`, `submit_leaderboard.py`, `base_lb.py`, `explain.py` — all built on the frozen harness, none of them part of it) for everything mechanical from phase 2 on. Workers and the reviewer are subagents of *this* session, not tickets and not dispatcher-visible sessions: no lane slot is consumed, and — this is load-bearing, see [Workers never touch GitHub](#non-negotiables) — they hold no `gh` access.

Full background: [ML research orchestration: batched experiments over a frozen harness](https://github.com/vltnkiz/ticket-dispatch/issues/18)'s Notes, and this map's Notes. Read both before running a batch if anything below is unclear; this skill is the operational half, they are the reasoning behind it.

## Shape of a batch

```
Phase 0  Setup       pin/verify the base, check the base's LB is recorded, load or init batch state
Phase 1  Screen      for each idea: worktree -> worker -> reviewer -> score -> record -> cleanup
Phase 2  Matrix      score every on/off combination of the screened ideas
Phase 3  Confirm     re-score the winning cell and the base on confirmation seeds; that sets the label
Phase 4  Auto-tune   tune every model of the result, in turn, on the tuner's own seeds
Phase 5  LB gate     one submission for the final config, only if it's a keep or near-miss
Land     Merge survivors into worktree-ticket-<n>, push, resolve the ticket
```

Never more than **4 ideas × 2 attempts** in a batch. Check headroom **between** ideas, not mid-idea (see [Budget and graceful stop](#budget-and-graceful-stop)).

## Phase 0 — setup

1. **Pin the base.** Run `uv run python -m harness.score configs/main.toml`. If the report says the base is stale, re-pin it: `uv run python -m harness.score configs/main.toml --pin`. Record the resulting `cv`, `digest`, and `commit` as this batch's `base` — fixed for the whole batch, even if `configs/main.json` changes later (it won't: the work gate keeps a repo with an open landing ticket from claiming a new code-writing ticket, so no other batch can move `main` under this one).
2. **Check the base's LB is recorded.** Run `uv run python -m scripts.base_lb`. Phase 5's gate reads this, not a fresh submission of the base — recording a base's public score costs a submission only the first time. If it prints `no base LB recorded` or flags `STALE` (the pin has moved since it was last recorded, meaning a prior batch's landing skipped this step), that itself is a small, quota-permitting fix, not a blocker to raise: check quota (`uv run python -m scripts.submit_leaderboard --quota`); if any remain, submit the base once (`uv run python -m scripts.submit_leaderboard configs/main.toml -m "base LB: <why>"`) and record it (`uv run python -m scripts.base_lb SCORE -m "<why>"`). If quota is 0, this **is** a blocker — Phase 5 cannot gate against an unknown base score — so stop per [Dispatched sessions](#dispatched-sessions) rather than guess at a number.
3. **Load or init batch state** at `.batch/ticket-<n>.json` (gitignored; `<n>` is this ticket's number). If it exists and its `ticket` field matches, this is a **resume** — a prior instance of this same session was killed and relaunched by the dispatcher's crash retry or rate-limit resume. Skip re-sweeping anything already `screened` (any of `keep`/`drop`/`unmeasured`); pick up mid-idea or mid-phase from the state.

   ```jsonc
   {
     "ticket": 24,
     "started": "2026-09-27T12:00:00Z",
     "base": {"commit": "...", "cv": 0.80917, "digest": "...", "lb": 0.80523},
     "ideas": {
       "toy_a": {"status": "pending", "attempts": []}
       // status: pending | screening | keep | near_miss | drop | unmeasured
       // keep: single-screen CV delta >= +0.002. near_miss: delta >= 0 but < +0.002. drop: delta < 0.
       // this label is only a report (CONTEXT.md's Keep / near-miss / drop) -- every screened
       // idea, whatever its label, enters the matrix; only `unmeasured` ideas are left out.
       // each attempt: {"worker_run": "...", "reviewer": "ok|bug|concern", "note": "...", "result_path": "..."}
     },
     "phase": "screen",           // screen | matrix | confirm | tune | gate | done
     "matrix": null,              // filled by phase 2: {"cells": [...], "winner": {...}}
     "confirmation": null,        // filled by phase 3: {"winner": {...}, "base": {...}, "verdict": "..."}
     "tuned": null,               // filled by phase 4: {"model": {"before": cv, "after": cv, "kept": bool}, ...}
     "gate": null                 // filled by phase 5: {"submitted": bool, "public_score": ..., "landed": bool}
   }
   ```

   Write the state file after every meaningful step (idea claimed, attempt recorded, phase change) — it is the thing a resumed session reads, not memory.
4. **The ticket's theme** (its `## Question`, plus whatever the map's fog patch it graduated from said) yields the idea list, up to 4. Write them into `ideas` before starting phase 1. An idea is a short name plus a one-line description of what to try — the worker fills in the rest. **At most one of the up-to-4 may be a tuning idea** (a search over `scripts/tune.py`, see [Tuning ideas](#tuning-ideas)) — the planner's own choice, targeting a model already enabled (weight > 0) in `configs/main.toml`.

## Phase 1 — screen

For each `pending` idea, in order:

1. **Cut a worktree**: `git worktree add .claude/worktrees/exp-<idea> -b exp/<idea>` off the ticket branch (`worktree-ticket-<n>`, i.e. `HEAD`) — **nested under the ticket worktree itself**, not as a sibling of it. Mark the idea `screening` in state.

   This nesting isn't a style choice: an `Agent`-tool subagent is sandboxed to the *planner's own* pinned worktree and cannot reach a path outside it, even a sibling git worktree that exists on disk. Give the worker the **full path including the ticket worktree prefix** (e.g. `.../worktrees/ticket-<n>/.claude/worktrees/exp-<idea>`, not `.../worktrees/exp-<idea>`) — the shorter path looks right but doesn't exist from the worker's vantage point and it will report back unable to find it.
2. **Dispatch a worker** (`Agent` tool, no `subagent_type` override needed — a fresh general-purpose agent with no memory of this session is exactly right here, since it must not inherit any GitHub context). Give it, self-contained:
   - The idea's name and one-line description.
   - Its working directory: the `exp/<idea>` worktree.
   - The **component contract**, verbatim from `harness/registry.py`'s docstring and `README.md`'s "Adding an idea" section (paste both in; do not paraphrase — the contract's exact wording is what the reviewer checks against).
   - Explicit scope: it may create or edit files under `components/` and `configs/exp/` **only**. It must not touch `harness/`, `results/`, `.batch/`, or run `git` commands beyond inspecting its own worktree. **It must never call `gh`, or read/write anything about issues, labels, or assignment** — that is the planner's job alone.
   - Its task: add one component (or a small number that together express the idea) and one `configs/exp/<idea>.toml` (`extends = "main"` plus the new lines), then run `uv run python -m harness.score configs/exp/<idea>.toml` and report the full output plus a diff of everything it changed (`git diff` inside its worktree).
   - Tell it to report back **the diff and the raw score output**, not a summary — the reviewer needs the diff, and the planner needs the exact numbers.
3. **Dispatch the reviewer** (`Agent` tool, fresh, fed the worker's diff and score output — not the worker's own commentary about whether it's fine). Give it the checklist verbatim from [Reviewer checklist](#reviewer-checklist) below and this instruction: **answer every item before being told the score**; withhold the CV number and verdict from its prompt entirely, only the diff and the fact that it ran. Its answer is `ok`, `concern: <note>`, or `bug: <file>:<line> <what's wrong>`. A `bug` verdict with no cited file:line is not a bug verdict — treat it as `ok` and note the reviewer failed to cite.
4. **Resolve the attempt:**
   - `bug` (cited) → the run is **invalid**: do not record its score anywhere. This attempt is spent. If this idea has a remaining attempt (≤ 2 total), discard the worktree, cut a fresh `exp/<idea>` worktree, and retry with the bug quoted in the worker's brief so it doesn't repeat it. If attempts are exhausted, mark `unmeasured` with the cited bug as the reason; move on.
   - `ok` or `concern` → the score stands (a `concern` is recorded as a note, not a rerun). Apply the keep rule from `CONTEXT.md`: **keep** if CV improves on the pinned base by ≥ +0.002; **near_miss** if it improves (CV delta ≥ 0) but by less than +0.002; **drop** if CV delta < 0. **This label is only a report** (`CONTEXT.md`'s **Keep / near-miss / drop**) — commit the worker's `components/`/`configs/` files onto the ticket branch (`git -C <ticket-worktree> checkout exp/<idea> -- components/ configs/exp/`, or cherry-pick the worker's commit if it made one), so the idea's code and its `results/<run>.json` both land in `worktree-ticket-<n>` regardless of label (a drop's file is what makes it re-verifiable rather than re-arguable). Every screened idea — keep, near_miss, **and drop** — enters phase 2's matrix; only `unmeasured` (no valid run at all) is left out. There is no per-idea leaderboard check any more: the leaderboard gate runs once, in phase 5, against the batch's final config.
   - A crash (worker or reviewer erroring, not a cited bug) also spends an attempt and retries once, same as a cited bug — but is **not** what "a bad score is a result, not a failure" is about: only a completed, reviewed run's score is final.
5. **Clean up**: `git worktree remove .claude/worktrees/exp-<idea>` and `git branch -D exp/<idea>` once its code is merged onto the ticket branch (or once it's abandoned unmeasured — nothing from a dead attempt is kept).
6. **Check budget** ([below](#budget-and-graceful-stop)) before starting the next idea.

Only one worker runs at a time to start. The loop above is written per-idea, not hardcoded to one — running N workers concurrently (still all sequential with the reviewer, since a review needs a finished diff) is a config change, not a rewrite, when that number becomes anything but 1.

### Tuning ideas

A tuning idea replaces steps 1–3 above with a single deterministic script call — there's no free-form code for a reviewer to check, so there's no worker/reviewer dispatch:

1. Mark the idea `screening` in state (same as any idea), then run, directly on the ticket branch (no `exp/<idea>` worktree — `scripts/tune.py` only ever writes `configs/exp/<name>.toml`, never `components/`): `uv run python -m scripts.tune MODEL`. `MODEL` must already be enabled in `configs/main.toml`; the search itself is capped at ≤30 trials / ≤2h (`scripts/tune.py`'s own default, see its docstring) — never raise it past that default. Its sqlite study (`.tune/<model>.db`, gitignored) means a crash mid-search resumes on retry rather than losing progress.
2. Score the written config the normal way: `uv run python -m harness.score configs/exp/<name>.toml`. **The keep bar is stricter than an ordinary idea's**, because the search already re-partitioned the same dev rows it's later judged on (see `CONTEXT.md`'s **Tuning idea** entry): **keep** needs CV delta ≥ +0.002 **and** ≥11 of 15 folds better; short of either part but still non-negative is a **near_miss** (still matrix-eligible); negative is a **drop**. This replaces the ordinary keep/near_miss/drop test from step 4 above for this idea only, and, like any single-screen label, it is only a report — everything else (commit the config onto the ticket branch regardless of label, the ≤2-attempts-per-idea cap, entering phase 2's matrix) applies exactly as it does to any other idea. This is a *phase-1* tuning idea (the planner's own choice, screened like any idea); it is separate from phase 4's auto-tune, which is not optional and runs on every model of the batch's result regardless of what phase 1 screened.
3. Skip the worktree cleanup in step 5 above — there is none to remove.

## Phase 2 — matrix

Skip straight to landing (with nothing new) only if **every** idea came back `unmeasured` — with at least one valid screen, even an all-drop one, the matrix still runs, because a solo drop can combine well (this is the case a full matrix catches that the old greedy combine couldn't: it never tried a solo drop, and couldn't reach a pair of near-misses).

Score **every on/off combination** of the batch's screened ideas — 2^k cells for k ideas (≤4 ideas ⇒ ≤16 cells), the all-off cell being the pinned base itself — with `scripts/matrix.py`, which merges each idea's already-written `configs/exp/<idea>.toml` config-only (no new components, no re-review) and scores each cell the normal way (`harness.score`, default seeds 0-2, cached):

```
uv run python -m scripts.matrix idea_a idea_b idea_c idea_d
```

- If any two ideas are **mutually exclusive** (the planner's own call — e.g. two features that redundantly encode the same signal, or a feature designed to replace another outright), pass `--exclude idea_a,idea_b` once per excluded pair; those cells are skipped rather than scored.
- A **combiner** idea (`configs/exp/<idea>.toml` = `extends = "main"` plus a `[combiner]` table, see README "The combiner") is config-only and goes through the matrix like any other. Two combiner ideas always exclude each other: pass `--exclude` for every pair (the matrix errors on a cell holding two). Auto-tune never tunes combiner parameters.
- Every idea from phase 1 enters, regardless of its single-screen label — keep, near_miss, **and drop**. Only `unmeasured` ideas (no valid run) are left out, since there is nothing to combine.
- Record the full cell table and the winning cell (highest CV; may be the all-off cell, i.e. no idea combination beat the base) as `matrix` in state.
- A CatBoost run is about 2–7 min, so a full 16-cell matrix is roughly 0.5–2h; this is the batch's main time cost, budget accordingly (see [Budget and graceful stop](#budget-and-graceful-stop)).

## Phase 3 — confirm

The matrix winner is, by construction, whichever cell scored highest on the CV seeds — the single noisiest pick among however many were tried (`CONTEXT.md`'s **Confirmation seeds**: picking the top of many noisy scores inflates it). Re-score it, and the base, on seeds neither the single screens nor the matrix ever touched:

```
uv run python -m scripts.confirm configs/exp/_matrix/cell-<winner>.toml --against configs/main.toml
```

(If the winning cell is the all-off cell — no combination beat the base — there is nothing to confirm against itself; the batch's result is unchanged from the base, skip straight to phase 4 with the base as "the result", nothing to submit in phase 5.)

`scripts/confirm.py` prints the delta and a verdict on the confirmation seeds' own numbers — **this delta, not the matrix's, is the label that decides the batch** (`CONTEXT.md`'s **Keep / near-miss / drop**): keep ≥ +0.002, near-miss ≥ 0, drop otherwise. Record it as `confirmation` in state. A confirmed `drop` still writes `configs/exp/batch-<n>.toml` for the winning cell (the report), but the batch's result for phases 4–5 is the *base*, not the dropped cell — a batch-level drop is not landed as a code change.

## Phase 4 — auto-tune

Not optional, and not gated by phase 3's label: whatever the batch's result is (the confirmed cell, or the base if phase 3 confirmed a drop or the matrix winner was the all-off cell), tune **every model enabled in it, one at a time**, chaining each accepted tune into the next model's baseline:

1. For each model with weight > 0 in the result: `uv run python -m scripts.tune MODEL` (seeds 100-102, ≤30 trials / ≤2h — never raise past `scripts/tune.py`'s own defaults), then score the written config against the *current* result (`uv run python -m harness.score`, or `scripts.confirm --against` if re-confirming — ordinary CV seeds are enough here, tuning already used disjoint seeds). Apply the tuning keep bar from `CONTEXT.md`'s **Tuning idea**: keep needs CV delta ≥ +0.002 **and** ≥11 of 15 folds better; short of either but non-negative is a near_miss (kept anyway — a tuning near-miss is still better or equal, so take it); negative is a drop (discard the tune, keep the untuned params).
2. A kept or near-miss tune becomes the new baseline for the *next* model's tune. Record each model's before/after CV and its verdict as `tuned` in state.
3. When every enabled model has been tried, the final tuned config **is** the batch's result. Write it (or re-write `configs/exp/batch-<n>.toml`) to reflect every accepted tune.

## Phase 5 — leaderboard gate

One submission, for the batch's final config, only if phase 3's confirmation labelled it **keep** or **near_miss** (a confirmed drop, or an unchanged base, is never submitted — nothing changed to gate). Compare against the base's own public score, not CV:

1. Read the base's LB: `uv run python -m scripts.base_lb` (phase 0 already ensured this is present and not stale).
2. Check quota: `uv run python -m scripts.submit_leaderboard --quota`.
   - **0 remaining:** a `keep` lands anyway, flagged **unverified** in the resolution comment and state — do not hold it. A `near_miss` cannot be verified this batch; it **waits**: land nothing this batch beyond what phase 4 already produced as the base, and put the near-miss config back in the map's fog for a future batch's quota to confirm.
   - **Quota available:** submit — `uv run python -m scripts.submit_leaderboard configs/exp/batch-<n>.toml -m "batch-<n>: <one-line summary>"` — and read the printed public score.
3. Apply the gate (`CONTEXT.md`'s **Leaderboard gate**, band 0.010 = √(0.005²+0.009²)):
   - `keep` lands **unless** LB < base LB − 0.010, in which case it is **auto-dropped** — no human hold, no exception. The old CV-vs-LB veto's `hitl` hold is retired; this is a deterministic rule now.
   - `near_miss` lands **only if** LB > base LB + 0.010; otherwise it does not land (record the reading, put it back in the fog — it may combine differently in a future matrix).
4. If it lands: this config is the new pinned base. `uv run python -m harness.score configs/main.toml --pin` after copying the batch's config over `configs/main.toml` (on the ticket branch, before merging), then `uv run python -m scripts.base_lb <public_score> -m "batch-<n>: <one-line summary>"` to record the new base's LB — this is what makes the *next* batch's gate work without re-submitting.
5. Record `gate` in state: whether submitted, the public score (if any), and whether it landed.

## Landing

1. Merge the batch's final result onto `worktree-ticket-<n>` (it should already be there from phase 1's per-idea merges and phase 4's tuned config — landing adds the combined/tuned `configs/exp/batch-<n>.toml`, and, if phase 5 promoted it, the updated `configs/main.toml` / `configs/main.json` / `configs/main.lb.json`).
2. Confirm no `exp/*` worktrees or branches remain, and `configs/exp/_matrix/`'s scratch cells are left as-is (gitignored — nothing to clean up).
3. If the landing session is this one (it usually is — see `README.md`'s and `CONTEXT.md`'s note that the holdout is read only at landing), run `uv run python -m harness.audit configs/exp/batch-<n>.toml` and quote its number in the resolution comment. **Do not** write it into `results/`, the config, or the map.
4. **Explain the result** (`CONTEXT.md`'s **Explanation**): if the batch's final config differs from the phase-0 base in any way (tune-only changes included), explain it against the phase-0 base. By now phase 5 may have copied the result over `configs/main.toml`, so take the base from its own commit (state's `base.commit`) into the gitignored scratch dir:

   ```
   git show <base.commit>:configs/main.toml > configs/exp/_matrix/main.toml
   uv run python -m scripts.explain configs/exp/batch-<n>.toml --against configs/exp/_matrix/main.toml
   ```

   Commit the two `results/explain/*.json` files it writes and put its table in the resolution comment, labelled descriptive and not a keep input. Nothing in it changes a label, a matrix pick, or the gate. If the result is unchanged from the base, skip the run and write "result unchanged, no explanation".
5. **Pin the seed-0–2 CV of the landed config** (`CONTEXT.md`'s **Batch**): if phase 5 promoted a new base, its CV is already pinned by step 4 there; if nothing was promoted, note the confirmed result's CV (seeds 0-2, from the matrix or its own score) in the resolution comment anyway, alongside the confirmation-seed number that set its label — the two are expected to differ slightly (confirmation seeds are one honest re-draw, therefore noisier for a single config; quoting both, not just the flattering one, is the point).
6. `git push -u origin worktree-ticket-<n>`.
7. Resolve the ticket per `docs/agents/issue-tracker.md`'s wayfinding operations: comment with the recap below, close the issue, and (this being a `wayfinder:experiment` ticket on the map, handled the same as any other resolution) append a one-line gist to the map's Decisions-so-far pointing at the ticket.

**The resolution comment** is what the user actually reads at landing. It must contain, and must not contain anything it can't support:

- The pinned base going in: commit and CV.
- A delta table, one row per idea in the batch: name, status (keep / near_miss / drop / unmeasured), CV delta (omit for unmeasured), reviewer verdict and any note, attempts spent. A tuning idea's row has no reviewer verdict (nothing was reviewed); show its folds-better count instead.
- The **matrix**: every cell scored, ranked, with the winner marked.
- The **confirmation**: the winning cell's and the base's CV on the confirmation seeds, the delta, and the verdict that decided the batch.
- The **auto-tune** pass: each tuned model's before/after CV and verdict, in the order tuned.
- The **leaderboard gate** reading, if it ran: the public score next to the base LB it was checked against, the band, and whether it landed, was auto-dropped, waited for quota, or was flagged unverified.
- The holdout reading, if fetched this session, labelled as a smoke-test number, not a keep input.
- The **explanation** table from Landing step 4 (or "result unchanged, no explanation"), labelled descriptive, not a keep input.
- Any ideas from the ticket's theme that were never run (budget or usage cut the batch short), and any near-miss left waiting on quota — these go back into the map's fog (**Not yet specified**), not silently dropped.

## Reviewer checklist

Run on every screened keep and on any diff touching `harness/` (should never fire — a tripwire). Answered **item by item, before the score is shown**:

1. Does the diff touch anything outside `components/` and `configs/exp/`?
2. Is any statistic (mean, fill value, encoding, threshold) computed over rows outside the training folds it will be applied to — i.e. does a feature step compute something from `dev + test` combined that a real deployment couldn't know, or from all folds instead of just the training ones?
3. Does anything read the target column (`Transported`) for rows it then predicts, directly or via a joined/grouped statistic that includes those rows' own label?
4. Is any stopping criterion, hyperparameter, or threshold chosen by looking at the scored fold or the CV result itself (vs. fixed up front, as `lgbm`'s `PARAMS` are)?
5. Is the scorer used the harness's own, unmodified — no local monkeypatching, no bypassing `harness/score.py`?

Verdict: `bug: <file>:<line> <what's wrong>` for any yes on 2–5, or any touch on 1 outside the allowed dirs; `concern: <note>` for something suspicious but not a clear violation (e.g. a feature that's technically fold-safe but fragile); `ok` otherwise. A bug verdict the planner can't find at the cited line does not count — go back to the reviewer once with the mismatch, and if it can't produce a real citation, treat as `ok` with a note that review was inconclusive.

## Non-negotiables

- **A worker never touches GitHub.** No `gh`, no issue reads, no labels, no assignment. This is what makes fan-out safe again after [#10](https://github.com/vltnkiz/ticket-dispatch/issues/10)'s race — spell this out explicitly in every worker's brief, don't rely on it being obvious from tool access alone.
- **A bad score is a result, not a failure.** Negative CV delta ⇒ drop, non-negative-but-below-threshold ⇒ near_miss (still matrix-eligible) — either way recorded with its number, no retry. Retries exist only for a cited reviewer bug or a crash.
- **No shared append log.** Every run writes its own `results/<run-id>.json` (the harness already does this); never edit another run's file, never maintain a batch-wide log outside `.batch/ticket-<n>.json`, which is state, not the record.
- **Budget is a hard cap**, not a target: ≤ 4 ideas, ≤ 2 attempts each.
- **State survives a kill.** Write `.batch/ticket-<n>.json` after every step that would be expensive to redo. A resumed session trusts it over re-deriving anything.
- **A stopped batch still lands what it has.** Losing headroom mid-batch is not an error path to apologize for: abandon the in-flight worktree, run whichever of phases 2–5 the survivors still support (even a matrix of one idea, even zero), merge, push, resolve with the delta table plus an explicit list of unrun ideas for the fog. A batch that finishes one of four ideas is a small success.
- **Clean up `exp/*` worktrees and branches** before finishing; only the ticket branch leaves this session.
- **The leaderboard gate is deterministic, never a human hold.** A keep auto-drops on a bad LB reading; nothing about phases 2–5 pauses for a person mid-batch (the sole exception is phase 0's base-LB check, when quota is 0 and there is truly no number to gate against).

## Budget and graceful stop

The hard limit is the idea/attempt cap above — that alone bounds a batch, but phase 2's matrix is now the dominant time cost (up to 16 full CV scores). Stop starting a **new** idea (phase 1) or a **new** matrix cell (phase 2) once this session's headroom looks thin: if your environment surfaces a usage or context indicator, treat crossing roughly 70% as the stop signal (below the dispatcher's 80% claim guard, so a batch has room to land tidily rather than being cut off mid-merge); if no such signal is available in-session, treat the idea cap itself as the operative limit and prefer stopping early over risking a mid-merge cutoff. Either way, the response to hitting the limit is the same: stop starting new work, finish or abandon whatever's in flight (an in-flight matrix cell can simply be dropped — cells are independent and idempotent to resume), and go straight to [Landing](#landing) with what's screened and scored so far. A batch that never reaches phase 4 or 5 still lands its matrix result as a report; auto-tune and the leaderboard gate are what a *future* batch can pick up, not a reason to hold this one open.

The 10/day leaderboard quota is a separate budget from the above and never stops phases 1–4, only phase 5's submit: at most one submission per batch (plus, rarely, one to establish a never-before-submitted base's LB), so a batch stays comfortably inside 10. If `--quota` reports 0 at phase 5, land a `keep` anyway flagged unverified; a `near_miss` waits for a future batch's quota, per phase 5's own instructions above.

## Dispatched sessions

This ticket is claimed and worked by a dispatched session exactly as `wayfinder`'s own "Dispatched sessions" section describes — read it before starting if you haven't this session. In particular: resolving the ticket sends a one-line gist with `PushNotification` (this is an AFK ticket type; nobody is watching a terminal), and if the batch hits a decision only a human can make (the idea list itself is ambiguous, the keep rule doesn't cover a tie, a worker's diff can't be reviewed because the contract itself is unclear), stop rather than guess: comment what it's blocked on, add `hitl` and `ready`, unassign, notify, end the turn.
