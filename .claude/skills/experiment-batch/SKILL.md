---
name: experiment-batch
description: Run a batch of screened ideas against the frozen harness (screen against a pinned base, then greedily combine survivors) and land the result. Use when a wayfinder:experiment ticket on the Spaceship Titanic map is claimed — the ticket's own session is the planner this skill drives.
---

An experiment ticket no longer chases one idea; it dispatches a **batch**. This session, the one that claimed the `wayfinder:experiment` ticket, **is the planner**. It runs a screen-then-combine search over `components/` and `configs/` against `harness/` (frozen — see `README.md` and `harness/registry.py` for the component contract, `CONTEXT.md` for vocabulary), using `Agent`-tool subagents as workers and a reviewer. Workers and the reviewer are subagents of *this* session, not tickets and not dispatcher-visible sessions: no lane slot is consumed, and — this is load-bearing, see [Workers never touch GitHub](#non-negotiables) — they hold no `gh` access.

Full background: [ML research orchestration: batched experiments over a frozen harness](https://github.com/vltnkiz/ticket-dispatch/issues/18)'s Notes, and this map's Notes. Read both before running a batch if anything below is unclear; this skill is the operational half, they are the reasoning behind it.

## Shape of a batch

```
Phase 0  Setup      pin/verify the base, load or init batch state
Phase 1  Screen     for each idea: worktree -> worker -> reviewer -> score -> record -> cleanup
Phase 2  Combine    greedy forward selection over the keeps, re-measuring every step
Land     Merge survivors into worktree-ticket-<n>, push, resolve the ticket
```

Never more than **4 ideas × 2 attempts** in a batch. Check headroom **between** ideas, not mid-idea (see [Budget and graceful stop](#budget-and-graceful-stop)).

## Phase 0 — setup

1. **Pin the base.** Run `uv run python -m harness.score configs/main.toml`. If the report says the base is stale, re-pin it: `uv run python -m harness.score configs/main.toml --pin`. Record the resulting `cv`, `digest`, and `commit` as this batch's `base` — fixed for the whole batch, even if `configs/main.json` changes later (it won't: the work gate keeps a repo with an open landing ticket from claiming a new code-writing ticket, so no other batch can move `main` under this one).
2. **Load or init batch state** at `.batch/ticket-<n>.json` (gitignored; `<n>` is this ticket's number). If it exists and its `ticket` field matches, this is a **resume** — a prior instance of this same session was killed and relaunched by the dispatcher's crash retry or rate-limit resume. Skip re-sweeping anything already `screened` (any of `keep`/`drop`/`unmeasured`); pick up mid-idea or mid-phase from the state.

   ```jsonc
   {
     "ticket": 24,
     "started": "2026-09-27T12:00:00Z",
     "base": {"commit": "...", "cv": 0.80917, "digest": "..."},
     "ideas": {
       "toy_a": {"status": "pending", "attempts": []}
       // status: pending | screening | keep | near_miss | drop | unmeasured
       // keep: CV delta >= +0.002. near_miss: CV delta >= 0 but < +0.002. drop: CV delta < 0.
       // keep and near_miss are both phase-2-eligible; drop never is.
       // each attempt: {"worker_run": "...", "reviewer": "ok|bug|concern", "note": "...", "result_path": "..."}
     },
     "phase": "screen",           // screen | combine | done
     "combination": null          // filled once phase 2 runs
   }
   ```

   Write the state file after every meaningful step (idea claimed, attempt recorded, phase change) — it is the thing a resumed session reads, not memory.
3. **The ticket's theme** (its `## Question`, plus whatever the map's fog patch it graduated from said) yields the idea list, up to 4. Write them into `ideas` before starting phase 1. An idea is a short name plus a one-line description of what to try — the worker fills in the rest.

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
   - `ok` or `concern` → the score stands (a `concern` is recorded as a note, not a rerun). Apply the keep rule from `CONTEXT.md`: **keep** if CV improves on the pinned base by ≥ +0.002; **near_miss** if it improves (CV delta ≥ 0) but by less than +0.002; **drop** if CV delta < 0. All three are a completed screen — commit the worker's `components/`/`configs/` files onto the ticket branch (`git -C <ticket-worktree> checkout exp/<idea> -- components/ configs/exp/`, or cherry-pick the worker's commit if it made one), so the idea's code and its `results/<run>.json` both land in `worktree-ticket-<n>` regardless of label (a drop's file is what makes it re-verifiable rather than re-arguable). Only `keep` and `near_miss` carry into phase 2; a `drop`'s negative delta is the report, full stop.
   - A crash (worker or reviewer erroring, not a cited bug) also spends an attempt and retries once, same as a cited bug — but is **not** what "a bad score is a result, not a failure" is about: only a completed, reviewed run's score is final.
5. **Clean up**: `git worktree remove .claude/worktrees/exp-<idea>` and `git branch -D exp/<idea>` once its code is merged onto the ticket branch (or once it's abandoned unmeasured — nothing from a dead attempt is kept).
6. **Check budget** ([below](#budget-and-graceful-stop)) before starting the next idea.

Only one worker runs at a time to start. The loop above is written per-idea, not hardcoded to one — running N workers concurrently (still all sequential with the reviewer, since a review needs a finished diff) is a config change, not a rewrite, when that number becomes anything but 1.

## Phase 2 — combine

Skip this phase if zero ideas screened non-negative (no keeps and no near-misses) — go straight to landing with nothing new.

Greedy forward selection over every **keep and near-miss** — every idea that screened with CV delta ≥ 0, not only those that individually cleared +0.002. A `drop` (negative delta), like `family` at −0.00087, never enters: a near-miss earns a shot at combining precisely because it's non-negative, not because it's a keep.

1. Start from the pinned base config (`configs/main.toml`, i.e. nothing added yet).
2. For each not-yet-added keep or near-miss, build the config `base + that idea` and score it (`uv run python -m harness.score`, on a temp config extending main plus the trial addition — reuse of cached OOF/test predictions from phase 1 makes this cheap: only the *blend* changes, not any retraining, unless the combination changes a shared feature's output). Record every trial.
3. Add whichever trial scores highest, **but only if it still clears the current combination by ≥ +0.002** — gains do not add (the prior run's lgbm+catboost blends all scored below CatBoost alone), so re-measure, never sum deltas. This is the same +0.002 bar as the phase-1 keep label, but here it gates each step's combination, not which ideas were allowed to be tried.
4. Repeat from the new combined base until no remaining keep or near-miss improves it further, or every one of them has been tried.
5. The final combination (could be zero, one, or several ideas, keeps and near-misses alike) is `combination` in state. Write a combined config at `configs/exp/batch-<n>.toml` on the ticket branch listing exactly what it contains.

If an idea's addition to the combination is *worse* than it scored alone, that is exactly the data point the combined resolution comment must show — it is why phase 2 exists. A near-miss that turns out to combine well (e.g. `cabin_bins` or `spend_logs` alongside a `catboost` keep) is exactly the case this phase exists to catch.

## Landing

1. Merge the phase-2 combination onto `worktree-ticket-<n>` (it should already be there from phase 1's per-idea merges — phase 2 only adds the combined config file, no new component code).
2. Confirm no `exp/*` worktrees or branches remain.
3. If the landing session is this one (it usually is — see `README.md`'s and `CONTEXT.md`'s note that the holdout is read only at landing), run `uv run python -m harness.audit configs/exp/batch-<n>.toml` and quote its number in the resolution comment. **Do not** write it into `results/`, the config, or the map.
4. `git push -u origin worktree-ticket-<n>`.
5. Resolve the ticket per `docs/agents/issue-tracker.md`'s wayfinding operations: comment with the recap below, close the issue, and (this being a `wayfinder:experiment` ticket on the map, handled the same as any other resolution) append a one-line gist to the map's Decisions-so-far pointing at the ticket.

**The resolution comment** is what the user actually reads at landing. It must contain, and must not contain anything it can't support:

- The pinned base: commit and CV.
- A delta table, one row per idea in the batch: name, status (keep / near_miss / drop / unmeasured), CV delta (omit for unmeasured), reviewer verdict and any note, attempts spent.
- The phase-2 combination **actually measured** — never a combined gain inferred by adding individual deltas.
- The holdout reading, if fetched this session, labelled as a smoke-test number, not a keep input.
- Any ideas from the ticket's theme that were never run (budget or usage cut the batch short) — these go back into the map's fog (**Not yet specified**), not silently dropped.

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
- **A bad score is a result, not a failure.** Negative CV delta ⇒ drop, non-negative-but-below-threshold ⇒ near_miss (still phase-2-eligible) — either way recorded with its number, no retry. Retries exist only for a cited reviewer bug or a crash.
- **No shared append log.** Every run writes its own `results/<run-id>.json` (the harness already does this); never edit another run's file, never maintain a batch-wide log outside `.batch/ticket-<n>.json`, which is state, not the record.
- **Budget is a hard cap**, not a target: ≤ 4 ideas, ≤ 2 attempts each.
- **State survives a kill.** Write `.batch/ticket-<n>.json` after every step that would be expensive to redo. A resumed session trusts it over re-deriving anything.
- **A stopped batch still lands what it has.** Losing headroom mid-batch is not an error path to apologize for: abandon the in-flight worktree, run phase 2 over whatever survived (even zero), merge, push, resolve with the delta table plus an explicit list of unrun ideas for the fog. A batch that finishes one of four ideas is a small success.
- **Clean up `exp/*` worktrees and branches** before finishing; only the ticket branch leaves this session.

## Budget and graceful stop

The hard limit is the idea/attempt cap above — that alone bounds a batch. Alongside it, stop starting a **new** idea once this session's headroom looks thin: if your environment surfaces a usage or context indicator, treat crossing roughly 70% as the stop signal (below the dispatcher's 80% claim guard, so a batch has room to land tidily rather than being cut off mid-merge); if no such signal is available in-session, treat the idea cap itself as the operative limit and prefer stopping an idea early over risking a mid-merge cutoff. Either way, the response to hitting the limit is the same: stop starting new ideas, finish or abandon whatever's in flight, and go straight to [Landing](#landing) with what's screened.

## Dispatched sessions

This ticket is claimed and worked by a dispatched session exactly as `wayfinder`'s own "Dispatched sessions" section describes — read it before starting if you haven't this session. In particular: resolving the ticket sends a one-line gist with `PushNotification` (this is an AFK ticket type; nobody is watching a terminal), and if the batch hits a decision only a human can make (the idea list itself is ambiguous, the keep rule doesn't cover a tie, a worker's diff can't be reviewed because the contract itself is unclear), stop rather than guess: comment what it's blocked on, add `hitl` and `ready`, unassign, notify, end the turn.
