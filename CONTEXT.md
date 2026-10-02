# Spaceship Titanic

Predict which passengers of the Spaceship Titanic were transported to another dimension.

## Language

### The data

**Passenger**:
One row of the competition data, identified by `PassengerId` (`gggg_pp`).

**Travel group**:
Passengers sharing the `gggg` prefix of their PassengerId; often family, not always. No travel group appears in both the training and the test data.
_Avoid_: family (a family is inferred from surname and may span groups)

**Cabin**:
`Deck/Num/Side`: deck letter, cabin number, and side (P = Port, S = Starboard).

**Spend categories**:
The five onboard amenity bills: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck.

**Holdout**:
The ~15% of training passengers, whole travel groups at a time, set aside once and never used to train, score or choose anything. Read only at landing, as a smoke detector for gross divergence, never as a keep input.
_Avoid_: validation set, test set

**Dev rows**:
The training passengers outside the holdout; the only rows a CV score is computed over.

### Research

**Harness**:
The frozen code that loads the data, carves the holdout, builds the folds and produces every score. Research adds to what it runs; it never edits it.

**Component**:
One named, add-only unit the harness can switch on: a feature step or a model. Off unless a config turns it on.
_Avoid_: plugin, module

**Config**:
The list of components a pipeline switches on, with each model's blend weight and any parameter overrides.

**Run**:
One scoring of one config by the harness, recorded as its own result.
_Avoid_: experiment (that is now a batch)

**Pinned base**:
The config on `main` together with its recorded CV score; every run in a batch is compared against this one fixed number.
_Avoid_: baseline (that is the original LightGBM)

**Batch**:
One experiment ticket's work: up to four ideas screened singly against the pinned base, every allowed combination of them measured in a matrix, the best cell confirmed, every model in the result tuned, and the outcome landed once.
_Avoid_: experiment ticket per idea

**Matrix**:
Every on/off combination of a batch's ideas (2^k cells for k ideas, the all-off cell being the pinned base), so ideas that only help together are measured together. Cells combining ideas declared mutually exclusive are skipped.
_Avoid_: forward selection, greedy combination (the matrix replaced it)

**Confirmation seeds**:
Seeds never used for screening or tuning, on which the matrix's best cell and the pinned base are both re-scored; the best cell's label comes from this re-scored delta, because picking the top of many noisy scores inflates it.

**Experiment**:
A batch. It no longer means a single change.

**CV score**:
Mean accuracy over 5-fold stratified CV repeated with 3 seeds, over the dev rows only. The score that labels every change; only the leaderboard gate can overrule it.

**Blend**:
A weighted average of predicted probabilities from several models.

**Decision threshold**:
The probability above which a passenger is predicted Transported. The harness always cuts at 0.5; a tuned threshold is an idea like any other, learned only from the training fold's own inner folds and expressed by a component shifting its model's probabilities so that 0.5 falls where the tuned cutoff would.
_Avoid_: cutoff tuned on out-of-fold predictions (that leaks into the CV score)

**Keep / near-miss / drop**:
The label a scored change gets against the pinned base, same folds, same seeds: **keep** if CV improves by at least +0.002; **near-miss** if it improves by less; **drop** if it doesn't improve at all. A single screen's label is only a report; the label that decides a batch is the best matrix cell's, on the confirmation seeds.

**Leaderboard gate**:
The public leaderboard's say over a batch's final result, compared against the pinned base's own public score: a keep is dropped if it scores more than 0.010 below the base; a near-miss is promoted only if it scores more than 0.010 above. A drop is never submitted.
_Avoid_: veto (the leaderboard now promotes as well as blocks)

**Final submission**:
The config named as this effort's answer when the user stops: the highest public leaderboard score among configs that were ever promoted to pinned base. It need not be the current pinned base.

**Explanation**:
What a config's models rely on: how much worse their predictions get on the validation folds when one component's columns are shuffled. Descriptive only, like the holdout reading, never a keep input.
_Avoid_: feature importance (model-specific built-ins, which differ per model)

**Unmeasured**:
An idea abandoned because its runs were invalid, so it has no score; distinct from a drop.

**Tuning idea**:
New parameter values for one model already in the config, found by a search on fold seeds the CV score never uses. Every batch ends by tuning each model of its result in turn. Because the search saw the same dev rows, its keep label needs both the +0.002 improvement and a majority of better folds; short of that but non-negative, it is a near-miss.
_Avoid_: tuning experiment, tuning ticket
