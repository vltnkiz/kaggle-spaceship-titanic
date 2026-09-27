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
One experiment ticket's work: several ideas screened against the pinned base, then the survivors combined and re-measured, landing once.
_Avoid_: experiment ticket per idea

**Experiment**:
A batch. It no longer means a single change.

**CV score**:
Mean accuracy over 5-fold stratified CV repeated with 3 seeds, over the dev rows only. The only score that drives keep/drop decisions.

**Blend**:
A weighted average of predicted probabilities from several models.

**Keep / near-miss / drop**:
Phase 1's report label for a completed screen, same folds and same seeds: **keep** if CV improves on the pinned base by at least +0.002; **near-miss** if it improves but by less than +0.002; **drop** if it doesn't improve at all. The +0.002 threshold is this label's cutoff only — it is *not* the gate for entering phase 2's forward selection, which draws from every non-negative screen (keeps and near-misses alike). The same +0.002 bar reappears inside phase 2, but there it's gating each forward-selection step's combined score against the current combination, not gating which ideas are eligible to be tried.

**Unmeasured**:
An idea abandoned because its runs were invalid, so it has no score; distinct from a drop.
