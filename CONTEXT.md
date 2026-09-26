# Spaceship Titanic

Predict which passengers of the Spaceship Titanic were transported to another dimension.

## Language

**Passenger**:
One row of the competition data, identified by `PassengerId` (`gggg_pp`).

**Travel group**:
Passengers sharing the `gggg` prefix of their PassengerId; often family, not always.
_Avoid_: family (a family is inferred from surname and may span groups)

**Cabin**:
`Deck/Num/Side`: deck letter, cabin number, and side (P = Port, S = Starboard).

**Spend categories**:
The five onboard amenity bills: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck.

**Experiment**:
One evaluated combination of feature set, model(s) and blend, scored by CV and compared against the pipeline on `main`.

**CV score**:
Mean accuracy over 5-fold stratified CV repeated with 3 seeds. The only score that drives keep/drop decisions.

**Blend**:
A weighted average of predicted probabilities from several models.

**Keep / drop**:
The decision an experiment ends in: keep if CV score improves by at least +0.002 over `main`, else drop.
