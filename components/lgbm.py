"""LightGBM with a fixed number of rounds: nothing is chosen on the scored fold."""
import lightgbm as lgb

NAME = "lgbm"
KIND = "model"
PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def build(params, seed):
    return lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
