"""Baseline: LightGBM with 5-fold stratified CV, writes a submission."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.config import ID, RAW, SEED, SUBMISSIONS, TARGET
from src.features import build_features


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")

    # Build features on train+test together so group sizes / categories are consistent
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    feats = build_features(full)
    X, X_test = feats.iloc[: len(train)], feats.iloc[len(train):]
    y = train[TARGET].astype(int).values

    params = dict(n_estimators=1000, learning_rate=0.02, num_leaves=31,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, verbose=-1)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y[tr], eval_X=X.iloc[va], eval_y=y[va],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
        print(f"fold {fold}: acc={accuracy_score(y[va], oof[va] > 0.5):.4f}")

    print(f"CV accuracy: {accuracy_score(y, oof > 0.5):.4f}")

    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / "lgbm_baseline.csv"
    pd.DataFrame({ID: test[ID], TARGET: test_pred > 0.5}).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
