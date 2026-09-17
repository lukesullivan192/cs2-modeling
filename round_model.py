"""
Trains an XGBoost classifier on data/round_data.csv to predict a round's
winner (winner_is_ct) from its pre-round state -- see parse_demos.py's
module docstring for how round_data.csv is built. This is the round-level
win-probability model meant to feed a match-level Monte Carlo/DP
simulation, not a match-outcome model itself.

Two modes:
  train  Fits on the entire dataset and saves the model to
         data/round_model.json. Use this once you're done evaluating, to
         get the strongest model for actual simulation use -- every real
         round outcome is useful signal at that point, so there's no
         reason to hold any of it out. The number of trees is still chosen
         by early stopping against a held-out slice of matches (see
         `_pick_n_estimators`), then the final model is refit on 100% of
         the data with that fixed tree count.
  test   Fits on 80% of matches and evaluates on the held-out 20%, to
         estimate how well the model generalizes to matches it hasn't
         seen. The split is grouped by match_id, not a random row split:
         rounds from the same match are correlated (same two teams, same
         map), so a row-level split would leak match-specific patterns
         across train/test and inflate the apparent score.

At ~100 matches, this problem is starved for data relative to how much
signal a plain "throw every raw column at a deep XGBoost" setup needs --
the fit will overfit to per-match noise well before it exhausts the real
economic signal in the data (e.g. ct_equip_value - t_equip_value alone is
a strong predictor). Two things counter that here, and both were verified
to help with 5-fold grouped cross-validation, not just eyeballed on one
split:
  - equip_diff / score_diff / loss_bonus_diff are added as explicit
    features. XGBoost can in principle learn "the difference between these
    two columns matters" on its own, but doing that from raw
    ct_equip_value/t_equip_value costs splits that this little data can't
    spare -- handing it the difference directly is a large, free win here.
  - Regularization (shallow trees, min_child_weight, reg_lambda) plus
    early stopping against a held-out slice of matches, instead of a fixed
    n_estimators picked without looking at validation performance.

Usage:
    python round_model.py train
    python round_model.py test
"""
import argparse
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = "data"
ROUND_DATA_CSV = os.path.join(DATA_DIR, "round_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "round_model.json")

TARGET_COL = "winner_is_ct"
ID_COL = "match_id"  # identifier for grouped splitting, not a feature -- see parse_demos.py

TEST_SIZE = 0.2
# Fraction of the *training* matches held out to pick the number of trees
# via early stopping (both in test mode and, from the full dataset, in
# train mode). Separate from TEST_SIZE, which is reserved purely for
# reporting how the model generalizes and must never be touched during
# fitting.
VAL_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42


def _add_engineered_features(df):
    df = df.copy()
    df["equip_diff"] = df["ct_equip_value"] - df["t_equip_value"]
    df["score_diff"] = df["ct_score"] - df["t_score"]
    df["loss_bonus_diff"] = df["ct_loss_bonus_streak"] - df["t_loss_bonus_streak"]
    return df


def _load_data():
    df = _add_engineered_features(pd.read_csv(ROUND_DATA_CSV))
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    return df[feature_cols], df[TARGET_COL], df[ID_COL]


def _require_multiple_groups(groups, purpose):
    n = groups.nunique()
    if n < 2:
        raise SystemExit(
            f"Only {n} match(es) available for {purpose} -- need at least 2 "
            f"(one to train on, one to hold out). Parse more demos first."
        )


def _make_model(n_estimators, early_stopping_rounds=None):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=10,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
    )


def _pick_n_estimators(X, y, groups):
    """Fits with early stopping against a held-out slice of matches carved
    out of (X, y, groups), and returns the resulting best tree count. The
    caller is responsible for then fitting the model it actually keeps."""
    _require_multiple_groups(groups, "early-stopping validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(X, y, groups))

    model = _make_model(n_estimators=2000, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X.iloc[fit_idx], y.iloc[fit_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=False,
    )
    return model.best_iteration + 1


def run_test(X, y, groups):
    _require_multiple_groups(groups, "a match-grouped test split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(f"Train: {len(X_train)} rounds from {groups_train.nunique()} match(es)")
    print(f"Test:  {len(X_test)} rounds from {groups.iloc[test_idx].nunique()} match(es)")

    n_estimators = _pick_n_estimators(X_train, y_train, groups_train)
    print(f"Early stopping picked {n_estimators} trees.")

    model = _make_model(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"\nAccuracy:    {accuracy_score(y_test, preds):.4f}")
    print(f"Log loss:    {log_loss(y_test, proba, labels=[0, 1]):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, proba):.4f}")
    if y_test.nunique() > 1:
        print(f"ROC AUC:     {roc_auc_score(y_test, proba):.4f}")

    majority_baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\n(Majority-class baseline accuracy: {majority_baseline:.4f}; "
          f"always-50% log loss: {log_loss([0, 1], [0.5, 0.5]):.4f})")


def run_train(X, y, groups):
    n_estimators = _pick_n_estimators(X, y, groups)
    print(f"Early stopping picked {n_estimators} trees.")

    model = _make_model(n_estimators=n_estimators)
    model.fit(X, y)

    os.makedirs(DATA_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Trained on all {len(X)} round(s). Saved model to {MODEL_PATH}.")


def main():
    argp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argp.add_argument(
        "mode", choices=["train", "test"],
        help="'train' fits on all data and saves a model; 'test' evaluates on a held-out 20%% of matches.",
    )
    args = argp.parse_args()

    if not os.path.exists(ROUND_DATA_CSV):
        raise SystemExit(f"{ROUND_DATA_CSV} not found -- run parse_demos.py first.")

    X, y, groups = _load_data()

    if args.mode == "test":
        run_test(X, y, groups)
    else:
        run_train(X, y, groups)


if __name__ == "__main__":
    main()
