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
         reason to hold any of it out.
  test   Fits on 80% of matches and evaluates on the held-out 20%, to
         estimate how well the model generalizes to matches it hasn't
         seen. The split is grouped by match_id, not a random row split:
         rounds from the same match are correlated (same two teams, same
         map), so a row-level split would leak match-specific patterns
         across train/test and inflate the apparent score.

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
RANDOM_STATE = 42


def _load_data():
    df = pd.read_csv(ROUND_DATA_CSV)
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    return df[feature_cols], df[TARGET_COL], df[ID_COL]


def _make_model():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def run_test(X, y, groups):
    n_matches = groups.nunique()
    if n_matches < 2:
        raise SystemExit(
            f"Only {n_matches} match(es) in {ROUND_DATA_CSV} -- need at least 2 to hold "
            f"one out for a match-grouped test split. Parse more demos first."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train)} rounds from {groups.iloc[train_idx].nunique()} match(es)")
    print(f"Test:  {len(X_test)} rounds from {groups.iloc[test_idx].nunique()} match(es)")

    model = _make_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print(f"\nAccuracy:    {accuracy_score(y_test, preds):.4f}")
    print(f"Log loss:    {log_loss(y_test, proba, labels=[0, 1]):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, proba):.4f}")
    if y_test.nunique() > 1:
        print(f"ROC AUC:     {roc_auc_score(y_test, proba):.4f}")

    majority_baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"\n(Majority-class baseline accuracy for reference: {majority_baseline:.4f})")


def run_train(X, y):
    model = _make_model()
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
        run_train(X, y)


if __name__ == "__main__":
    main()
