"""
Trains an XGBoost classifier on data/round_data.csv to predict a round's
winner (winner_is_ct) from its pre-round state -- see parse_demos.py's
module docstring for how round_data.csv is built. This is the round-level
win-probability model meant to feed a match-level Monte Carlo/DP
simulation, not a match-outcome model itself.

The model trains on ct_equip_value and t_equip_value alone (see
FEATURE_COLS). A permutation-importance test across 10 match-grouped
train/test splits found the equip values (whether given raw or as their
CT-minus-T difference) were the only columns with a consistent,
non-noise-level effect on round-winner predictions -- score, loss-bonus
streaks, helmets, defusers, and map all came back indistinguishable from
zero (some, like ct_loss_bonus_streak, were mildly *negative* across every
split, i.e. the model did marginally better without them). round_data.csv
still carries ct_score/t_score and the loss-bonus streaks alongside the
equip values, but those are there for a planned round-to-round economy
transition model, not as features for this one -- see parse_demos.py's
module docstring.

Two modes:
  train  Fits on the entire dataset and saves the model to
         data/round_model.json.
  test   Fits on 80% of matches and evaluates on the held-out 20%, to
         estimate how well the model generalizes to matches it hasn't
         seen. The split is grouped by match_id, not a random row split:
         rounds from the same match are correlated (same two teams, same
         map), so a row-level split would leak match-specific patterns
         across train/test and inflate the apparent score.

At ~100 matches, this problem is starved for data relative to how much
signal a plain "throw defaults at a deep XGBoost" setup needs -- the fit
overfits to per-match noise well before it exhausts the real economic
signal in ct_equip_value/t_equip_value. Regularization (shallow trees,
min_child_weight, reg_lambda, row/column subsampling) plus early stopping
against a held-out slice of matches (instead of a fixed n_estimators
picked without looking at validation performance) both counter that, as
does MONOTONE_CONSTRAINTS -- see its comment above FEATURE_COLS.

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
# The only features this model uses -- see module docstring for why the
# other round_data.csv columns (score, loss-bonus streaks) are excluded.
FEATURE_COLS = ["ct_equip_value", "t_equip_value"]
# Win probability must be non-decreasing in ct_equip_value and
# non-increasing in t_equip_value -- that's known a priori, not something
# ~2K rows should have to (re)learn. Enforcing it as a hard constraint
# during split-finding (rather than hoping the data implies it) acts as
# extra regularization on top of the depth/subsample/reg_lambda settings
# below, and measurably improved log loss/Brier/AUC across 10
# match-grouped test splits versus the unconstrained model (10/10 splits
# favored the constrained model, mean log loss 0.604 -> 0.600).
MONOTONE_CONSTRAINTS = (1, -1)  # order matches FEATURE_COLS

TEST_SIZE = 0.2
# Fraction of the *training* matches held out to pick the number of trees
# via early stopping (both in test mode and, from the full dataset, in
# train mode). Separate from TEST_SIZE, which is reserved purely for
# reporting how the model generalizes and must never be touched during
# fitting.
VAL_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 50
# Picked by a max_depth x learning_rate grid search (2-6 x 0.01-0.1),
# averaged over 5 match-grouped train/test splits so the pick isn't just
# fitting one split's noise. max_depth=2 beat every deeper value at every
# learning rate tried (mean log loss ~0.614 vs ~0.622+ for depth 3), which
# tracks: two raw features can't support many splits per tree before a
# leaf is fitting per-match noise instead of signal. learning_rate barely
# moved the result at max_depth=2 (0.614-0.615 across the whole range) --
# 0.03 edged out the rest but the difference is noise-level.
MAX_DEPTH = 2
LEARNING_RATE = 0.03
RANDOM_STATE = 42


def _load_data():
    df = pd.read_csv(ROUND_DATA_CSV)
    return df[FEATURE_COLS], df[TARGET_COL], df[ID_COL]


def _require_multiple_groups(groups, purpose):
    n = groups.nunique()
    if n < 2:
        raise SystemExit(
            f"Only {n} match(es) available for {purpose} -- need at least 2 "
            f"(one to train on, one to hold out). Parse more demos first."
        )


def _make_model(n_estimators, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE, early_stopping_rounds=None):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=10,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
    )


def _pick_n_estimators(X, y, groups, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE):
    """Fits with early stopping against a held-out slice of matches carved
    out of (X, y, groups), and returns the resulting best tree count. The
    caller is responsible for then fitting the model it actually keeps."""
    _require_multiple_groups(groups, "early-stopping validation")
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(X, y, groups))

    model = _make_model(
        n_estimators=2000, max_depth=max_depth, learning_rate=learning_rate,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
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
