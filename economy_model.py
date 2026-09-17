"""
Trains an XGBoost regressor that predicts a team's equipment value at the
start of the *next* round from its economy state entering the round that
just ended, plus round_model's predicted P(CT wins) for that round.

Why round_model's *prediction* and not the round's true winner: at the
point this model is meant to be used (chaining round-to-round forecasts
into a match simulation), the true winner of the round being forecast
isn't known yet -- only round_model's win probability is available.
Training on the true winner would fit a relationship ("what actually
happened last round predicts next round's buy") that doesn't hold at
inference time, where "what actually happened" has to be replaced by
"what round_model thinks will happen." Training on round_model's own
output keeps train and inference consistent.

The model is side-symmetric: rather than fitting separate "predict CT's
next equip" and "predict T's next equip" models, every round_data.csv row
is expanded into two perspective rows -- one from the CT side's point of
view, one from T's -- with columns renamed to own_equip_value /
opp_equip_value / own_loss_bonus_streak / own_win_prob, and a single
model is fit on the result. This doubles the training data for free
(same trick round_model.py's docstring notes this project can't afford
to skip, given how few matches there are) and encodes, structurally, the
fact that "next round's equip" should depend on being the own side vs.
the opponent, not on being labeled CT vs. T.

Excluded from training: the transition into a halftime pistol round.
Two things happen simultaneously at that boundary that have nothing to
do with economy modeling: (1) equipment resets to a fixed low pistol-round
value regardless of what either team saved, and (2) the CT/T labels swap
teams, so naively pairing "this row's ct_equip_value" with "the next
row's ct_equip_value" would actually be pairing two different teams'
economies. This was confirmed empirically by walking round_data.csv:
every row where ct_score + t_score == 12 (12 rounds already decided) has
ct_equip_value/t_equip_value in the ~$3-5K pistol range no matter how
rich either side was the round before, and the score columns show the
CT/T win counters swapping which team they track at exactly that row.
Overtime introduces its own resets (loss-bonus streaks reset to 1/1 at
the start of each 6-round OT period, at ct_score+t_score in
{24, 30, 36, ...}) but does NOT reset equipment or swap sides mid-period,
so those transitions are left in; the loss-bonus streak feature already
reflects the reset value at that row (it's the real pre-round streak, not
something this model invents), and the swap only ever happens mid-period,
3 rounds in, not at the period boundary itself.

Two modes:
  train  Fits on the entire dataset and saves the model to
         data/economy_model.json.
  test   Fits on 80% of matches and evaluates MAE/RMSE on the held-out
         20%, match-grouped the same way round_model.py's test mode is
         (see its docstring for why a row-level split would leak).

Also usable as a library: predict_next_equip_values() takes a round's
pre-round state plus a CT-win probability (e.g. from round_model) and
returns the model's forecast of both sides' next-round equip values --
this is the piece a round-to-round match simulation would call each
simulated round.

Usage:
    python economy_model.py train
    python economy_model.py test
    python economy_model.py predict --ct-equip 4200 --t-equip 4200 \
        --ct-streak 1 --t-streak 1 --ct-win-prob 0.55
"""
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

import round_model

DATA_DIR = "data"
ROUND_DATA_CSV = os.path.join(DATA_DIR, "round_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "economy_model.json")

ID_COL = "match_id"
FEATURE_COLS = ["own_equip_value", "opp_equip_value", "own_loss_bonus_streak", "own_win_prob"]
TARGET_COL = "next_own_equip_value"

# Rounds already decided (ct_score + t_score) at which the *next* row is a
# halftime pistol round -- see module docstring for how this was found and
# why those transitions are excluded from training.
HALFTIME_ROUNDS_PLAYED = 12

# Loose sanity ceiling on predicted equip value -- roughly a full 5-player
# max buy (rifle + armor + nades + a couple of utility items each), and
# above the highest value seen in round_data.csv. Only guards against a
# regressor extrapolating wildly on an out-of-distribution input; it is not
# a modeled constraint.
EQUIP_VALUE_MAX = 40000

TEST_SIZE = 0.2
VAL_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 50
# Picked by a max_depth x learning_rate grid search (2-6 x 0.01-0.1),
# averaged over 5 match-grouped train/test splits -- same protocol as
# round_model.py's search, but the result runs the other way: MAE dropped
# steadily from depth 2 to depth 4 (mean ~5232 -> ~5130) instead of
# favoring the shallowest tree, then flattened through depth 6 (~5121-5130,
# a noise-level spread). Four features evidently support more splits per
# tree here than round_model.py's two before a leaf starts fitting
# per-match noise. learning_rate moved the result about as little as it
# did for round_model.py; 0.03 was consistently near the best of the range
# tried at every depth.
MAX_DEPTH = 4
LEARNING_RATE = 0.03
RANDOM_STATE = 42


def _load_round_model():
    if not os.path.exists(round_model.MODEL_PATH):
        raise SystemExit(
            f"{round_model.MODEL_PATH} not found -- run `python round_model.py train` first."
        )
    model = xgb.XGBClassifier()
    model.load_model(round_model.MODEL_PATH)
    return model


def _perspective_frame(df, next_df, own_prefix, opp_prefix, own_win_prob):
    frame = pd.DataFrame({
        "own_equip_value": df[f"{own_prefix}_equip_value"],
        "opp_equip_value": df[f"{opp_prefix}_equip_value"],
        "own_loss_bonus_streak": df[f"{own_prefix}_loss_bonus_streak"],
        "own_win_prob": own_win_prob,
        TARGET_COL: next_df[f"{own_prefix}_equip_value"],
        ID_COL: df[ID_COL],
    })
    # Rows with no next round (end of match) or whose next round is a
    # halftime reset -- see module docstring.
    rounds_played_next = next_df["ct_score"] + next_df["t_score"]
    valid = next_df[f"{own_prefix}_equip_value"].notna() & (rounds_played_next != HALFTIME_ROUNDS_PLAYED)
    return frame[valid]


def _load_data(round_clf=None):
    """round_clf lets a caller (match_model.py's test mode) supply an
    already-fitted round-winner classifier instead of loading
    round_model.json from disk -- needed so its match-grouped test split
    can fit round_model on the train matches only and feed that (not the
    full-data model, which would leak the held-out matches' own outcomes
    into this model's ct_win_prob feature) into economy_model's own
    training data."""
    df = pd.read_csv(ROUND_DATA_CSV)
    rmodel = round_clf if round_clf is not None else _load_round_model()
    ct_win_prob = rmodel.predict_proba(df[round_model.FEATURE_COLS])[:, 1]

    next_df = df.groupby(ID_COL).shift(-1)
    ct_perspective = _perspective_frame(df, next_df, "ct", "t", ct_win_prob)
    t_perspective = _perspective_frame(df, next_df, "t", "ct", 1 - ct_win_prob)

    combined = pd.concat([ct_perspective, t_perspective], ignore_index=True)
    return combined[FEATURE_COLS], combined[TARGET_COL], combined[ID_COL]


def _require_multiple_groups(groups, purpose):
    n = groups.nunique()
    if n < 2:
        raise SystemExit(
            f"Only {n} match(es) available for {purpose} -- need at least 2 "
            f"(one to train on, one to hold out). Parse more demos first."
        )


def _make_model(n_estimators, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE, early_stopping_rounds=None):
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=10,
        eval_metric="mae",
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
    )


def _pick_n_estimators(X, y, groups, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE):
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

    print(f"Train: {len(X_train)} perspective-rows from {groups_train.nunique()} match(es)")
    print(f"Test:  {len(X_test)} perspective-rows from {groups.iloc[test_idx].nunique()} match(es)")

    n_estimators = _pick_n_estimators(X_train, y_train, groups_train)
    print(f"Early stopping picked {n_estimators} trees.")

    model = _make_model(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # Naive baseline: assume next round's equip equals this round's --
    # the same role round_model.py's majority-class baseline plays, so the
    # model's error is judged against "predicting nothing changed" rather
    # than against zero.
    baseline_preds = X_test["own_equip_value"]
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = mean_squared_error(y_test, baseline_preds) ** 0.5

    print(f"\nMAE:  {mae:.0f}  (no-change baseline: {baseline_mae:.0f})")
    print(f"RMSE: {rmse:.0f}  (no-change baseline: {baseline_rmse:.0f})")


def run_train(X, y, groups):
    n_estimators = _pick_n_estimators(X, y, groups)
    print(f"Early stopping picked {n_estimators} trees.")

    model = _make_model(n_estimators=n_estimators)
    model.fit(X, y)

    os.makedirs(DATA_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Trained on all {len(X)} perspective-row(s). Saved model to {MODEL_PATH}.")


def _load_economy_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"{MODEL_PATH} not found -- run `python economy_model.py train` first.")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def predict_next_equip_values(model, ct_equip_value, t_equip_value, ct_loss_bonus_streak,
                               t_loss_bonus_streak, ct_win_prob):
    """Given a round's pre-round state and round_model's P(CT wins) for
    that round, forecasts both sides' equip value entering the *next*
    round. Does not know about halftime/OT resets -- those are rule-based,
    not economic, and are the caller's responsibility (see module
    docstring) to apply instead of calling this."""
    rows = pd.DataFrame([
        {
            "own_equip_value": ct_equip_value,
            "opp_equip_value": t_equip_value,
            "own_loss_bonus_streak": ct_loss_bonus_streak,
            "own_win_prob": ct_win_prob,
        },
        {
            "own_equip_value": t_equip_value,
            "opp_equip_value": ct_equip_value,
            "own_loss_bonus_streak": t_loss_bonus_streak,
            "own_win_prob": 1 - ct_win_prob,
        },
    ])[FEATURE_COLS]
    preds = np.clip(model.predict(rows), 0, EQUIP_VALUE_MAX)
    return float(preds[0]), float(preds[1])


def run_predict(args):
    model = _load_economy_model()
    next_ct, next_t = predict_next_equip_values(
        model, args.ct_equip, args.t_equip, args.ct_streak, args.t_streak, args.ct_win_prob,
    )
    print(f"Predicted next round: CT equip = {next_ct:.0f}, T equip = {next_t:.0f}")


def main():
    argp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = argp.add_subparsers(dest="mode", required=True)
    sub.add_parser("train", help="Fit on all data and save the model.")
    sub.add_parser("test", help="Evaluate on a held-out 20%% of matches.")

    predict_argp = sub.add_parser("predict", help="Forecast next-round equip values for one round state.")
    predict_argp.add_argument("--ct-equip", type=float, required=True)
    predict_argp.add_argument("--t-equip", type=float, required=True)
    predict_argp.add_argument("--ct-streak", type=float, required=True, help="ct_loss_bonus_streak entering this round")
    predict_argp.add_argument("--t-streak", type=float, required=True, help="t_loss_bonus_streak entering this round")
    predict_argp.add_argument("--ct-win-prob", type=float, required=True, help="round_model's P(CT wins) for this round")

    args = argp.parse_args()

    if args.mode == "predict":
        run_predict(args)
        return

    if not os.path.exists(ROUND_DATA_CSV):
        raise SystemExit(f"{ROUND_DATA_CSV} not found -- run parse_demos.py first.")

    X, y, groups = _load_data()

    if args.mode == "test":
        run_test(X, y, groups)
    else:
        run_train(X, y, groups)


if __name__ == "__main__":
    main()
