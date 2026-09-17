"""
Chains round_model.py (P(CT wins) the upcoming round from its equip
values) and economy_model.py (each side's forecast equip value entering
the *next* round) into a full match-outcome dynamic program: starting
from any round's pre-round state, it branches on who wins that round,
deterministically fills in the rest of the resulting next-round state
(score, loss-bonus streaks -- and, at halftime/overtime boundaries,
equipment) per CS2's actual rules, and recurses to compute P(CT wins the
match) from that point onward.

CS2 rules encoded here (see _match_winner / _advance_state), pinned down
empirically against round_data.csv rather than assumed:
  - Regulation: first to 13 wins; 24 rounds max (MR12).
  - Halftime (entering round 13, i.e. 12 rounds already decided): the
    CT/T labels swap teams, so ct_score/t_score and the loss-bonus
    streaks *swap* rather than reset (a team's own tally carries over
    under its new side label); equipment genuinely does reset, to a
    fixed low pistol-round value. That value was measured, not guessed:
    for each match, the single biggest round-to-round drop in
    (ct_equip_value + t_equip_value) reliably identifies its real
    halftime transition (some demos show it landing one round later
    than a naive "totals==12" check would assume, apparently an
    engine/plugin timing quirk in how some servers fire the restart
    relative to the freezetime-end tick we sample -- checking both
    candidates per match and taking whichever shows the bigger drop
    sidesteps that instead of guessing). The resulting ~99-match sample
    gives a tight ~$4250 (CT) / ~$4350 (T) reset value.
  - Overtime: triggered at 12-12. Periods of 6 rounds starting every 6
    rounds after that (24, 30, 36, ... rounds decided); the first side to
    win 4 rounds *within* a period wins the match, and a 3-3 period
    starts another one. Loss-bonus streaks reset to 1/1 at the start of
    each period (confirmed in economy_model.py's docstring); the sides
    swap again at each period's midpoint (3 rounds in), the same
    label-swap (not reset) treatment as halftime. Equipment is NOT
    force-reset in OT -- economy_model was deliberately trained on those
    transitions (see its docstring), so its normal learned prediction
    already covers them.

Both branches of a round (CT wins / T wins) reuse the SAME economy_model
forecast for the next round's equip values, rather than one forecast per
branch. This isn't a simplification made here -- it's how economy_model
was built: it's trained on round_model's *predicted* win probability, not
the round's true winner, specifically so it can be called once per round
with "what round_model thinks will happen" and produce one well-calibrated
forecast, rather than needing to know the outcome it's forecasting past.

Because the DP only ever moves forward in total rounds decided, it can't
cycle, but nothing stops it from continuing through many 3-3 overtime
periods in principle -- vanishingly likely at any given period, but not
impossible, so a hard cap (MAX_OT_PERIODS) forces a resolution once
continuing further couldn't matter.

MatchSimulator computes this as a *forward* probability propagation
(a probability distribution over states, advanced one round at a time),
not a top-down recursion -- a naive "recurse into both branches per
round, one model call per node" version means up to ~2^24 tree nodes for
regulation alone, each paying full Python/pandas/XGBoost call overhead,
and was confirmed intractable (killed after 5+ minutes with zero output
on real data). The forward version batches every currently-active state's
round_model and economy_model calls into one vectorized prediction per
round, splits each state's probability mass into its win/lose branches,
and re-aggregates by a lightly-quantized state (see EQUIP_ROUND_TO) so
branches that land on essentially the same state merge their mass instead
of both continuing to branch separately. A branch that lands on a decided
match has its mass folded into the running total immediately rather than
propagated further, so the active distribution is guaranteed to empty out
within a bounded number of rounds.

Two modes:
  train  Ensures round_model.json and economy_model.json exist by
         training both on the full dataset (delegating to
         round_model.run_train / economy_model.run_train) -- the DP
         itself has no parameters of its own to fit; it's a fixed
         function of those two models plus CS2's deterministic rules.
  test   Uses the same match-grouped 80/20 split round_model.py and
         economy_model.py use (same TEST_SIZE/RANDOM_STATE), but refits
         both models on the 80% train matches only -- reusing the
         full-data models here would leak each held-out match's own
         outcome into the very predictions being evaluated on it. Then,
         for every real round of every held-out match, runs the DP from
         that round's *actual* historical pre-round state to get P(CT
         wins the match), compares the implied favorite against that
         match's real final winner, and reports accuracy broken down by
         round number, plus one overall average pooling every (match,
         round) pair in the test set. Also reports a naive "whoever's
         currently leading on the scoreboard" baseline per round, for
         context -- the same role every other model file's baseline
         plays.

Usage:
    python match_model.py train
    python match_model.py test
"""
import argparse
import os
from collections import defaultdict, namedtuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import economy_model
import round_model

DATA_DIR = "data"
ROUND_DATA_CSV = os.path.join(DATA_DIR, "round_data.csv")
ID_COL = "match_id"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# CS2 match-format constants -- see module docstring.
REGULATION_WIN = 13
REGULATION_ROUNDS = 24
OT_PERIOD_ROUNDS = 6
OT_PERIOD_WINS = 4

# Empirical post-halftime pistol-round equip value -- see module
# docstring for how this was measured (not assumed) from round_data.csv.
HALFTIME_CT_EQUIP = 4250.0
HALFTIME_T_EQUIP = 4350.0

# Quantization step (dollars) for DP memoization keys only -- doesn't
# affect the actual value fed to either model. XGBoost's predictions are
# piecewise-constant over fairly wide regions (round_model.py's trees are
# only depth 2), so this is a safe, cheap way to collapse many
# near-identical hypothetical branches into one memoized evaluation.
EQUIP_ROUND_TO = 100
# Defensive cap for the memo key only -- round_data.csv's own streaks
#(uncapped in the raw data) top out around 12; this just guards against
# an OT-heavy simulated path pushing the memo key space up unnecessarily.
STREAK_MEMO_CAP = 20
# Hard stop on how many overtime periods the DP will simulate before just
# resolving in favor of whoever's ahead -- see module docstring. 8 periods
# past regulation (48 more rounds) is astronomically unlikely to ever
# matter to the final probability; it exists so a pathological path can't
# recurse forever instead of because real matches get anywhere near it.
MAX_OT_PERIODS = 8

_State = namedtuple("_State", "ct_score t_score ct_equip t_equip ct_streak t_streak")


def _match_winner(ct_score, t_score):
    """Returns 'CT', 'T', or None if the match isn't decided yet at this
    (ct_score, t_score) -- see module docstring for the exact rules."""
    total = ct_score + t_score
    if total < REGULATION_ROUNDS:
        if ct_score >= REGULATION_WIN:
            return "CT"
        if t_score >= REGULATION_WIN:
            return "T"
        return None

    period = (total - REGULATION_ROUNDS) // OT_PERIOD_ROUNDS
    open_score = REGULATION_ROUNDS // 2 + (OT_PERIOD_ROUNDS // 2) * period
    threshold = open_score + OT_PERIOD_WINS
    if ct_score >= threshold:
        return "CT"
    if t_score >= threshold:
        return "T"
    if period >= MAX_OT_PERIODS:
        # Never realistically reached -- see MAX_OT_PERIODS.
        return "CT" if ct_score >= t_score else "T"
    return None


def _advance_state(state, ct_won, next_ct_equip, next_t_equip):
    """Applies one round's outcome to `state`, given the next round's
    already-forecast equip values (the same forecast for both branches --
    see module docstring for why), and returns the resulting next-round
    pre-round state, including halftime/overtime side-swaps and resets."""
    ct_score = state.ct_score + (1 if ct_won else 0)
    t_score = state.t_score + (0 if ct_won else 1)
    ct_streak = 0 if ct_won else state.ct_streak + 1
    t_streak = 0 if not ct_won else state.t_streak + 1
    ct_equip, t_equip = next_ct_equip, next_t_equip

    total = ct_score + t_score
    if total == REGULATION_ROUNDS // 2:  # == 12: entering round 13, halftime
        ct_score, t_score = t_score, ct_score
        ct_streak, t_streak = t_streak, ct_streak
        ct_equip, t_equip = HALFTIME_CT_EQUIP, HALFTIME_T_EQUIP
    elif total >= REGULATION_ROUNDS:
        into_period = (total - REGULATION_ROUNDS) % OT_PERIOD_ROUNDS
        if into_period == 0:  # start of a new OT period
            ct_streak, t_streak = 1, 1
        elif into_period == OT_PERIOD_ROUNDS // 2:  # mid-period side swap
            ct_score, t_score = t_score, ct_score
            ct_streak, t_streak = t_streak, ct_streak

    return _State(ct_score, t_score, ct_equip, t_equip, ct_streak, t_streak)


def _quantize(state):
    """Snaps a state's equip values to EQUIP_ROUND_TO and caps its streaks
    at STREAK_MEMO_CAP, so states from different branches that are
    effectively the same collapse onto one dict key. The DP treats this
    quantized state as canonical from here on (not just a lookup key) --
    see module docstring on why quantizing is safe given round_model's
    depth-2 trees."""
    return _State(
        state.ct_score, state.t_score,
        round(state.ct_equip / EQUIP_ROUND_TO) * EQUIP_ROUND_TO,
        round(state.t_equip / EQUIP_ROUND_TO) * EQUIP_ROUND_TO,
        min(state.ct_streak, STREAK_MEMO_CAP),
        min(state.t_streak, STREAK_MEMO_CAP),
    )


class MatchSimulator:
    """Wraps a fitted round_model classifier and economy_model regressor
    to compute P(CT wins the match) from any pre-round state via the DP
    described in the module docstring.

    Implemented as a *forward* probability propagation, not a top-down
    recursion: a naive "recurse on each of the 2 branches per round, one
    model call per node" approach means up to ~2^24 tree nodes just for
    regulation, each paying full Python/pandas/XGBoost call overhead --
    intractably slow in practice (confirmed: killed after 5+ minutes with
    zero output on real data). Instead, this keeps a probability
    distribution over quantized states one round at a time -- {state:
    probability mass} -- and at each round batches every active state's
    round_model and economy_model calls into a single vectorized
    prediction each, splits each state's mass into its win/lose branches,
    and re-aggregates by quantized state so branches that reconverge
    merge instead of multiplying. Any branch that lands on a decided
    match has its mass folded into the running CT/T total immediately
    and drops out of the active distribution. Because every round strictly
    increases total rounds decided and _match_winner forces a resolution
    by MAX_OT_PERIODS, the active distribution is guaranteed to empty out
    within a bounded number of rounds.
    """

    def __init__(self, round_clf, economy_reg):
        self.round_clf = round_clf
        self.economy_reg = economy_reg
        self._start_cache = {}

    def p_ct_wins_match(self, ct_score, t_score, ct_equip_value, t_equip_value,
                         ct_loss_bonus_streak, t_loss_bonus_streak):
        winner = _match_winner(ct_score, t_score)
        if winner is not None:
            return 1.0 if winner == "CT" else 0.0

        cache_key = (ct_score, t_score, float(ct_equip_value), float(t_equip_value),
                     int(ct_loss_bonus_streak), int(t_loss_bonus_streak))
        cached = self._start_cache.get(cache_key)
        if cached is not None:
            return cached

        start = _quantize(_State(
            ct_score, t_score, float(ct_equip_value), float(t_equip_value),
            int(ct_loss_bonus_streak), int(t_loss_bonus_streak),
        ))

        ct_total = 0.0
        active = {start: 1.0}
        while active:
            states = list(active.keys())
            masses = [active[s] for s in states]

            round_df = pd.DataFrame(
                [{"ct_equip_value": s.ct_equip, "t_equip_value": s.t_equip} for s in states]
            )[round_model.FEATURE_COLS]
            p_ct_round = self.round_clf.predict_proba(round_df)[:, 1]

            econ_rows = []
            for s, p in zip(states, p_ct_round):
                econ_rows.append({
                    "own_equip_value": s.ct_equip, "opp_equip_value": s.t_equip,
                    "own_loss_bonus_streak": s.ct_streak, "own_win_prob": p,
                })
                econ_rows.append({
                    "own_equip_value": s.t_equip, "opp_equip_value": s.ct_equip,
                    "own_loss_bonus_streak": s.t_streak, "own_win_prob": 1 - p,
                })
            econ_df = pd.DataFrame(econ_rows)[economy_model.FEATURE_COLS]
            econ_preds = self.economy_reg.predict(econ_df).clip(0, economy_model.EQUIP_VALUE_MAX)
            next_ct_equip = econ_preds[0::2]
            next_t_equip = econ_preds[1::2]

            next_active = defaultdict(float)
            for s, mass, p_ct, nce, nte in zip(states, masses, p_ct_round, next_ct_equip, next_t_equip):
                for ct_won, branch_mass in ((True, mass * p_ct), (False, mass * (1 - p_ct))):
                    branch = _quantize(_advance_state(s, ct_won, nce, nte))
                    winner = _match_winner(branch.ct_score, branch.t_score)
                    if winner == "CT":
                        ct_total += branch_mass
                    elif winner == "T":
                        pass  # only need the CT total; T's share is implicitly 1 - ct_total
                    else:
                        next_active[branch] += branch_mass

            active = next_active

        self._start_cache[cache_key] = ct_total
        return ct_total


def _fit_round_model(df):
    X, y, groups = df[round_model.FEATURE_COLS], df[round_model.TARGET_COL], df[ID_COL]
    n_estimators = round_model._pick_n_estimators(X, y, groups)
    model = round_model._make_model(n_estimators=n_estimators)
    model.fit(X, y)
    return model


def _fit_economy_model(df, round_clf):
    X, y, groups = economy_model._load_data(round_clf=round_clf)
    n_estimators = economy_model._pick_n_estimators(X, y, groups)
    model = economy_model._make_model(n_estimators=n_estimators)
    model.fit(X, y)
    return model


def run_train():
    if not os.path.exists(ROUND_DATA_CSV):
        raise SystemExit(f"{ROUND_DATA_CSV} not found -- run parse_demos.py first.")

    print("Training round_model on all data...")
    X, y, groups = round_model._load_data()
    round_model.run_train(X, y, groups)

    print("\nTraining economy_model on all data...")
    X, y, groups = economy_model._load_data()
    economy_model.run_train(X, y, groups)

    print(
        "\nBoth models trained. match_model has no parameters of its own -- "
        "run `python match_model.py test` to evaluate the resulting match-level DP."
    )


def run_test():
    if not os.path.exists(ROUND_DATA_CSV):
        raise SystemExit(f"{ROUND_DATA_CSV} not found -- run parse_demos.py first.")

    df = pd.read_csv(ROUND_DATA_CSV)
    round_model._require_multiple_groups(df[ID_COL], "a match-grouped test split")

    # Same split round_model.py/economy_model.py use (same TEST_SIZE/
    # RANDOM_STATE, same row order from the same CSV) so "held-out match"
    # means the same thing everywhere.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[ID_COL]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    print(f"Train: {len(train_df)} rounds from {train_df[ID_COL].nunique()} match(es)")
    print(f"Test:  {len(test_df)} rounds from {test_df[ID_COL].nunique()} match(es)")

    print("\nFitting round_model on the train matches only...")
    round_clf = _fit_round_model(train_df)
    print("Fitting economy_model on the train matches only...")
    economy_reg = _fit_economy_model(train_df, round_clf)

    simulator = MatchSimulator(round_clf, economy_reg)

    per_round_correct = defaultdict(list)
    per_round_baseline_correct = defaultdict(list)

    for _, match_rows in test_df.groupby(ID_COL):
        true_ct_won = bool(match_rows["winner_is_ct"].iloc[-1])

        for _, r in match_rows.iterrows():
            round_num = int(r["ct_score"] + r["t_score"]) + 1  # derived -- round_num isn't a stored column

            p_ct = simulator.p_ct_wins_match(
                r["ct_score"], r["t_score"], r["ct_equip_value"], r["t_equip_value"],
                r["ct_loss_bonus_streak"], r["t_loss_bonus_streak"],
            )
            predicted_ct_won = p_ct >= 0.5
            per_round_correct[round_num].append(predicted_ct_won == true_ct_won)

            # Naive baseline: whoever's currently leading on the scoreboard
            # (equip advantage breaks a tied score) wins the match.
            if r["ct_score"] != r["t_score"]:
                baseline_ct_won = r["ct_score"] > r["t_score"]
            else:
                baseline_ct_won = r["ct_equip_value"] >= r["t_equip_value"]
            per_round_baseline_correct[round_num].append(baseline_ct_won == true_ct_won)

    print(f"\n{'Round':>6}  {'DP acc':>8}  {'Baseline':>8}  {'n':>5}")
    all_correct, all_baseline = [], []
    for round_num in sorted(per_round_correct):
        correct = per_round_correct[round_num]
        baseline = per_round_baseline_correct[round_num]
        all_correct.extend(correct)
        all_baseline.extend(baseline)
        print(
            f"{round_num:>6}  {sum(correct) / len(correct):>8.4f}  "
            f"{sum(baseline) / len(baseline):>8.4f}  {len(correct):>5}"
        )

    print(
        f"\nOverall DP accuracy:       {sum(all_correct) / len(all_correct):.4f} "
        f"(n={len(all_correct)} round-level match predictions across {test_df[ID_COL].nunique()} matches)"
    )
    print(f"Overall baseline accuracy: {sum(all_baseline) / len(all_baseline):.4f}")


def main():
    argp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argp.add_argument(
        "mode", choices=["train", "test"],
        help="'train' trains round_model and economy_model on all data; "
             "'test' evaluates the DP's match-winner accuracy on a held-out 20%% of matches.",
    )
    args = argp.parse_args()

    if args.mode == "train":
        run_train()
    else:
        run_test()


if __name__ == "__main__":
    main()
