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

MatchSimulator computes this as a *forward* graph expansion (a set of
active states, advanced one round-depth at a time), not a top-down
recursion -- a naive "recurse into both branches per round, one model
call per node" version means up to ~2^24 tree nodes for regulation alone,
each paying full Python/pandas/XGBoost call overhead, and was confirmed
intractable (killed after 5+ minutes with zero output on real data). It
also resolves a whole *batch* of seed states at once (test mode's every
(match, round) query, in one call) rather than one independent
simulation per seed: at each round-depth it batches every not-yet-
resolved state's round_model/economy_model calls into one vectorized
prediction, and dedupes children by a lightly-quantized state (see
EQUIP_ROUND_TO) so branches -- from the same seed *or different seeds* --
that land on essentially the same state become one node instead of
separate subtrees. Once expansion empties out (guaranteed -- see below),
each state's P(CT wins) is resolved bottom-up (children before parents,
valid because every round strictly increases total rounds decided) and
cached for the simulator's lifetime, so later `resolve` calls reuse any
state an earlier call already worked out. This sharing is most of the
DP's real cost: two rounds of the same match share almost their entire
future, and re-deriving that future from scratch per (match, round)
query -- thousands of times over a test set -- was the actual bottleneck,
not any single simulation being slow.

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
import time
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


def _swaps_sides(total):
    """Whether the pre-round state for `total` rounds-already-decided sits
    on the far side of a CT/T label swap from `total - 1` -- i.e. whether
    _advance_state's ct_score/t_score (and ct_streak/t_streak) for this
    `total` refer to the OPPOSITE physical teams from `total - 1`'s. True
    at halftime (entering round 13) and at each OT period's mid-point --
    see module docstring. Shared with resolve()'s bottom-up combine step,
    which must flip a child's cached "P(CT wins)" (defined per-state, in
    that state's own CT/T labels) before folding it into a parent whose
    CT/T labels refer to a different physical team whenever this is True
    for the parent-to-child transition."""
    if total == REGULATION_ROUNDS // 2:  # == 12: entering round 13, halftime
        return True
    if total >= REGULATION_ROUNDS:
        return (total - REGULATION_ROUNDS) % OT_PERIOD_ROUNDS == OT_PERIOD_ROUNDS // 2
    return False


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
    if _swaps_sides(total):
        ct_score, t_score = t_score, ct_score
        ct_streak, t_streak = t_streak, ct_streak
        if total == REGULATION_ROUNDS // 2:
            ct_equip, t_equip = HALFTIME_CT_EQUIP, HALFTIME_T_EQUIP
    elif total >= REGULATION_ROUNDS and (total - REGULATION_ROUNDS) % OT_PERIOD_ROUNDS == 0:
        ct_streak, t_streak = 1, 1  # start of a new OT period

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

    Implemented as a *forward* graph expansion, not a top-down recursion:
    a naive "recurse on each of the 2 branches per round, one model call
    per node" approach means up to ~2^24 tree nodes just for regulation,
    each paying full Python/pandas/XGBoost call overhead -- intractably
    slow in practice (confirmed: killed after 5+ minutes with zero output
    on real data). `resolve` additionally takes a whole batch of seed
    states at once (e.g. every held-out round across every match in
    run_test) and expands them together, one shared frontier keyed on the
    *quantized* state: at each round-depth it batches every currently
    unresolved state's round_model/economy_model calls into a single
    vectorized prediction, and dedupes children by quantized state so
    branches from *different* seeds that land on essentially the same
    state (constant in practice -- most rounds only differ by a few
    dollars of equip value, quantized away by EQUIP_ROUND_TO) become one
    node instead of separate subtrees.

    Once the frontier empties (guaranteed -- every round strictly
    increases total rounds decided, and _match_winner forces a resolution
    by MAX_OT_PERIODS), each state's value is resolved bottom-up in
    reverse discovery order (children before parents -- valid because
    discovery order is already topological, for the same reason) and
    cached in `self._value_cache` for the simulator's lifetime. Resolving
    a state is a pure function of that state onward -- it doesn't depend
    on how much probability mass reached it or from which seed -- so the
    cache carries over across `resolve` calls too: a later call that
    happens to revisit a state an earlier call already resolved gets it
    for free. This sharing is most of where the DP's cost actually goes:
    a naive one-simulation-per-query approach redundantly re-derives
    almost the same future for every one of the thousands of (match,
    round) queries run_test makes (round 5 and round 6 of the same match
    share nearly their entire remaining match), and re-doing that from
    scratch per query -- rather than once, shared -- was the real
    bottleneck.
    """

    def __init__(self, round_clf, economy_reg):
        self.round_clf = round_clf
        self.economy_reg = economy_reg
        self._value_cache = {}  # quantized _State -> P(CT wins); includes terminal states

    @property
    def resolved_state_count(self):
        return len(self._value_cache)

    def _resolved(self, state):
        """Returns state's cached-or-terminal value, or None if it still
        needs a model-driven expansion. Terminal states are cached on
        first sight so later lookups (including from other seeds/calls)
        skip _match_winner entirely."""
        cached = self._value_cache.get(state)
        if cached is not None:
            return cached
        winner = _match_winner(state.ct_score, state.t_score)
        if winner is None:
            return None
        value = 1.0 if winner == "CT" else 0.0
        self._value_cache[state] = value
        return value

    def p_ct_wins_match(self, ct_score, t_score, ct_equip_value, t_equip_value,
                         ct_loss_bonus_streak, t_loss_bonus_streak):
        """Convenience single-state wrapper around `resolve` (see there
        for the batched form run_test uses)."""
        return self.resolve([(
            ct_score, t_score, ct_equip_value, t_equip_value,
            ct_loss_bonus_streak, t_loss_bonus_streak,
        )])[0]

    def resolve(self, seeds, verbose=False):
        """Computes P(CT wins the match) for every (ct_score, t_score,
        ct_equip_value, t_equip_value, ct_loss_bonus_streak,
        t_loss_bonus_streak) tuple in `seeds`, sharing all downstream
        model-driven work -- across the seeds in this call *and* across
        any previous `resolve`/`p_ct_wins_match` calls on this simulator
        -- via `self._value_cache`. Returns a list of probabilities
        aligned with `seeds`."""
        start_states = [
            _quantize(_State(cs, ts, float(ce), float(te), int(cls_), int(tls)))
            for cs, ts, ce, te, cls_, tls in seeds
        ]

        # Forward discovery pass: find every not-yet-resolved state
        # reachable from these seeds. Seeds start at different round
        # numbers (e.g. round 1 of one match, round 20 of another), so
        # the number of BFS *iterations* to reach a given state isn't a
        # valid topological order -- two different seeds can discover the
        # very same state after different numbers of steps. What's
        # invariant is the state's own total rounds decided
        # (ct_score + t_score), which strictly increases by 1 every round
        # regardless of path length, so states are bucketed by that for
        # resolving in reverse below, separately from the iteration loop
        # (kept only to batch model calls across everything active).
        children = {}
        by_total = defaultdict(set)
        frontier = {s for s in start_states if self._resolved(s) is None}
        visited = set(frontier)
        for s in frontier:
            by_total[s.ct_score + s.t_score].add(s)
        iterations = 0
        while frontier:
            iterations += 1
            states = list(frontier)

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

            next_frontier = set()
            for s, p_ct, nce, nte in zip(states, p_ct_round, next_ct_equip, next_t_equip):
                ct_child = _quantize(_advance_state(s, True, nce, nte))
                t_child = _quantize(_advance_state(s, False, nce, nte))
                # Both branches land on the same total-rounds-decided, so
                # whether this edge crosses a label swap depends only on s.
                swapped = _swaps_sides(s.ct_score + s.t_score + 1)
                children[s] = (p_ct, ct_child, t_child, swapped)
                for child in (ct_child, t_child):
                    if child not in visited and self._resolved(child) is None:
                        visited.add(child)
                        next_frontier.add(child)
                        by_total[child.ct_score + child.t_score].add(child)
            frontier = next_frontier

            if verbose:
                print(
                    f"  iteration {iterations}: {len(states)} active state(s), "
                    f"{self.resolved_state_count} resolved so far",
                    end="\r", flush=True,
                )
        if verbose and iterations:
            print()

        # Resolve bottom-up in strictly decreasing total-rounds-decided
        # order: every state's children have total + 1, so by the time a
        # total bucket is processed, both its children are already
        # terminal/cached or sitting in an already-processed (higher)
        # bucket.
        for total in sorted(by_total, reverse=True):
            for s in by_total[total]:
                p_ct, ct_child, t_child, swapped = children[s]
                ct_val = self._value_cache[ct_child]
                t_val = self._value_cache[t_child]
                if swapped:
                    # ct_child/t_child's own "P(CT wins)" is in THEIR
                    # CT/T labels, which belong to the opposite physical
                    # teams from s's -- flip both before folding them into
                    # s's value, or s ends up holding P(s's T team wins).
                    ct_val, t_val = 1.0 - ct_val, 1.0 - t_val
                self._value_cache[s] = p_ct * ct_val + (1 - p_ct) * t_val

        return [self._value_cache[s] for s in start_states]


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

    total_matches = test_df[ID_COL].nunique()
    # Last row per match_id gives that match's real final winner.
    match_final_ct_won = test_df.groupby(ID_COL)["winner_is_ct"].last()

    # Collect every held-out round's pre-round state as one big batch of
    # DP seeds, instead of looping match-by-match/round-by-round -- see
    # MatchSimulator.resolve for why resolving them together (rather than
    # as independent simulations) is what actually makes this fast: most
    # of these seeds share almost their entire remaining-match future.
    seeds, round_nums, match_ids = [], [], []
    for r in test_df.itertuples(index=False):
        seeds.append((
            r.ct_score, r.t_score, r.ct_equip_value, r.t_equip_value,
            r.ct_loss_bonus_streak, r.t_loss_bonus_streak,
        ))
        round_nums.append(int(r.ct_score + r.t_score) + 1)  # derived -- round_num isn't a stored column
        match_ids.append(getattr(r, ID_COL))

    print(
        f"\nResolving the DP over {len(seeds)} round-level states "
        f"across {total_matches} held-out match(es)..."
    )
    start_time = time.time()
    p_ct_all = simulator.resolve(seeds, verbose=True)
    elapsed = time.time() - start_time
    print(f"  done in {elapsed:.1f}s ({simulator.resolved_state_count} distinct states resolved)")

    per_round_correct = defaultdict(list)
    per_round_baseline_correct = defaultdict(list)

    for (ct_score, t_score, ct_equip_value, t_equip_value, *_), round_num, match_id, p_ct in zip(
        seeds, round_nums, match_ids, p_ct_all
    ):
        true_ct_won = bool(match_final_ct_won[match_id])
        predicted_ct_won = p_ct >= 0.5
        per_round_correct[round_num].append(predicted_ct_won == true_ct_won)

        # Naive baseline: whoever's currently leading on the scoreboard
        # (equip advantage breaks a tied score) wins the match.
        if ct_score != t_score:
            baseline_ct_won = ct_score > t_score
        else:
            baseline_ct_won = ct_equip_value >= t_equip_value
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
