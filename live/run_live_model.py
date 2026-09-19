"""Runs models/match_model.py's match-winner DP live against one HLTV
match: streams live/parse_live_round.py's round-by-round state and
prints models/match_model.py's P(CT wins the match) once per round, as
soon as parse_live_round.py locks in that round's state (the gamestate
right after freeze time ends).

Usage: python live/run_live_model.py <hltv_match_url_or_id>

    python live/run_live_model.py https://www.hltv.org/matches/2398160/saw-youngsters-vs-revenix-hyperx-retake-season-12
    python live/run_live_model.py 2398160

Must be run from the repo root (models/saves/*.json are loaded/saved by
path relative to cwd, same as models/match_model.py itself expects).

If models/saves/round_model.json and models/saves/economy_model.json
already exist, they're loaded as-is. Otherwise this trains both first (the
same as running `python models/match_model.py train`), which needs
data/round_data.csv (i.e. parse_demos.py must have already been run).
"""
import argparse
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

try:
    from .parse_live_round import fetch_live_round_state
except ImportError:
    from parse_live_round import fetch_live_round_state


def load_simulator():
    """Ensures a trained round_model/economy_model exist (training both, via
    models/match_model.py's train mode, if either is missing) and returns a
    ready models/match_model.py MatchSimulator."""
    if MODELS_DIR not in sys.path:
        sys.path.insert(0, MODELS_DIR)
    import economy_model
    import match_model
    import round_model

    if not os.path.exists(round_model.MODEL_PATH) or not os.path.exists(economy_model.MODEL_PATH):
        print("No saved round_model/economy_model found -- training both on the full dataset...")
        match_model.run_train()

    round_clf = economy_model._load_round_model()
    economy_reg = economy_model._load_economy_model()
    return match_model.MatchSimulator(round_clf, economy_reg)


def predict_match_winner(round_state, simulator):
    """Feeds one parse_live_round.fetch_live_round_state() row into
    `simulator` (see load_simulator) and returns P(CT wins the match)."""
    return simulator.p_ct_wins_match(
        round_state["ct_score"], round_state["t_score"],
        round_state["ct_equip_value"], round_state["t_equip_value"],
        round_state["ct_loss_bonus_streak"], round_state["t_loss_bonus_streak"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the match-winner DP live against one HLTV match")
    ap.add_argument("match", help="HLTV match URL or numeric match id")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N RoundEnd events (default: run until Ctrl+C)")
    args = ap.parse_args()

    simulator = load_simulator()

    predicted_round_num = None
    between_rounds_printed = False
    try:
        for round_state in fetch_live_round_state(args.match, rounds=args.rounds):
            if round_state is None:
                if not between_rounds_printed:
                    print("between rounds -- no state to predict from", flush=True)
                between_rounds_printed = True
                continue
            between_rounds_printed = False
            # parse_live_round.py keeps refining a round's equip_value
            # tick by tick until that round's first kill locks it in (see
            # its docstring) -- the FIRST tick of a new round is captured
            # right as freeze time starts, before anyone's bought
            # anything, so equip_value there can genuinely read $0.
            # Predicting off that tick instead of the locked one is what
            # produced misleading $0 equip predictions; wait for "locked".
            if not round_state["locked"] or round_state["round_num"] == predicted_round_num:
                continue
            predicted_round_num = round_state["round_num"]

            p_ct = predict_match_winner(round_state, simulator)
            print(
                f"round {round_state['round_num']}: {round_state['ct_score']}-{round_state['t_score']} "
                f"ct_equip={round_state['ct_equip_value']} t_equip={round_state['t_equip_value']} "
                f"ct_loss_streak={round_state['ct_loss_bonus_streak']} t_loss_streak={round_state['t_loss_bonus_streak']} "
                f"| P(CT wins match)={p_ct:.3f}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
