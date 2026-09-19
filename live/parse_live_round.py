"""Live equivalent of a round_data.csv row (see parse_demos.py / README):
for each round of a live HLTV match, extracts match_id, round_num,
ct_score, t_score, ct_equip_value, t_equip_value, ct_loss_bonus_streak,
t_loss_bonus_streak -- the exact pre-round state
models/match_model.py's MatchSimulator.p_ct_wins_match() takes -- so a live
round's state can be fed straight into the match-winner DP. See
live/run_match_model.py to do that live, round by round.

Usage: python live/parse_live_round.py <hltv_match_url_or_id>

    python live/parse_live_round.py https://www.hltv.org/matches/2398160/saw-youngsters-vs-revenix-hyperx-retake-season-12
    python live/parse_live_round.py 2398160

Or import fetch_live_round_state() and iterate it directly:

    from live.parse_live_round import fetch_live_round_state

    for round_state in fetch_live_round_state("2398160"):
        if round_state is None:
            continue   # between rounds: freeze time, a timeout, halftime, etc.
        ...

fetch_live_round_state() yields one item per underlying scorebot tick:
  - the current round's pre-round state, while a round is actually live
  - None, whenever the match is between rounds (freeze time, a round-end
    lull, a timeout, halftime, warmup) -- there's no valid pre-round state
    to predict from

Equipment values: HLTV's scorebot exposes an aggregate per-side equip
total (ctInitialEquipmentValue/tInitialEquipmentValue), but capturing a
live match's feed and diffing consecutive ticks showed it keeps climbing
for several real seconds after currentRoundState has already flipped to
"started" -- a reporting lag, not real buying (CS2's buy window closes at
freeze time). Summing each living player's own equipmentValue instead
settles correctly within a tick or two of freeze time actually ending, and
--exactly like round_data.csv's current_equip_value, which this mirrors --
only otherwise changes when a player dies. So each round's state keeps
refreshing, tick by tick, until that round's first kill (all players still
report alive=true), then locks: the freshest approximation of "the instant
freeze time ends" a live feed can give, short of parsing the eventual demo
the way parse_demos.py does.

Loss-bonus streaks aren't in the scorebot's per-tick payload directly, but
each side's match history list (ctMatchHistory/terroristMatchHistory --
firstHalf + secondHalf, concatenated in chronological order) is: a side's
current loss-bonus streak is the number of trailing "lost" entries in its
own history. HLTV's backend already re-attributes that history to the
correct physical team across halftime's CT/T label swap, the same
team-not-label tracking models/match_model.py's DP applies manually via
CS2's known rules.
"""
import argparse

try:
    from .fetch_live_match import fetch_live_match, resolve_url_and_id
except ImportError:
    from fetch_live_match import fetch_live_match, resolve_url_and_id

LIVE_ROUND_STATE = "started"


def _equip_value(players):
    return int(sum(p.get("equipmentValue", 0) for p in players))


def _trailing_losses(history):
    """Counts the trailing consecutive "lost" entries in a side's match
    history (ctMatchHistory or terroristMatchHistory) -- its current
    loss-bonus streak entering the next round. `history` is HLTV's
    {"firstHalf": [...], "secondHalf": [...]} shape; the two halves are
    concatenated in chronological order first."""
    if not history:
        return 0
    played = list(history.get("firstHalf") or []) + list(history.get("secondHalf") or [])
    streak = 0
    for r in reversed(played):
        if r.get("type") != "lost":
            break
        streak += 1
    return streak


def _extract_round_state(data, match_id):
    return {
        "match_id": match_id,
        "round_num": data.get("currentRound"),
        "ct_score": data.get("counterTerroristScore"),
        "t_score": data.get("terroristScore"),
        "ct_equip_value": _equip_value(data.get("CT", [])),
        "t_equip_value": _equip_value(data.get("TERRORIST", [])),
        "ct_loss_bonus_streak": _trailing_losses(data.get("ctMatchHistory")),
        "t_loss_bonus_streak": _trailing_losses(data.get("terroristMatchHistory")),
    }


def fetch_live_round_state(match: str, rounds: int = 0, verbose: bool = True):
    """Yields, once per scorebot tick, either the current round's pre-round
    state (ready for models/match_model.py's MatchSimulator -- see
    live/run_match_model.py) or None while the match is between rounds.

    match: HLTV match URL or numeric match id.
    rounds: stop after N RoundEnd events (0: run until the caller stops iterating).
    """
    _url, match_id = resolve_url_and_id(match)
    frozen_data = None   # this round's most recent still-nobody-dead scoreboard tick
    locked = False        # True once this round's first kill has been seen

    for ev in fetch_live_match(match, rounds=rounds, verbose=verbose):
        if ev["event"] != "scoreboard":
            continue
        data = ev["data"]
        is_live_round = data.get("live", True) and data.get("currentRoundState") == LIVE_ROUND_STATE

        if is_live_round:
            all_alive = all(p.get("alive", True) for side in ("CT", "TERRORIST") for p in data.get(side, []))
            if not locked:
                if all_alive:
                    frozen_data = data
                elif frozen_data is None:
                    if verbose:
                        print("joined mid-round after the first kill; using the "
                              "current (imperfect) snapshot", flush=True)
                    frozen_data = data
                    locked = True
                else:
                    locked = True
        else:
            frozen_data = None
            locked = False

        yield _extract_round_state(frozen_data, match_id) if (is_live_round and frozen_data) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a live HLTV match's pre-round state (round_data.csv shape)")
    ap.add_argument("match", help="HLTV match URL or numeric match id")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N RoundEnd events (default: run until Ctrl+C)")
    args = ap.parse_args()

    last_printed = None
    between_rounds_printed = False
    try:
        for round_state in fetch_live_round_state(args.match, rounds=args.rounds):
            if round_state is None:
                if not between_rounds_printed:
                    print("between rounds -- no state yet", flush=True)
                between_rounds_printed = True
                last_printed = None
                continue
            between_rounds_printed = False
            if round_state == last_printed:
                continue
            last_printed = round_state
            print(round_state, flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
