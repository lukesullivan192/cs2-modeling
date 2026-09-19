"""Live equivalent of a round_data.csv row (see parse_demos.py / README):
for each round of a live HLTV match, extracts match_id, round_num,
ct_score, t_score, ct_equip_value, t_equip_value, ct_loss_bonus_streak,
t_loss_bonus_streak -- the exact pre-round state
models/run_match_model.py's MatchSimulator.p_ct_wins_match() takes -- so a live
round's state can be fed straight into the match-winner DP. See
live/run_live_model.py to do that live, round by round.

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
team-not-label tracking models/run_match_model.py's DP applies manually via
CS2's known rules.
"""
import argparse

try:
    from .fetch_live_match import fetch_live_match, resolve_url_and_id
except ImportError:
    from fetch_live_match import fetch_live_match, resolve_url_and_id

LIVE_ROUND_STATE = "started"
# How much roundTimeRemainingMS must drop from its reading at this round's
# first live tick before an equip snapshot with no kill yet is trusted as
# final. Measured empirically against live traffic (see
# fetch_live_round_state): per-player equipmentValue fully settled within
# ~2000ms of round-clock time in every clean round-start observed; this
# adds margin rather than cutting it close.
EQUIP_SETTLE_MS = 3000


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


def _extract_round_state(data, match_id, locked):
    return {
        "match_id": match_id,
        "round_num": data.get("currentRound"),
        "ct_score": data.get("counterTerroristScore"),
        "t_score": data.get("terroristScore"),
        "ct_equip_value": _equip_value(data.get("CT", [])),
        "t_equip_value": _equip_value(data.get("TERRORIST", [])),
        "ct_loss_bonus_streak": _trailing_losses(data.get("ctMatchHistory")),
        "t_loss_bonus_streak": _trailing_losses(data.get("terroristMatchHistory")),
        # True once this round's first kill has locked the equip snapshot
        # in place (see module docstring); False means equip_value is
        # still climbing as buys land and shouldn't be trusted as final --
        # a caller predicting once per round should wait for this to flip
        # True rather than acting on the first tick of a new round, which
        # is captured right at freeze time's start before anyone's bought
        # anything (equip genuinely reads $0 at that instant).
        "locked": locked,
    }


def fetch_live_round_state(match: str, rounds: int = 0, verbose: bool = True):
    """Yields, once per scorebot tick, either the current round's pre-round
    state (ready for models/run_match_model.py's MatchSimulator -- see
    live/run_live_model.py) or None while the match is between rounds.

    match: HLTV match URL or numeric match id.
    rounds: stop after N RoundEnd events (0: run until the caller stops iterating).
    """
    _url, match_id = resolve_url_and_id(match)
    frozen_data = None          # this round's most recent still-nobody-dead scoreboard tick
    locked = False               # True once this round's equip snapshot is considered final
    round_start_remain_ms = None  # roundTimeRemainingMS at this round's first live tick

    for ev in fetch_live_match(match, rounds=rounds):
        if ev["event"] != "scoreboard":
            continue
        data = ev["data"]
        ct_players, t_players = data.get("CT", []), data.get("TERRORIST", [])
        # Require both rosters to actually be populated: Python's all() is
        # vacuously True over an empty list, so without this check a tick
        # where one side's roster is momentarily empty (e.g. mid-rebuild
        # right at a round transition) would silently pass as "all alive"
        # below and get captured as this round's state -- summing that
        # empty roster's equipmentValue then reports $0 for a side that,
        # on the actual HLTV page, plainly isn't at $0.
        is_live_round = (
            data.get("live", True)
            and data.get("currentRoundState") == LIVE_ROUND_STATE
            and ct_players and t_players
        )

        if is_live_round:
            if round_start_remain_ms is None:
                round_start_remain_ms = data.get("roundTimeRemainingMS")
            all_alive = all(p.get("alive", True) for side in (ct_players, t_players) for p in side)
            # roundTimeRemainingMS counts down from ~115000 starting on
            # this round's very first "started" tick -- a server clock,
            # not something derived from our own poll timing. Captured
            # live traffic showed per-player equipmentValue fully settles
            # within ~2000ms of round-clock time after that first tick
            # (e.g. 0 -> 700 -> 1400 -> ... -> final value, done by
            # roundTimeRemainingMS dropping ~2000 from its start-of-round
            # reading), so once that much round-clock time has elapsed,
            # the latest all-alive snapshot can be trusted as final
            # without waiting for a kill -- which could otherwise be
            # 10-100+ seconds into the round on a slow-playing side.
            settled = (
                round_start_remain_ms is not None
                and data.get("roundTimeRemainingMS") is not None
                and round_start_remain_ms - data["roundTimeRemainingMS"] >= EQUIP_SETTLE_MS
            )
            if not locked:
                if all_alive:
                    frozen_data = data
                    if settled:
                        locked = True
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
            round_start_remain_ms = None

        yield _extract_round_state(frozen_data, match_id, locked) if (is_live_round and frozen_data) else None


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
