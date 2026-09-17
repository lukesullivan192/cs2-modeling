"""
Parses CS2 .dem files into two training sets and writes them to ./data/:

- data.csv: mid-round game-state snapshots for a "live" round-outcome model.
  One output row = one sampled point in time during a live round (after
  freezetime ends, before the round ends), labeled with that round's eventual
  winner. This is the shape a "given the state right now, who wins the
  round" model needs -- as opposed to one row per round or one row per kill.

- round_data.csv: one row per round, taken at the instant freezetime ends
  (before anyone can move, buy, or take damage). This is the shape a
  "forecast a round that hasn't been played yet" model needs, for chaining
  round predictions into a match-level Monte Carlo/DP simulation: at
  simulation time you only ever have pre-round information (score,
  economy, loss bonus) to condition on, so columns that are always
  constant/uninformative at that instant -- alive counts, deaths-so-far,
  health, flash state, bomb state, elapsed time -- are dropped rather than
  carried over from data.csv's schema.

Every column in both CSVs is numeric so the files can be fed straight into
a model: X = df.iloc[:, :-1], y = df.iloc[:, -1]. The last column,
winner_is_ct, is a 0/1 label -- this is a binary classification problem
(predict which side wins the round from its current state), not a
regression one, so pick a classifier (e.g. logistic regression / gradient
boosted trees / random forest classifier) rather than a regressor.
match_id (first column) is a deterministic hash of the source demo
filename for match-aware splitting (e.g. sklearn GroupKFold, so rounds from
the same match don't end up on both sides of a train/test split) -- it's an
identifier, not a predictive feature, and should be excluded from X.
round_end_reason is deliberately NOT included: it's recorded at the same
event as the winner and near-perfectly determines it (e.g. "bomb_defused"
=> CT won), so it would leak the label rather than predict it.

Requires demoparser2 (https://github.com/LaihoE/demoparser). demoparser2's
tick-level field names have shifted across versions, so field selection is
defensive: we ask for a wishlist of fields and, if the library rejects one,
we parse its own error message to find out which fields it actually supports
and retry with the intersection.

Re-running this script never reparses a demo it already has rows for, and
never re-attempts a demo that failed with a permanent (e.g. corrupt-file)
error -- see `.parsed_demos.json` in the data dir, tracked the same way
demo_fetcher.py tracks already-downloaded matches.
"""
import argparse
import glob
import json
import os
import re
import sys
import zlib

import pandas as pd
from demoparser2 import DemoParser

DEMOS_DIR = "demos"
DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "data.csv")
ROUND_OUTPUT_CSV = os.path.join(DATA_DIR, "round_data.csv")
TRACKING_FILE = os.path.join(DATA_DIR, ".parsed_demos.json")

# How often (in seconds of game time) to sample state during a live round.
SNAPSHOT_INTERVAL_SECONDS = 5.0

# CS2's C4 fuse time.
BOMB_TIMER_SECONDS = 40.0

TEAM_T = 2
TEAM_CT = 3

# Fixed numeric codes for the CS2 map pool. Hardcoded (rather than
# assigned on the fly) so the same map always gets the same id across
# separate runs of this script that append to the same data.csv --
# assigning ids dynamically per-run would make the column mean different
# things in different rows. Unrecognized maps (e.g. a workshop/retired map)
# fall back to -1.
MAP_NAME_IDS = {
    "de_ancient": 0,
    "de_anubis": 1,
    "de_dust2": 2,
    "de_inferno": 3,
    "de_mirage": 4,
    "de_nuke": 5,
    "de_overpass": 6,
    "de_vertigo": 7,
    "de_train": 8,
    "de_cache": 9,
}
UNKNOWN_MAP_ID = -1

# Tick-level player props we'd like. Not all of these exist in every
# demoparser2 version -- see `_resolve_tick_props`.
WANTED_TICK_PROPS = [
    "health",
    "armor_value",
    "has_defuser",
    "has_helmet",
    "current_equip_value",
    "team_num",
    "team_clan_name",
    "is_alive",
    "flash_duration",
    "ct_losing_streak",
    "t_losing_streak",
]


def _resolve_tick_props(parser, wanted_props, probe_tick):
    """
    demoparser2 raises a ValueError naming valid fields when you ask for one
    it doesn't support. Probe with the full wishlist first (cheap: one
    tick); on failure, scrape the valid-fields list out of the error and
    retry with only the props we wanted that are actually supported.
    """
    try:
        parser.parse_ticks(wanted_props, ticks=[probe_tick])
        return wanted_props
    except Exception as e:
        msg = str(e)
        valid = set(re.findall(r"'([a-zA-Z0-9_]+)'", msg))
        resolved = [p for p in wanted_props if p in valid]
        if not resolved:
            raise RuntimeError(
                f"Could not resolve any tick props against installed demoparser2. "
                f"Wanted {wanted_props}, parser error: {msg}"
            )
        missing = set(wanted_props) - set(resolved)
        if missing:
            print(f"  Note: tick props unavailable in this demoparser2 version, skipping: {sorted(missing)}")
        return resolved


def _parse_winner_side(value):
    """round_end's winner column is numeric (2=T, 3=CT) in some demoparser2
    versions and the side string ("CT"/"T") in others -- normalize both.
    Returns None for unparseable/missing values: some demos contain junk
    round_end events (e.g. a tick-0 pre-game artifact, or a mid-match
    technical-restart event) with no real winner, which callers should
    skip rather than treat as a parse failure."""
    if isinstance(value, str):
        side = value.strip().upper()
        if side == "CT":
            return TEAM_CT
        if side == "T":
            return TEAM_T
        try:
            return int(side)
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _map_name_to_id(map_name):
    map_id = MAP_NAME_IDS.get(map_name, UNKNOWN_MAP_ID)
    if map_id == UNKNOWN_MAP_ID:
        print(f"  Note: unrecognized map '{map_name}', encoding as {UNKNOWN_MAP_ID}.")
    return map_id


def _match_id(demo_file):
    """Deterministic numeric id for the source demo, for match-aware
    train/test splitting. Not a predictive feature -- see module docstring."""
    return zlib.crc32(demo_file.encode()) % 1_000_000


def _first_present(df, candidates):
    """Return the first column name from `candidates` that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _is_corrupt_demo_error(exc):
    """
    demoparser2 surfaces unrecoverable bad-file-bytes errors (e.g. a
    truncated download or bad archive extraction) as decompression
    failures. These are worth distinguishing from other errors because
    they'll fail identically on every retry until the file itself is
    replaced -- so we shouldn't burn time reparsing them every run.
    """
    msg = str(exc)
    return "DecompressionFailure" in msg or "corrupt input" in msg.lower()


def parse_demo(demo_path):
    """Returns (rows, round_rows) for a single demo: `rows` is a list of row
    dicts, one per mid-round snapshot (for data.csv); `round_rows` is a list
    of row dicts, one per round, taken at freezetime-end (for
    round_data.csv)."""
    print(f"Parsing {demo_path}...")
    parser = DemoParser(demo_path)
    demo_file = os.path.basename(demo_path)

    header = parser.parse_header()
    map_name = header.get("map_name", header.get("map", "unknown"))
    tick_rate = header.get("tick_rate") or _infer_tick_rate(header)

    round_starts = parser.parse_event("round_start")
    freeze_ends = parser.parse_event("round_freeze_end")
    round_ends = parser.parse_event("round_end")

    if round_ends is None or len(round_ends) == 0:
        print(f"  No round_end events found in {demo_file}, skipping.")
        return [], []

    winner_col = _first_present(round_ends, ["winner", "winner_side", "team"])
    if winner_col is None:
        print(f"  Could not find a winner column in round_end events for {demo_file}, skipping.")
        return [], []
    # demoparser2's round_end carries the server's own gapless round counter
    # (1-based; round 0 is a pre-game/warmup artifact, filtered out below by
    # the degenerate-round check). Prefer it over our own row position:
    # positional numbering silently reproduces gaps whenever a round_end is
    # skipped (e.g. a technical restart), and -- for a demo that's actually
    # the second part of a paused/reconnected match -- positional numbering
    # would relabel the real round N as "round 1", which is actively
    # misleading (e.g. it can show a fresh-looking "round 1" with a loss
    # bonus streak already at 4, because the engine's own economy state
    # correctly carried over from the rounds recorded in the missing part).
    round_num_col = _first_present(round_ends, ["round", "round_num"])

    # Index round_start/freeze_end ticks by round number so we can line each
    # round_end up with its own start.
    round_ends = round_ends.sort_values("tick").reset_index(drop=True)
    round_starts = round_starts.sort_values("tick").reset_index(drop=True) if round_starts is not None else pd.DataFrame()
    freeze_ends = freeze_ends.sort_values("tick").reset_index(drop=True) if freeze_ends is not None else pd.DataFrame()

    tick_props = _resolve_tick_props(parser, WANTED_TICK_PROPS, probe_tick=int(round_ends["tick"].iloc[0]))
    have_clan_name = "team_clan_name" in tick_props

    bomb_planted_events = _safe_parse_event(parser, "bomb_planted")

    # --- Pass 1: work out each round's boundaries/metadata and collect
    # every tick we'll need player state for. We deliberately don't touch
    # tick-level state yet -- see the note on `all_ticks_df` below.
    round_meta = []
    needed_ticks = set()
    for row_pos, round_end_row in round_ends.iterrows():
        round_end_tick = int(round_end_row["tick"])
        round_num = int(round_end_row[round_num_col]) if round_num_col is not None else row_pos + 1

        starts_before = round_starts[round_starts["tick"] <= round_end_tick] if len(round_starts) else pd.DataFrame()
        freezes_before = freeze_ends[freeze_ends["tick"] <= round_end_tick] if len(freeze_ends) else pd.DataFrame()
        if len(starts_before) == 0:
            continue
        round_start_tick = int(starts_before["tick"].iloc[-1])
        freeze_end_tick = int(freezes_before["tick"].iloc[-1]) if len(freezes_before) else round_start_tick

        if round_end_tick <= freeze_end_tick:
            # Degenerate/short round (e.g. warmup or a forfeited round) -- nothing "mid-round" to sample.
            continue

        winner_side = _parse_winner_side(round_end_row[winner_col])
        if winner_side is None:
            # Junk round_end event (e.g. tick-0 pre-game artifact, or a
            # mid-match technical-restart) with no real winner -- not a
            # real round, nothing to label.
            continue

        # Bomb plant (if any) within this round.
        plant_tick = None
        if bomb_planted_events is not None and len(bomb_planted_events):
            in_round = bomb_planted_events[
                (bomb_planted_events["tick"] >= freeze_end_tick) & (bomb_planted_events["tick"] <= round_end_tick)
            ]
            if len(in_round):
                plant_tick = int(in_round["tick"].iloc[0])

        # Sample ticks at a fixed cadence from freeze_end to round_end.
        step_ticks = max(1, int(SNAPSHOT_INTERVAL_SECONDS * tick_rate))
        sample_ticks = list(range(freeze_end_tick + step_ticks, round_end_tick, step_ticks))
        if not sample_ticks:
            continue

        needed_ticks.add(freeze_end_tick)
        needed_ticks.update(sample_ticks)

        round_meta.append({
            "round_num": round_num,
            "freeze_end_tick": freeze_end_tick,
            "winner_side": winner_side,
            "plant_tick": plant_tick,
            "sample_ticks": sample_ticks,
        })

    if not round_meta:
        return [], []

    # --- Pass 2: fetch every round's player state in ONE parse_ticks call.
    # demoparser2 re-scans the whole demo stream from the start on every
    # parse_ticks() call, so calling it twice per round (as this script
    # used to) meant decoding a multi-hundred-MB demo dozens of times over.
    # Batching every tick we need across the whole demo into a single call
    # is the single biggest parsing-speed win available here.
    try:
        all_ticks_df = parser.parse_ticks(tick_props, ticks=sorted(needed_ticks))
    except Exception as e:
        print(f"  Could not read tick state for {demo_file}: {e}")
        return [], []

    ticks_by_tick = {tick: snap for tick, snap in all_ticks_df.groupby("tick")}

    rows = []
    round_rows = []
    team_wins = {}  # clan_name -> cumulative rounds won so far (only used if clan names are available)
    ct_score_fallback = 0
    t_score_fallback = 0

    for meta in round_meta:
        round_num = meta["round_num"]
        freeze_end_tick = meta["freeze_end_tick"]
        winner_side = meta["winner_side"]
        plant_tick = meta["plant_tick"]
        sample_ticks = meta["sample_ticks"]

        start_state = ticks_by_tick.get(freeze_end_tick)
        if start_state is None or len(start_state) == 0:
            continue

        ct0 = start_state[start_state["team_num"] == TEAM_CT]
        t0 = start_state[start_state["team_num"] == TEAM_T]

        ct_clan = t_clan = None
        if have_clan_name:
            ct_clan = ct0["team_clan_name"].iloc[0] if len(ct0) else None
            t_clan = t0["team_clan_name"].iloc[0] if len(t0) else None

        # Score entering this round.
        if have_clan_name and ct_clan is not None and t_clan is not None:
            ct_score = team_wins.get(ct_clan, 0)
            t_score = team_wins.get(t_clan, 0)
        else:
            ct_score, t_score = ct_score_fallback, t_score_fallback

        starting_ct_alive = int((start_state["team_num"] == TEAM_CT).sum())
        starting_t_alive = int((start_state["team_num"] == TEAM_T).sum())

        # One row per round for round_data.csv, taken at the freezetime-end
        # instant itself: equipment is locked in, nobody's moved or taken
        # damage yet, so alive counts/deaths/health/flash/bomb state are all
        # fixed, uninformative values (5v5, 0, 100, 0, unplanted) -- those
        # columns are dropped here rather than carried over from data.csv's
        # schema. See module docstring.
        round_rows.append({
            "match_id": _match_id(demo_file),
            "map_name": _map_name_to_id(map_name),
            "round_num": round_num,
            "ct_score": ct_score,
            "t_score": t_score,
            "ct_equip_value": int(ct0["current_equip_value"].sum()) if "current_equip_value" in ct0 else None,
            "t_equip_value": int(t0["current_equip_value"].sum()) if "current_equip_value" in t0 else None,
            "ct_helmets": int(ct0["has_helmet"].sum()) if "has_helmet" in ct0 else None,
            "ct_defusers": int(ct0["has_defuser"].sum()) if "has_defuser" in ct0 else None,
            "ct_loss_bonus_streak": int(ct0["ct_losing_streak"].iloc[0]) if "ct_losing_streak" in ct0 and len(ct0) else None,
            "t_loss_bonus_streak": int(t0["t_losing_streak"].iloc[0]) if "t_losing_streak" in t0 and len(t0) else None,
            "winner_is_ct": int(winner_side == TEAM_CT),  # target column -- must stay last
        })

        for snap_tick in sample_ticks:
            snap = ticks_by_tick.get(snap_tick)
            if snap is None or len(snap) == 0:
                continue

            ct = snap[snap["team_num"] == TEAM_CT]
            t = snap[snap["team_num"] == TEAM_T]

            ct_alive = int(ct["is_alive"].sum()) if "is_alive" in ct else len(ct)
            t_alive = int(t["is_alive"].sum()) if "is_alive" in t else len(t)

            planted_by_now = plant_tick is not None and snap_tick >= plant_tick

            row = {
                "match_id": _match_id(demo_file),  # identifier for grouped splitting, not a feature -- see module docstring
                "map_name": _map_name_to_id(map_name),
                "round_num": round_num,
                "seconds_since_freeze_end": (int(snap_tick) - freeze_end_tick) / tick_rate,
                "ct_score": ct_score,
                "t_score": t_score,
                "ct_alive": ct_alive,
                "t_alive": t_alive,
                "ct_deaths_so_far": max(0, starting_ct_alive - ct_alive),
                "t_deaths_so_far": max(0, starting_t_alive - t_alive),
                "alive_diff": ct_alive - t_alive,
                "ct_equip_value": int(ct["current_equip_value"].sum()) if "current_equip_value" in ct else None,
                "t_equip_value": int(t["current_equip_value"].sum()) if "current_equip_value" in t else None,
                "ct_avg_health": float(ct.loc[ct["is_alive"] == True, "health"].mean()) if "health" in ct and ct_alive else 0.0,
                "t_avg_health": float(t.loc[t["is_alive"] == True, "health"].mean()) if "health" in t and t_alive else 0.0,
                "ct_helmets": int(ct["has_helmet"].sum()) if "has_helmet" in ct else None,
                "ct_defusers": int(ct["has_defuser"].sum()) if "has_defuser" in ct else None,
                "ct_flashed": int((ct["flash_duration"] > 0).sum()) if "flash_duration" in ct else None,
                "t_flashed": int((t["flash_duration"] > 0).sum()) if "flash_duration" in t else None,
                # Consecutive rounds lost by each side entering this round --
                # this is what determines their CS2 loss-bonus cash tier, and
                # captures banked economic pressure that ct/t_equip_value
                # (money actually spent) doesn't: a team can be low-buy this
                # round yet sitting on a high loss bonus for the next one.
                "ct_loss_bonus_streak": int(ct["ct_losing_streak"].iloc[0]) if "ct_losing_streak" in ct and len(ct) else None,
                "t_loss_bonus_streak": int(t["t_losing_streak"].iloc[0]) if "t_losing_streak" in t and len(t) else None,
                "bomb_planted": int(planted_by_now),
                "bomb_time_left": (
                    max(0.0, BOMB_TIMER_SECONDS - (int(snap_tick) - plant_tick) / tick_rate)
                    if planted_by_now
                    else -1.0  # sentinel: bomb not planted yet (real values are always >= 0)
                ),
                "winner_is_ct": int(winner_side == TEAM_CT),  # target column -- must stay last
            }
            rows.append(row)

        # Now that the round is resolved, update cumulative team wins for next round's score.
        if have_clan_name and ct_clan is not None and t_clan is not None:
            winner_clan = ct_clan if winner_side == TEAM_CT else t_clan
            team_wins[winner_clan] = team_wins.get(winner_clan, 0) + 1
        else:
            if winner_side == TEAM_CT:
                ct_score_fallback += 1
            else:
                t_score_fallback += 1

    return rows, round_rows


def _safe_parse_event(parser, event_name):
    try:
        return parser.parse_event(event_name)
    except Exception:
        return None


def _infer_tick_rate(header):
    """Best-effort fallback when the header doesn't expose tick_rate directly."""
    for key in ("tick_rate", "tickrate", "server_tick_rate"):
        if key in header and header[key]:
            return float(header[key])
    return 64.0  # standard CS2 server tick rate


def _load_tracking():
    """
    Record of demos (keyed by match_id, not filename) we've already turned
    into rows (so re-runs never reparse them) and demos that failed with a
    permanent error like a corrupt/truncated download (so re-runs don't
    keep burning minutes reparsing a file that will fail identically every
    time). Mirrors demo_fetcher.py's `.downloaded_matches.json` pattern.
    """
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                data = json.load(f)
            parsed = {int(m) for m in data.get("parsed", [])}
            failed = {int(m): reason for m, reason in data.get("failed", {}).items()}
            return parsed, failed
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            print(f"Warning: could not read tracking file ({e}), starting fresh.")
    return set(), {}


def _save_tracking(parsed_match_ids, failed_match_ids):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TRACKING_FILE, "w") as f:
        json.dump({
            "parsed": sorted(parsed_match_ids),
            "failed": {str(m): reason for m, reason in failed_match_ids.items()},
        }, f, indent=2)


def _append_rows_to_csv(rows, csv_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=write_header, index=False)


def _discard_rows_for_match_ids(match_ids, csv_path):
    """Remove any already-written rows for `match_ids` from `csv_path`.
    Used when a match turns out to belong to a multi-part recording whose
    other part failed to parse, after some of its rows were already
    appended -- see `_fail_incomplete_series`."""
    if not match_ids or not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    kept = df[~df["match_id"].isin(match_ids)]
    removed = len(df) - len(kept)
    if removed:
        kept.to_csv(csv_path, index=False)
        print(f"  Discarded {removed} previously-written row(s) for match_id(s) {sorted(match_ids)} from {csv_path}.")


def _series_key(demo_file):
    """
    HLTV sometimes splits one continuous recording into several files when
    the server restarts mid-match (e.g. a technical pause) -- named
    "<match>-p1.dem", "<match>-p2.dem", etc, as seen with the
    alka-vs-turma-do-pagode ancient demo. Returns the shared "<match>"
    prefix so all parts of one recording can be found together; a demo
    that isn't part of a split recording just gets its own filename back,
    i.e. it's a "series" of one.
    """
    stem = os.path.splitext(demo_file)[0]
    m = re.match(r"^(.*)-p\d+$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def _fail_incomplete_series(series_map, parsed_match_ids, failed_match_ids):
    """
    Each part of a split recording is parsed as if it were the start of
    its own match -- round numbering, and especially ct_score/t_score, are
    reconstructed by walking events from tick 0 of that file. If an
    earlier part is missing or fails to parse (e.g. it's corrupt), a later
    part's "round 1" is actually some later real round, and its score
    silently comes out wrong (see the alka-vs-turma-do-pagode ancient
    match, where part 2's real first round is round 3, and ct/t_score
    read 0-0 instead of the true score). So: if any part of a multi-part
    demo failed to parse, treat every part as unusable.

    Mutates `parsed_match_ids`/`failed_match_ids` in place. Returns the
    set of match_ids that were newly moved from "parsed" to "failed" by
    this call, so the caller can discard their already-written rows.
    """
    newly_failed = set()
    for key, files in series_map.items():
        if len(files) < 2:
            continue
        match_ids = {_match_id(f): f for f in files}
        if not any(mid in failed_match_ids for mid in match_ids):
            continue  # whole series is fine, or hasn't been attempted yet
        for mid, fname in match_ids.items():
            if mid not in failed_match_ids:
                failed_match_ids[mid] = (
                    f"sibling part of this multi-part demo ({fname}) failed to parse; "
                    f"treating the whole recording as unusable"
                )
            if mid in parsed_match_ids:
                parsed_match_ids.discard(mid)
                newly_failed.add(mid)
    return newly_failed


def main():
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument(
        "--retry-failed", action="store_true",
        help="Also re-attempt demos that previously failed with a permanent error (e.g. corrupt file). "
             "Useful after re-downloading a bad demo.",
    )
    argp.add_argument(
        "--force", action="store_true",
        help="Reparse every demo, ignoring the tracking file entirely (parsed AND failed).",
    )
    args = argp.parse_args()

    demo_paths = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.dem")))
    if not demo_paths:
        print(f"No .dem files found in ./{DEMOS_DIR}. Run demo_fetcher.py first.")
        sys.exit(1)

    parsed_match_ids, failed_match_ids = ([], {}) if args.force else _load_tracking()
    parsed_match_ids = set(parsed_match_ids)

    series_map = {}
    for demo_path in demo_paths:
        series_map.setdefault(_series_key(os.path.basename(demo_path)), []).append(os.path.basename(demo_path))

    # Catch inconsistency left over from a previous run -- e.g. a sibling
    # part was parsed successfully before we learned another part of the
    # same recording is corrupt.
    if not args.force:
        newly_failed = _fail_incomplete_series(series_map, parsed_match_ids, failed_match_ids)
        if newly_failed:
            _discard_rows_for_match_ids(newly_failed, OUTPUT_CSV)
            _discard_rows_for_match_ids(newly_failed, ROUND_OUTPUT_CSV)
            _save_tracking(parsed_match_ids, failed_match_ids)

    total_rows = 0
    total_round_rows = 0
    demos_parsed_this_run = 0
    demos_skipped = 0

    for demo_path in demo_paths:
        demo_file = os.path.basename(demo_path)
        match_id = _match_id(demo_file)

        if not args.force and match_id in parsed_match_ids:
            demos_skipped += 1
            continue
        if not args.force and not args.retry_failed and match_id in failed_match_ids:
            print(f"Skipping {demo_file}: previously failed ({failed_match_ids[match_id]}). Use --retry-failed to retry.")
            demos_skipped += 1
            continue

        try:
            rows, round_rows = parse_demo(demo_path)
        except Exception as e:
            if _is_corrupt_demo_error(e):
                print(f"Failed to parse {demo_path}: file appears corrupt/truncated ({e}). "
                      f"Consider re-downloading it. Won't retry automatically -- use --retry-failed to force.")
            else:
                print(f"Failed to parse {demo_path}: {e}")
            failed_match_ids[match_id] = str(e)
        else:
            _append_rows_to_csv(rows, OUTPUT_CSV)
            _append_rows_to_csv(round_rows, ROUND_OUTPUT_CSV)
            total_rows += len(rows)
            total_round_rows += len(round_rows)
            demos_parsed_this_run += 1

            parsed_match_ids.add(match_id)
            failed_match_ids.pop(match_id, None)

            print(f"  Wrote {len(rows)} row(s) to {OUTPUT_CSV} and {len(round_rows)} row(s) to {ROUND_OUTPUT_CSV} from {demo_file}.")

        # Re-check series consistency after every demo (success or failure):
        # a sibling part may have failed earlier in this same run after this
        # one had already been (re-)parsed successfully, e.g. under
        # --retry-failed with parts processed in order p1, p2 where p1 still
        # fails -- p2's just-written rows need to be discarded too in that
        # case, not just when the failure happens to come first.
        newly_failed = _fail_incomplete_series(series_map, parsed_match_ids, failed_match_ids)
        if newly_failed:
            _discard_rows_for_match_ids(newly_failed, OUTPUT_CSV)
            _discard_rows_for_match_ids(newly_failed, ROUND_OUTPUT_CSV)
        _save_tracking(parsed_match_ids, failed_match_ids)

    print(
        f"\nParsed {demos_parsed_this_run} demo(s) this run "
        f"({total_rows} new row(s) appended to {OUTPUT_CSV}, "
        f"{total_round_rows} new row(s) appended to {ROUND_OUTPUT_CSV}), "
        f"skipped {demos_skipped} already-tracked demo(s)."
    )
    if failed_match_ids:
        print(f"{len(failed_match_ids)} demo(s) currently marked failed (see {TRACKING_FILE}): "
              f"{', '.join(str(m) for m in sorted(failed_match_ids))}")


if __name__ == "__main__":
    main()
