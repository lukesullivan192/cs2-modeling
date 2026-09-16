"""
Parses CS2 .dem files into mid-round game-state snapshots for round-outcome
modeling, and writes them to ./data/data.csv.

One output row = one sampled point in time during a live round (after
freezetime ends, before the round ends), labeled with that round's eventual
winner. This is the shape a "given the state right now, who wins the round"
model needs -- as opposed to one row per round or one row per kill.

Requires demoparser2 (https://github.com/LaihoE/demoparser). demoparser2's
tick-level field names have shifted across versions, so field selection is
defensive: we ask for a wishlist of fields and, if the library rejects one,
we parse its own error message to find out which fields it actually supports
and retry with the intersection.
"""
import glob
import os
import re
import sys

import pandas as pd
from demoparser2 import DemoParser

DEMOS_DIR = "demos"
OUTPUT_CSV = "data/data.csv"

# How often (in seconds of game time) to sample state during a live round.
SNAPSHOT_INTERVAL_SECONDS = 5.0

# CS2's C4 fuse time.
BOMB_TIMER_SECONDS = 40.0

TEAM_T = 2
TEAM_CT = 3

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


def _first_present(df, candidates):
    """Return the first column name from `candidates` that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_demo(demo_path):
    """Returns a list of row dicts, one per mid-round snapshot, for a single demo."""
    print(f"Parsing {demo_path}...")
    parser = DemoParser(demo_path)
    demo_file = os.path.basename(demo_path)

    header = parser.parse_header()
    map_name = header.get("map_name", header.get("map", "unknown"))

    round_starts = parser.parse_event("round_start")
    freeze_ends = parser.parse_event("round_freeze_end")
    round_ends = parser.parse_event("round_end")

    if round_ends is None or len(round_ends) == 0:
        print(f"  No round_end events found in {demo_file}, skipping.")
        return []

    winner_col = _first_present(round_ends, ["winner", "winner_side", "team"])
    reason_col = _first_present(round_ends, ["reason", "win_reason"])
    if winner_col is None:
        print(f"  Could not find a winner column in round_end events for {demo_file}, skipping.")
        return []

    # Index round_start/freeze_end ticks by round number so we can line each
    # round_end up with its own start.
    round_ends = round_ends.sort_values("tick").reset_index(drop=True)
    round_starts = round_starts.sort_values("tick").reset_index(drop=True) if round_starts is not None else pd.DataFrame()
    freeze_ends = freeze_ends.sort_values("tick").reset_index(drop=True) if freeze_ends is not None else pd.DataFrame()

    tick_props = _resolve_tick_props(parser, WANTED_TICK_PROPS, probe_tick=int(round_ends["tick"].iloc[0]))
    have_clan_name = "team_clan_name" in tick_props

    bomb_planted_events = _safe_parse_event(parser, "bomb_planted")
    bomb_site_col = _first_present(bomb_planted_events, ["site", "bombsite", "hostage"]) if bomb_planted_events is not None else None

    rows = []
    team_wins = {}  # clan_name -> cumulative rounds won so far (only used if clan names are available)
    ct_score_fallback = 0
    t_score_fallback = 0

    for round_num, round_end_row in round_ends.iterrows():
        round_end_tick = int(round_end_row["tick"])

        # Match this round_end to the closest preceding round_start / freeze_end.
        starts_before = round_starts[round_starts["tick"] <= round_end_tick] if len(round_starts) else pd.DataFrame()
        freezes_before = freeze_ends[freeze_ends["tick"] <= round_end_tick] if len(freeze_ends) else pd.DataFrame()
        if len(starts_before) == 0:
            continue
        round_start_tick = int(starts_before["tick"].iloc[-1])
        freeze_end_tick = int(freezes_before["tick"].iloc[-1]) if len(freezes_before) else round_start_tick

        if round_end_tick <= freeze_end_tick:
            # Degenerate/short round (e.g. warmup or a forfeited round) -- nothing "mid-round" to sample.
            continue

        winner_side = int(round_end_row[winner_col])
        reason = round_end_row[reason_col] if reason_col else None

        # Snapshot player state once at freeze_end to get side->clan_name mapping and starting alive counts.
        try:
            start_state = parser.parse_ticks(tick_props, ticks=[freeze_end_tick])
        except Exception as e:
            print(f"  Skipping round {round_num + 1}: could not read start-of-round state ({e})")
            continue
        if start_state is None or len(start_state) == 0:
            continue

        ct_clan = t_clan = None
        if have_clan_name:
            ct_rows = start_state[start_state["team_num"] == TEAM_CT]
            t_rows = start_state[start_state["team_num"] == TEAM_T]
            ct_clan = ct_rows["team_clan_name"].iloc[0] if len(ct_rows) else None
            t_clan = t_rows["team_clan_name"].iloc[0] if len(t_rows) else None

        # Score entering this round.
        if have_clan_name and ct_clan is not None and t_clan is not None:
            ct_score = team_wins.get(ct_clan, 0)
            t_score = team_wins.get(t_clan, 0)
        else:
            ct_score, t_score = ct_score_fallback, t_score_fallback

        starting_ct_alive = int((start_state["team_num"] == TEAM_CT).sum())
        starting_t_alive = int((start_state["team_num"] == TEAM_T).sum())

        # Bomb plant (if any) within this round.
        plant_tick = None
        bomb_site = None
        if bomb_planted_events is not None and len(bomb_planted_events):
            in_round = bomb_planted_events[
                (bomb_planted_events["tick"] >= freeze_end_tick) & (bomb_planted_events["tick"] <= round_end_tick)
            ]
            if len(in_round):
                plant_tick = int(in_round["tick"].iloc[0])
                bomb_site = in_round[bomb_site_col].iloc[0] if bomb_site_col else None

        tick_rate = header.get("tick_rate") or _infer_tick_rate(header)

        # Sample snapshots at a fixed cadence from freeze_end to round_end.
        step_ticks = max(1, int(SNAPSHOT_INTERVAL_SECONDS * tick_rate))
        sample_ticks = list(range(freeze_end_tick + step_ticks, round_end_tick, step_ticks))
        if not sample_ticks:
            continue

        try:
            snap_df = parser.parse_ticks(tick_props, ticks=sample_ticks)
        except Exception as e:
            print(f"  Skipping round {round_num + 1} snapshots: {e}")
            continue

        for snap_tick, snap in snap_df.groupby("tick"):
            ct = snap[snap["team_num"] == TEAM_CT]
            t = snap[snap["team_num"] == TEAM_T]

            ct_alive = int(ct["is_alive"].sum()) if "is_alive" in ct else len(ct)
            t_alive = int(t["is_alive"].sum()) if "is_alive" in t else len(t)

            row = {
                "demo_file": demo_file,
                "map_name": map_name,
                "round_num": round_num + 1,
                "tick": int(snap_tick),
                "seconds_since_round_start": (int(snap_tick) - round_start_tick) / tick_rate,
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
                "bomb_planted": bool(plant_tick is not None and snap_tick >= plant_tick),
                "bomb_site": bomb_site if (plant_tick is not None and snap_tick >= plant_tick) else None,
                "bomb_time_left": (
                    max(0.0, BOMB_TIMER_SECONDS - (int(snap_tick) - plant_tick) / tick_rate)
                    if (plant_tick is not None and snap_tick >= plant_tick)
                    else None
                ),
                "round_end_reason": reason,
                "winner_side": winner_side,
                "winner_is_ct": int(winner_side == TEAM_CT),
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

    return rows


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


def main():
    demo_paths = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.dem")))
    if not demo_paths:
        print(f"No .dem files found in ./{DEMOS_DIR}. Run demo_fetcher.py first.")
        sys.exit(1)

    all_rows = []
    for demo_path in demo_paths:
        try:
            all_rows.extend(parse_demo(demo_path))
        except Exception as e:
            print(f"Failed to parse {demo_path}: {e}")

    if not all_rows:
        print("No rows parsed from any demo.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows from {len(demo_paths)} demo(s) to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
