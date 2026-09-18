"""Watch a live HLTV match: print and save every raw scorebot frame as it arrives.

Usage: python live/watch_live.py <match_id> [--scrape] [--quiet]

Each frame is appended to data/raw/live_<match_id>_<start-time>.jsonl in the
same {"ts", "type", "data"} shape the historical parser emits, so the capture
can be replayed through the same code as parsed demos. The full payload is
also pretty-printed to stdout (--quiet prints a one-line summary instead).

Ctrl+C stops it. --scrape resolves the scorebot list id from the match page
if the server answers the bare match id with blank scoreboards.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.scorebot import (DEFAULT_SCOREBOT_URL, ScorebotClient,  # noqa: E402
                           fetch_match_connection_info)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")


class FrameRecorder(ScorebotClient):
    def __init__(self, scorebot_url: str, list_id: str, out_file: str, quiet: bool):
        super().__init__(scorebot_url, list_id, raw=True)
        self.out_file = out_file
        self.quiet = quiet
        self.n = 0

    def _dump(self, kind, data):
        frame = {"ts": time.time(), "type": kind, "data": data}
        with open(self.out_file, "a") as f:
            f.write(json.dumps(frame) + "\n")
        self.n += 1

        stamp = datetime.fromtimestamp(frame["ts"]).strftime("%H:%M:%S.%f")[:-3]
        print(f"\n===== frame {self.n} | {stamp} | type={kind} =====")
        if self.quiet:
            print(self._summary(kind, data))
        else:
            print(json.dumps(data, indent=2))
        sys.stdout.flush()

    @staticmethod
    def _summary(kind, data) -> str:
        if kind == "scoreboard":
            alive = {s: sum(1 for p in data.get(s, []) if p.get("alive")) for s in ("CT", "TERRORIST")}
            return (f"{data.get('ctTeamName')} {data.get('counterTerroristScore')} - "
                    f"{data.get('terroristScore')} {data.get('terroristTeamName')} | "
                    f"map={data.get('mapName')} round={data.get('currentRound')} "
                    f"state={data.get('currentRoundState')} alive={alive['CT']}v{alive['TERRORIST']} "
                    f"bomb={data.get('bombPlanted')} equip={data.get('ctInitialEquipmentValue')}/"
                    f"{data.get('tInitialEquipmentValue')}")
        if kind == "log":
            names = [name for ev in data for name in ev]
            head = ", ".join(names[:20])
            return f"{len(names)} events: {head}" + (", ..." if len(names) > 20 else "")
        if kind == "fullLog":
            return f"{len(data)} historical events"
        return json.dumps(data)[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description="Print and save raw HLTV scorebot frames for a live match")
    ap.add_argument("match_id", type=int, help="HLTV match id (hltv.org/matches/<id>/...)")
    ap.add_argument("--scrape", action="store_true", help="resolve list id by scraping the match page")
    ap.add_argument("--quiet", action="store_true", help="one-line summary per frame instead of full JSON")
    args = ap.parse_args()

    if args.scrape:
        scorebot_url, list_id = fetch_match_connection_info(args.match_id)
    else:
        scorebot_url, list_id = DEFAULT_SCOREBOT_URL, str(args.match_id)

    os.makedirs(RAW_DIR, exist_ok=True)
    out_file = os.path.join(RAW_DIR, f"live_{args.match_id}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    print(f"saving frames to {out_file}", file=sys.stderr)

    client = FrameRecorder(scorebot_url, list_id, out_file, args.quiet)
    try:
        client.run()
    except KeyboardInterrupt:
        client.stop()
    print(f"\nstopped after {client.n} frames -> {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
