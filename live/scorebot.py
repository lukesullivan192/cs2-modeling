"""HLTV Scorebot client.

Connects to HLTV's live match feed (the "Scorebot") and streams scoreboard
and event-log updates for a match.

Protocol (reverse-engineered from gigobyte/HLTV and andrewda/hltv-livescore):
  1. The match page (hltv.org/matches/<id>/<slug>) embeds the scorebot server
     URL and the match's "list id" in the #scoreboardElement element as
     data-scorebot-url and data-scorebot-id attributes.
  2. Connect to that URL with Socket.IO (old 2.x protocol / Engine.IO v3,
     hence the python-socketio<5 requirement).
  3. Emit "readyForMatch" with the JSON string '{"token": "", "listId": <id>}'.
  4. The server pushes:
       - "scoreboard": full match state (teams, scores, per-player HP/money/
         stats, round history, bomb state) as a dict
       - "log":        incremental events (Kill, RoundStart, RoundEnd,
         BombPlanted, BombDefused, ...) as a JSON string
       - "fullLog":    the complete event log so far, as a JSON string

Usage:
    python scorebot.py <match_id>              # match id from the hltv.org URL
    python scorebot.py <match_id> --raw        # dump raw payloads as JSONL
    python scorebot.py <match_id> --scrape     # resolve listId from the match page

By default the match id is sent directly as the listId (the server accepted
this in testing, but it answers unknown listIds with a blank scoreboard, so
if a live match streams nothing, retry with --scrape). --scrape needs
cloudscraper because hltv.org sits behind Cloudflare bot protection.

Dependencies:
    pip install "python-socketio[client]<5"    # server speaks Engine.IO v3
    pip install cloudscraper                   # only for --scrape
"""

import argparse
import json
import re
import sys
import time

import socketio

# Verified alive 2026-09-08 (EIO3 handshake OK). The old host from
# hltv-livescore (scorebot-secure.hltv.org) no longer resolves.
DEFAULT_SCOREBOT_URL = "https://scorebot-lb.hltv.org"

# Old markup (gigobyte/HLTV era) and current markup respectively; the id
# attribute only appears while a match is live.
SCOREBOT_URL_RE = re.compile(r'data-(?:scorebot-url|livescore-server-url)="([^"]+)"')
SCOREBOT_ID_RE = re.compile(r'data-scorebot-id="([^"]+)"')


class ScorebotPageError(RuntimeError):
    """Raised when the match page can't be fetched or parsed."""


def fetch_match_connection_info(match_id: int) -> tuple[str, str]:
    """Scrape the HLTV match page for the scorebot URL and list id."""
    try:
        import cloudscraper
    except ImportError as e:
        raise ScorebotPageError(
            "cloudscraper is required to scrape the match page "
            "(pip install cloudscraper), or pass --list-id/--url directly"
        ) from e

    # any non "-" slug redirects to the canonical URL; "-" itself gets a 403
    url = f"https://www.hltv.org/matches/{match_id}/x"
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(url, timeout=30)
    if resp.status_code != 200:
        raise ScorebotPageError(
            f"Failed to fetch {url}: HTTP {resp.status_code}. "
            "HLTV's Cloudflare protection may be blocking this request; "
            "grab data-scorebot-url/data-scorebot-id from the page source in "
            "a browser and pass them via --url/--list-id."
        )

    url_match = SCOREBOT_URL_RE.search(resp.text)
    id_match = SCOREBOT_ID_RE.search(resp.text)
    if not url_match or not id_match:
        raise ScorebotPageError(
            f"Match page {url} has no scorebot id attribute. The match is "
            "probably not live (the id is only embedded while a match is "
            "live). Retry once it starts, or pass --list-id if you know it."
        )

    # data-scorebot-url can be a comma-separated list; the JS client uses the last one
    scorebot_url = url_match.group(1).split(",")[-1]
    list_id = id_match.group(1)
    return scorebot_url, list_id


class ScorebotClient:
    """Streams scoreboard/log updates for one match.

    Override the on_* methods or pass raw=True to just dump JSONL to stdout.
    """

    def __init__(self, scorebot_url: str, list_id: str, raw: bool = False):
        self.scorebot_url = scorebot_url
        self.list_id = str(list_id)
        self.raw = raw
        self.sio = socketio.Client(reconnection=True, logger=False)

        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        self.sio.on("scoreboard", self._on_scoreboard)
        self.sio.on("log", self._on_log)
        self.sio.on("fullLog", self._on_full_log)

    def run(self) -> None:
        print(f"Connecting to {self.scorebot_url} (listId={self.list_id})", file=sys.stderr)
        self.sio.connect(self.scorebot_url)
        self.sio.wait()

    def stop(self) -> None:
        self.sio.disconnect()

    # -- socket plumbing ------------------------------------------------

    def _on_connect(self):
        print("Connected, sending readyForMatch", file=sys.stderr)
        self.sio.emit("readyForMatch", json.dumps({"token": "", "listId": self.list_id}))

    def _on_disconnect(self):
        print("Disconnected", file=sys.stderr)

    def _on_scoreboard(self, data):
        if self.raw:
            self._dump("scoreboard", data)
        else:
            self.on_scoreboard(data)

    def _on_log(self, data):
        events = json.loads(data)["log"]
        if self.raw:
            self._dump("log", events)
        else:
            for event in events:
                for name, payload in event.items():
                    self.on_event(name, payload)

    def _on_full_log(self, data):
        events = json.loads(data)
        if self.raw:
            self._dump("fullLog", events)
        else:
            self.on_full_log(events)

    @staticmethod
    def _dump(kind, data):
        print(json.dumps({"ts": time.time(), "type": kind, "data": data}), flush=True)

    # -- override these -------------------------------------------------

    def on_scoreboard(self, sb: dict) -> None:
        """Full match state. Fired on connect and after most events."""
        ct, t = sb.get("ctTeamName", "CT"), sb.get("terroristTeamName", "T")
        print(
            f"[scoreboard] {ct} {sb.get('counterTerroristScore')} - "
            f"{sb.get('terroristScore')} {t} | map={sb.get('mapName')} "
            f"round={sb.get('currentRound')} live={sb.get('live')} "
            f"bomb_planted={sb.get('bombPlanted')}"
        )

    def on_event(self, name: str, e: dict) -> None:
        """One in-game event from the incremental log."""
        if name == "Kill":
            flash = f" (flashed by {e['flasherNick']})" if e.get("flasherNick") else ""
            hs = " HS" if e.get("headShot") else ""
            print(
                f"[kill] {e['killerNick']} ({e['killerSide']}) -> "
                f"{e['victimNick']} ({e['victimSide']}) with {e['weapon']}{hs}{flash}"
            )
        elif name == "RoundStart":
            print("[round] --- round start ---")
        elif name == "RoundEnd":
            print(
                f"[round] end: {e['winner']} wins ({e['winType']}) | "
                f"CT {e['counterTerroristScore']} - {e['terroristScore']} T"
            )
        elif name == "BombPlanted":
            print(f"[bomb] planted by {e['playerNick']} ({e['ctPlayers']}v{e['tPlayers']})")
        elif name == "BombDefused":
            print(f"[bomb] defused by {e['playerNick']}")
        elif name == "MatchStarted":
            print(f"[match] started on {e.get('map')}")
        else:
            print(f"[{name}] {json.dumps(e)}")

    def on_full_log(self, events: list) -> None:
        print(f"[fullLog] {len(events)} historical events received", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream HLTV Scorebot data for a live match")
    parser.add_argument("match_id", nargs="?", type=int,
                        help="HLTV match id (from hltv.org/matches/<id>/...)")
    parser.add_argument("--list-id", help="explicit scorebot list id")
    parser.add_argument("--url", default=DEFAULT_SCOREBOT_URL,
                        help=f"scorebot server URL (default: {DEFAULT_SCOREBOT_URL})")
    parser.add_argument("--scrape", action="store_true",
                        help="resolve listId (and server URL) by scraping the match page")
    parser.add_argument("--raw", action="store_true",
                        help="dump raw payloads as JSONL to stdout (pipe to a file for capture)")
    args = parser.parse_args()

    if args.list_id:
        scorebot_url, list_id = args.url, args.list_id
    elif args.match_id and args.scrape:
        scorebot_url, list_id = fetch_match_connection_info(args.match_id)
    elif args.match_id:
        scorebot_url, list_id = args.url, str(args.match_id)
    else:
        parser.error("provide a match_id or --list-id")

    client = ScorebotClient(scorebot_url, list_id, raw=args.raw)
    try:
        client.run()
    except KeyboardInterrupt:
        client.stop()


if __name__ == "__main__":
    main()
