"""Record EVERY live CS2 match: HLTV scorebot feeds + Polymarket order books.

Usage: python temp/record_all_live.py [--rounds N]

At start it takes a snapshot of what is live on both sides:
  - HLTV: every match on hltv.org/matches marked live
  - Polymarket: every cs2-tagged event the venue marks live
and records all of them until Ctrl+C. Matches that go live AFTER the start
are not picked up; restart to include them.

Output: one folder per session, data/raw/all_<start-time>/, holding
    hltv_<match_id>.jsonl       {"ts", "event": "scoreboard"|"log"|"fullLog", "data"}  (as live/watchlive.py)
    poly_<event_slug>.jsonl     {"ts", "msg"}                                        (as temp/poly_orderbook.py)
    pairs.json                  which HLTV match the script THINKS each Polymarket event is,
                                with the team-name evidence, plus everything it could not pair
Each stream is its own file so a wrong pairing never mixes data; the pairing is
only a hint for lining files up later by ts. Check pairs.json before trusting it.

Pairing rule: an HLTV team name (accents stripped, lowercased) must appear inside
the Polymarket event title, or vice versa. Abbreviated HLTV names ("LP" for
largadosypelados) only pair through the other team. Ambiguous = unpaired.

--rounds N stops an HLTV stream after N RoundEnd events; the Polymarket streams
run until every HLTV stream has stopped or Ctrl+C.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
import unicodedata
from datetime import datetime

import socketio

CS2_MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cs2_model/
sys.path.insert(0, CS2_MODEL)

from list_live_matches import parse_matches_page          # noqa: E402
from parsing.web import fetch_page, make_session          # noqa: E402
from venues.polymarket.client import PolymarketClient     # noqa: E402

SCOREBOT_URL = "https://scorebot-lb.hltv.org"
RAW_DIR = os.path.join(CS2_MODEL, "data", "raw")
SUBSCRIBE_BATCH = 100      # slugs per subscribe request (venue caps subscriptions per connection)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return s.lower().replace(".", "").strip()


def pair(hltv_live, poly_live):
    """Returns (pairs, unpaired_hltv, unpaired_poly). pairs: [{match_id, event_slug, evidence}]."""
    pairs, used_events, used_matches = [], set(), set()
    for m in hltv_live:
        candidates = []
        for e in poly_live:
            title = norm(e.get("title"))
            evidence = []
            for team in (m.team1, m.team2):
                t = norm(team)
                if len(t) >= 3 and (t in title or any(len(w) >= 3 and w in t for w in title.replace(" vs ", "|").split("|"))):
                    evidence.append(team)
            if evidence:
                candidates.append((len(evidence), e, evidence))
        if not candidates:
            continue
        candidates.sort(key=lambda c: -c[0])
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            continue   # ambiguous: two events match equally well
        _, e, evidence = candidates[0]
        if e["slug"] in used_events:
            continue
        pairs.append({"match_id": m.match_id, "hltv": f"{m.team1} vs {m.team2}",
                      "event_slug": e["slug"], "polymarket": e.get("title"), "evidence": evidence})
        used_events.add(e["slug"]); used_matches.add(m.match_id)
    unpaired_hltv = [f"{m.match_id} {m.team1} vs {m.team2}" for m in hltv_live if m.match_id not in used_matches]
    unpaired_poly = [f"{e['slug']} {e.get('title')}" for e in poly_live if e["slug"] not in used_events]
    return pairs, unpaired_hltv, unpaired_poly


def main() -> None:
    ap = argparse.ArgumentParser(description="Record every live CS2 match: HLTV feed + Polymarket books")
    ap.add_argument("--rounds", type=int, default=0, help="stop each HLTV stream after N RoundEnds (default: Ctrl+C)")
    args = ap.parse_args()

    # ---- what is live
    hltv_live = [m for m in parse_matches_page(fetch_page(make_session(), "/matches")) if m.live]
    print(f"HLTV live: {len(hltv_live)}")
    for m in hltv_live:
        print(f"   {m.match_id}  {m.team1} vs {m.team2}  |  {m.event}")

    venue = PolymarketClient()
    sdk = venue.client
    poly_events, offset = [], 0
    while True:
        r = sdk.events.list({"tagSlug": "cs2", "active": True, "closed": False, "limit": 100, "offset": offset})
        batch = r.get("events") or []
        poly_events += batch
        if len(batch) < 100:
            break
        offset += 100
        time.sleep(0.3)
    poly_live = [e for e in poly_events if e.get("live")]
    print(f"Polymarket live cs2 events: {len(poly_live)} (of {len(poly_events)} active)")
    for e in poly_live:
        print(f"   {e['slug']}  {e.get('title')}")

    pairs, unpaired_hltv, unpaired_poly = pair(hltv_live, poly_live)
    print("\nPAIRED:")
    for p in pairs:
        print(f"   {p['match_id']} {p['hltv']}  <->  {p['event_slug']} ({p['polymarket']})  via {p['evidence']}")
    print("UNPAIRED HLTV:", unpaired_hltv or "none")
    print("UNPAIRED POLYMARKET:", unpaired_poly or "none")
    if not hltv_live and not poly_live:
        sys.exit("nothing live on either side")

    # ---- output folder
    session_dir = os.path.join(RAW_DIR, f"all_{datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(session_dir)
    with open(os.path.join(session_dir, "pairs.json"), "w") as f:
        json.dump({"pairs": pairs, "unpaired_hltv": unpaired_hltv, "unpaired_polymarket": unpaired_poly,
                   "started": datetime.now().isoformat()}, f, indent=2)
    print("\nwriting to", session_dir)

    files, lock = {}, threading.Lock()
    counts = {}

    def write(key: str, record: dict) -> None:
        with lock:
            if key not in files:
                files[key] = open(os.path.join(session_dir, f"{key}.jsonl"), "a")
            files[key].write(json.dumps(record) + "\n")
            files[key].flush()
            counts[key] = counts.get(key, 0) + 1

    # ---- Polymarket: every unresolved market of every live event, routed to the event's file
    slug_to_event = {}
    for e in poly_live:
        ev = (sdk.events.retrieve_by_slug(e["slug"]) or {}).get("event") or {}
        for m in ev.get("markets") or []:
            if m.get("status") != "MARKET_STATUS_RESOLVED":
                slug_to_event[m["slug"]] = e["slug"]
        time.sleep(0.2)
    print(f"Polymarket markets to subscribe: {len(slug_to_event)}")

    # ---- HLTV: one client per live match, same handlers as live/watchlive.py
    stop = threading.Event()
    hltv_done = set()
    clients = []

    def make_client(m):
        key = f"hltv_{m.match_id}"
        state = {"backlog_seen": False, "rounds_done": 0}
        sio = socketio.Client()

        @sio.on("connect")
        def on_connect():
            sio.emit("readyForMatch", json.dumps({"token": "", "listId": str(m.match_id)}))
            print(f"{key}: connected", flush=True)

        @sio.on("scoreboard")
        def on_scoreboard(data):
            write(key, {"ts": time.time(), "event": "scoreboard", "data": data})
            print(f"{key}: {data.get('ctTeamName')} {data.get('counterTerroristScore')} - "
                  f"{data.get('terroristScore')} {data.get('terroristTeamName')} r{data.get('currentRound')} "
                  f"{data.get('currentRoundState')}", flush=True)

        @sio.on("log")
        def on_log(data):
            events = json.loads(data)
            names = [name for ev in events["log"] for name in ev]
            if not state["backlog_seen"]:
                state["backlog_seen"] = True
                print(f"{key}: skipped history log ({len(names)} events)", flush=True)
                return
            write(key, {"ts": time.time(), "event": "log", "data": events})
            print(f"{key}: log {names}", flush=True)
            state["rounds_done"] += names.count("RoundEnd")
            if args.rounds and state["rounds_done"] >= args.rounds:
                print(f"{key}: {state['rounds_done']} rounds, stopping", flush=True)
                hltv_done.add(key)
                sio.disconnect()

        @sio.on("fullLog")
        def on_full_log(data):
            write(key, {"ts": time.time(), "event": "fullLog", "data": json.loads(data)})

        return sio

    for m in hltv_live:
        sio = make_client(m)
        clients.append(sio)
        sio.connect(SCOREBOT_URL)
    # engineio installs its own Ctrl+C handler on connect that disconnects every
    # client and JOINS its polling thread, which never returns from its long-poll
    # read, so Ctrl+C would hang. Put Python's default handler back so Ctrl+C
    # raises KeyboardInterrupt in the main thread as usual.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # ---- Polymarket websocket on the main thread
    def on_message(message):
        ts = time.time()
        md = message.get("marketData") or {}
        event_slug = slug_to_event.get(md.get("marketSlug"))
        key = f"poly_{event_slug}" if event_slug else "poly_unrouted"
        write(key, {"ts": ts, "msg": message})
        if md:
            bids, offers = md.get("bids") or [], md.get("offers") or []
            bb = bids[0]["px"]["value"] if bids else "-"
            ba = offers[0]["px"]["value"] if offers else "-"
            print(f"{key}: {md.get('marketSlug')} bid {bb} ask {ba}", flush=True)
        else:
            print(f"poly: {json.dumps(message)[:160]}", flush=True)

    async def run_polymarket():
        if not slug_to_event:
            while not stop.is_set():
                await asyncio.sleep(0.5)
            return
        ws = sdk.ws.markets()
        ws.on("message", on_message)
        ws.on("error", lambda e: print("poly: WS ERROR:", e, flush=True))
        ws.on("close", lambda: print("poly: WS CLOSED", flush=True))
        await ws.connect()
        slugs = list(slug_to_event)
        for i in range(0, len(slugs), SUBSCRIBE_BATCH):
            await ws.subscribe_market_data(f"orderbook-{i // SUBSCRIBE_BATCH}", slugs[i:i + SUBSCRIBE_BATCH])
            await asyncio.sleep(0.1)
        try:
            while ws.is_connected and not stop.is_set():
                if args.rounds and hltv_live and len(hltv_done) == len(hltv_live):
                    print("all HLTV streams finished their rounds, stopping", flush=True)
                    break
                await asyncio.sleep(0.5)
        finally:
            await ws.close()

    try:
        asyncio.run(run_polymarket())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for sio in clients:
            try:
                sio.eio.disconnect(abort=True)    # abort: don't wait for the stuck polling thread
            except Exception:
                pass
        for f in files.values():
            f.close()
        print(f"\nwrote to {session_dir}:")
        for key, n in sorted(counts.items()):
            print(f"   {key}.jsonl  {n} lines")
        sys.stdout.flush()
        # python-socketio 4's polling thread is non-daemon and blocks in an
        # HTTP long-poll with no timeout even after disconnect(), so a normal
        # exit hangs forever. Files are closed above; leave directly.
        os._exit(0)


if __name__ == "__main__":
    main()
