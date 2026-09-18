"""Record the HLTV scorebot feed and the Polymarket order book at the same time.

Usage: python temp/record_both.py <hltv_match_id> <polymarket_event_slug> [--rounds N]

    python temp/record_both.py 2398102 cs2-vit-furia-2026-09-18

Both feeds go into ONE file, data/raw/both_<match_id>_<event_slug>_<start-time>.jsonl,
one JSON line per message, each stamped with the wall-clock time it arrived so the
two streams can be lined up:

    {"ts": <unix seconds>, "source": "hltv",       "event": "scoreboard"|"log"|"fullLog", "data": <payload>}
    {"ts": <unix seconds>, "source": "polymarket", "msg": <websocket message untouched>}

HLTV side: same client and handlers as live/watchlive.py. The first "log" after
subscribing is the match history and is skipped; everything after it is kept.
With --rounds N the script stops after N RoundEnd events; without it, it runs
until Ctrl+C. Polymarket side: same as temp/poly_orderbook.py, every market of
the event that is not resolved, full book on every change.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

import socketio

CS2_MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cs2_model/
sys.path.insert(0, CS2_MODEL)

from venues.polymarket.client import PolymarketClient  # noqa: E402

SCOREBOT_URL = "https://scorebot-lb.hltv.org"
RAW_DIR = os.path.join(CS2_MODEL, "data", "raw")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record HLTV scorebot + Polymarket order book together")
    ap.add_argument("match_id", help="HLTV match id")
    ap.add_argument("event_slug", help="Polymarket event slug, e.g. cs2-vit-furia-2026-09-18")
    ap.add_argument("--rounds", type=int, default=0, help="stop after N RoundEnd events (default: run until Ctrl+C)")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"both_{args.match_id}_{args.event_slug}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    out = open(out_path, "a")
    lock = threading.Lock()          # hltv writes from socketio's thread, polymarket from the main thread
    count = {"hltv": 0, "polymarket": 0}
    stop = threading.Event()

    def write(record: dict) -> None:
        with lock:
            out.write(json.dumps(record) + "\n")
            out.flush()
        count[record["source"]] += 1

    # ---- Polymarket: look up the event's markets first so a bad slug fails before anything connects
    venue = PolymarketClient()
    sdk = venue.client
    resp = sdk.events.retrieve_by_slug(args.event_slug)
    event = resp.get("event") or resp
    markets = event.get("markets") or []
    slugs = [m["slug"] for m in markets if m.get("status") != "MARKET_STATUS_RESOLVED"]
    print(f"polymarket: {event.get('title')} | {len(markets)} markets, {len(slugs)} not resolved")
    for m in markets:
        if m["slug"] in slugs:
            print("  ", m["slug"], "|", m.get("question"))
    if not slugs:
        sys.exit("no unresolved polymarket markets to subscribe to")
    print("writing to", out_path)

    # ---- HLTV: same as live/watchlive.py
    state = {"backlog_seen": False, "rounds_done": 0}
    sio = socketio.Client()

    @sio.on("connect")
    def on_connect():
        print("hltv: connected via", sio.transport(), flush=True)
        sio.emit("readyForMatch", json.dumps({"token": "", "listId": args.match_id}))

    @sio.on("disconnect")
    def on_disconnect():
        print("hltv: disconnected", flush=True)

    @sio.on("scoreboard")
    def on_scoreboard(data):
        write({"ts": time.time(), "source": "hltv", "event": "scoreboard", "data": data})
        print(f"hltv: scoreboard {data.get('ctTeamName')} {data.get('counterTerroristScore')} - "
              f"{data.get('terroristScore')} {data.get('terroristTeamName')} | round {data.get('currentRound')} "
              f"{data.get('currentRoundState')}", flush=True)

    @sio.on("log")
    def on_log(data):
        events = json.loads(data)
        names = [name for ev in events["log"] for name in ev]
        if not state["backlog_seen"]:
            state["backlog_seen"] = True     # match history, not live; skip it
            print(f"hltv: skipped history log ({len(names)} events)", flush=True)
            return
        write({"ts": time.time(), "source": "hltv", "event": "log", "data": events})
        print("hltv: log", names, flush=True)
        state["rounds_done"] += names.count("RoundEnd")
        if args.rounds and state["rounds_done"] >= args.rounds:
            print(f"hltv: {state['rounds_done']} rounds recorded, stopping", flush=True)
            stop.set()

    @sio.on("fullLog")
    def on_full_log(data):
        write({"ts": time.time(), "source": "hltv", "event": "fullLog", "data": json.loads(data)})
        print("hltv: fullLog received", flush=True)

    # ---- Polymarket: same as temp/poly_orderbook.py
    def on_message(message):
        ts = time.time()
        write({"ts": ts, "source": "polymarket", "msg": message})
        md = message.get("marketData")
        if md:
            bids, offers = md.get("bids") or [], md.get("offers") or []
            bb = bids[0]["px"]["value"] if bids else "-"
            ba = offers[0]["px"]["value"] if offers else "-"
            print(f"poly: {md.get('marketSlug')} bid {bb} ask {ba} | {len(bids)} bid lvls, {len(offers)} ask lvls", flush=True)
        else:
            print(f"poly: {json.dumps(message)[:160]}", flush=True)

    async def run_polymarket():
        ws = sdk.ws.markets()
        ws.on("message", on_message)
        ws.on("error", lambda e: print("poly: WS ERROR:", e, flush=True))
        ws.on("close", lambda: print("poly: WS CLOSED", flush=True))
        await ws.connect()
        await ws.subscribe_market_data("orderbook", slugs)
        try:
            while ws.is_connected and not stop.is_set():
                await asyncio.sleep(0.5)
        finally:
            await ws.close()

    sio.connect(SCOREBOT_URL)              # socketio polls on its own background thread
    signal.signal(signal.SIGINT, signal.default_int_handler)   # undo engineio's blocking Ctrl+C handler
    try:
        asyncio.run(run_polymarket())      # polymarket runs here on the main thread
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sio.eio.disconnect(abort=True)      # don't wait for the stuck polling thread
        out.close()
        print(f"\nwrote {count['hltv']} hltv + {count['polymarket']} polymarket lines -> {out_path}")
        sys.stdout.flush()
        os._exit(0)     # socketio's polling thread never exits on its own (see record_all_live.py)


if __name__ == "__main__":
    main()
