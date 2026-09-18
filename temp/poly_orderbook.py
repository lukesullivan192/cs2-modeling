"""Record every Polymarket US order-book update for one event over the websocket.

Usage: python temp/poly_orderbook.py [event_slug]
       (default: cs2-vit-furia-2026-09-18)

Find event slugs with the venue's search, e.g. client.client.search.query({"query": "cs2"}).

Every market-data message the venue sends is written as one JSON line to
    cs2_model/data/raw/polymarket_<event_slug>_<start-time>.jsonl
as {"ts": <unix seconds received>, "msg": <message exactly as received>}.
The venue pushes the FULL book (all bid and offer levels) on every change,
so each line is a complete snapshot. Nothing is filtered or reshaped.

Subscribes to every market of the event that is not resolved. Ctrl+C stops.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

CS2_MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # cs2_model/
sys.path.insert(0, CS2_MODEL)

from venues.polymarket.client import PolymarketClient  # noqa: E402

RAW_DIR = os.path.join(CS2_MODEL, "data", "raw")


def main() -> None:
    event_slug = sys.argv[1] if len(sys.argv) > 1 else "cs2-vit-furia-2026-09-18"

    venue = PolymarketClient()                  # signs requests with the .env creds
    sdk = venue.client                          # the polymarket_us SDK underneath

    resp = sdk.events.retrieve_by_slug(event_slug)
    event = resp.get("event") or resp
    markets = event.get("markets") or []
    slugs = [m["slug"] for m in markets if m.get("status") != "MARKET_STATUS_RESOLVED"]
    print(f"{event.get('title')} | start {event.get('startDate')} | "
          f"{len(markets)} markets, {len(slugs)} not resolved")
    for m in markets:
        if m["slug"] in slugs:
            print("  ", m["slug"], "|", m.get("question"))
    if not slugs:
        sys.exit("nothing to subscribe to")

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"polymarket_{event_slug}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    print("writing to", out_path)
    out = open(out_path, "a")
    count = {"n": 0}

    def on_message(message):
        ts = time.time()
        out.write(json.dumps({"ts": ts, "msg": message}) + "\n")
        out.flush()
        count["n"] += 1
        md = message.get("marketData")
        if md:
            bids, offers = md.get("bids") or [], md.get("offers") or []
            bb = bids[0]["px"]["value"] if bids else "-"
            ba = offers[0]["px"]["value"] if offers else "-"
            print(f"{datetime.fromtimestamp(ts):%H:%M:%S.%f} {md.get('marketSlug')} "
                  f"bid {bb} ask {ba} | {len(bids)} bid lvls, {len(offers)} ask lvls", flush=True)
        else:
            print(f"{datetime.fromtimestamp(ts):%H:%M:%S.%f} {json.dumps(message)[:160]}", flush=True)

    async def run():
        ws = sdk.ws.markets()
        ws.on("message", on_message)            # every message, whatever its type
        ws.on("error", lambda e: print("WS ERROR:", e, flush=True))
        ws.on("close", lambda: print("WS CLOSED", flush=True))
        await ws.connect()
        await ws.subscribe_market_data("orderbook", slugs)
        try:
            while ws.is_connected:
                await asyncio.sleep(1)
        finally:
            await ws.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        print(f"\nwrote {count['n']} messages -> {out_path}")


if __name__ == "__main__":
    main()
