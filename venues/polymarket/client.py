"""PolymarketClient — read-side VenueClient for Polymarket US.

Wraps the polymarket_us SDK (creds from the repo-root .env; the same
client object later serves the executor's order endpoints). The old
PolymarketInfo's discovery/audit-log duties do NOT move here — that
belongs to the separate ingestion project. This client is quotes,
books, market info, reward programs. Nothing else.

Venue facts baked in (do not lose):
- book/bbo responses are wrapped in "marketData"
- all prices are YES-space value-strings ({"value": "0.04"})
- dust: 0.01-share orders legally rest one tick inside the touch to
  steer mids (observed 2026-07-30) -> Book.dust_min = 1.0 share
- bulk markets.list quotes carry NO sizes -> change detector only
"""
import asyncio
import os
import time
from pathlib import Path

from contracts.market import Book, MarketInfo, Quote, RewardInfo
from venues.base import VenueClient

# .env lives at the repo root (venues/polymarket/<file>.py -> parents[2]).
_ENV = Path(__file__).resolve().parents[2] / ".env"

MIN_BBO_QTY = 1.0    # dust threshold, shares


class PolymarketClient(VenueClient):
    name = "polymarket"

    def __init__(self, client=None):
        if client is None:
            from dotenv import load_dotenv
            from polymarket_us import PolymarketUS
            load_dotenv(_ENV)
            client = PolymarketUS(
                key_id=os.getenv("POLYMARKET_KEY_ID"),
                secret_key=os.getenv("POLYMARKET_SECRET_KEY"))
        self.client = client

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _val(v) -> float | None:
        """Unwrap {"value": "0.04"} / "0.04" / 0.04 -> float."""
        if isinstance(v, dict):
            v = v.get("value")
        if v in (None, ""):
            return None
        return float(v)

    def list_markets(self, symbols: list[str]) -> dict[str, dict]:
        """RAW bulk market records via the markets.list slug filter,
        chunked + self-paced — one request per ~100 slugs. Public on
        purpose: the typed accessors (quotes / market_info) cover the
        common fields, but the full record carries more (outcomePrices,
        marketSides, tags, ...) — use this when you need them."""
        out: dict[str, dict] = {}
        for i in range(0, len(symbols), self.chunk_size):
            if i:
                time.sleep(self.rate_delay)
            chunk = symbols[i:i + self.chunk_size]
            resp = self.client.markets.list(
                {"slug": list(chunk), "limit": len(chunk)})
            for m in resp.get("markets") or []:
                out[m["slug"]] = m
        return out

    # -- VenueClient -----------------------------------------------------

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Bulk BBO — NO SIZES on this endpoint, so these quotes are
        change detectors; decision prices come from book()."""
        return {
            s: Quote(bid=self._val(m.get("bestBidQuote")),
                     ask=self._val(m.get("bestAskQuote")))
            for s, m in self.list_markets(symbols).items()
        }

    def book(self, symbol: str) -> Book:
        raw = (self.client.markets.book(symbol) or {}).get("marketData", {})

        def levels(side):
            out = []
            for lvl in raw.get(side) or []:
                px = self._val(lvl.get("px"))
                qty = lvl.get("qty")
                if px is None or qty in (None, ""):
                    continue
                out.append((px, float(qty)))
            return out

        bids = sorted(levels("bids"), key=lambda l: -l[0])
        asks = sorted(levels("offers"), key=lambda l: l[0])
        return Book(bids=tuple(bids), asks=tuple(asks),
                    dust_min=MIN_BBO_QTY)

    def event_start(self, event_slug: str) -> str | None:
        """The EVENT object's startDate. Market records often leave
        eventStartTime empty (golf, awards) while the event carries it —
        the runner backfills RewardInfo from here so the event guards
        aren't blind (the LIV hole, bitten 2026-08-14)."""
        resp = self.client.get(f"/v1/events/slug/{event_slug}",
                               authenticated=False)
        return (resp.get("event") or {}).get("startDate")

    def market_info(self, symbols: list[str]) -> dict[str, MarketInfo]:
        out: dict[str, MarketInfo] = {}
        for s, m in self.list_markets(symbols).items():
            out[s] = MarketInfo(
                venue=self.name, symbol=s,
                status=m.get("status") or "",
                close_time=m.get("endDate"),
                tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
                min_qty=float(m.get("minimumTradeQty") or 1.0),
                fee_coefficient=(float(m["feeCoefficient"])
                                 if m.get("feeCoefficient") is not None
                                 else None),
                raw=m)
        return out

    def settlement(self, symbol: str) -> float | None:
        """YES-space settlement value of a RESOLVED market, else None.
        Resolved markets vanish from the bulk list endpoint (observed
        2026-08-25: donsay absent from list_markets, retrievable by
        slug with MARKET_STATUS_RESOLVED), so this goes by slug. None
        is the answer on ANY doubt — unresolved, unknown slug, fetch
        error: the caller (settlement retirement) may only erase a
        ledger position on a verified value, and an unverified
        disappearance must stay loud."""
        try:
            d = self.client.markets.retrieve_by_slug(symbol) or {}
            m = d.get("market") or d
            if m.get("status") != "MARKET_STATUS_RESOLVED":
                return None
            s = self.client.markets.settlement(symbol) or {}
            v = s.get("settlement")
            return None if v is None else float(v)
        except Exception:
            return None

    def reward_programs(self, symbols: list[str]) -> dict[str, RewardInfo]:
        """ACTIVE liquidityProgram time period per market (largest pool
        when several are active), with `period` and eventStartTime
        carried — the denominator semantics and the event guard both
        read them (DESIGN.md §3)."""
        out: dict[str, RewardInfo] = {}
        for i in range(0, len(symbols), self.chunk_size):
            if i:
                time.sleep(self.rate_delay)
            resp = self.client.get(
                "/v1/incentives",
                query={"symbols": list(symbols[i:i + self.chunk_size])},
                authenticated=False)
            for m in resp.get("programs") or []:
                active = [p for p in m.get("timePeriods") or []
                          if p.get("status") == "active"
                          and p.get("programType") == "liquidityProgram"]
                if not active:
                    continue
                p = max(active, key=lambda p: float(p["rewardPool"]))
                out[m["marketSlug"]] = RewardInfo(
                    symbol=m["marketSlug"],
                    program_id=p.get("programId") or "",
                    pool=float(p["rewardPool"]),
                    target_size=float(p.get("targetSize") or 0),
                    discount=float(p.get("discountFactor") or 0),
                    period=p.get("period") or "daily_event",
                    start=p.get("start"), end=p.get("end"),
                    event_start_time=m.get("eventStartTime"),
                    raw=m)
        return out

    # -- websocket depth ---------------------------------------------------

    def books_ws(self, symbols: list[str], *, batch_size: int = 100,
                 timeout: float = 30.0, subscribe_pause: float = 0.1, strict: bool = True) -> dict[str, Book]:
        """Full depth for MANY markets over ONE market-data websocket
        connection (wss://api.polymarket.us, signed) instead of one HTTP
        book request per market against the public gateway, whose
        per-market book path is rate-limited far more tightly than the
        bulk list (Cloudflare 1015 bans, 2026-09-12/13).

        Connects, then per batch of `batch_size` slugs: subscribe, keep the
        FIRST marketData snapshot per slug, unsubscribe (the venue caps live
        subscriptions per connection). Same Book shape as book().
        Raises TimeoutError naming the slugs that sent no snapshot within
        `timeout` seconds — a silent gap would look like an empty market."""
        wanted = list(dict.fromkeys(symbols))
        books: dict[str, Book] = {}

        def to_book(payload: dict) -> Book:
            def levels(side):
                out = []
                for lvl in payload.get(side) or []:
                    px = self._val(lvl.get("px"))
                    qty = lvl.get("qty")
                    if px is None or qty in (None, ""):
                        continue
                    out.append((px, float(qty)))
                return out
            bids = sorted(levels("bids"), key=lambda l: -l[0])
            asks = sorted(levels("offers"), key=lambda l: l[0])
            return Book(bids=tuple(bids), asks=tuple(asks), dust_min=MIN_BBO_QTY,
                        ts=time.time())

        async def run():
            # The venue caps live subscriptions per connection ("max
            # subscriptions per connection reached" at ~130 x 100 slugs,
            # 2026-09-13), so each batch is subscribed, drained, and
            # unsubscribed before the next one goes out.
            batch_done = asyncio.Event()
            pending: set = set()
            errors: list = []

            def on_market_data(message):
                payload = message.get("marketData") or {}
                slug = payload.get("marketSlug")
                if slug in pending:
                    books[slug] = to_book(payload)
                    pending.discard(slug)
                    if not pending:
                        batch_done.set()

            def on_error(error):
                errors.append(error)
                batch_done.set()

            ws = self.client.ws.markets()
            ws.on("market_data", on_market_data)
            ws.on("error", on_error)
            ws.on("close", batch_done.set)
            await ws.connect()
            try:
                for n, i in enumerate(range(0, len(wanted), batch_size)):
                    batch = [s for s in wanted[i:i + batch_size] if s not in books]
                    if not batch:
                        continue
                    pending.update(batch)
                    batch_done.clear()
                    request_id = f"books-ws-{n}"
                    await ws.subscribe_market_data(request_id, batch)
                    try:
                        await asyncio.wait_for(batch_done.wait(), timeout)
                    except asyncio.TimeoutError:
                        pass
                    pending.clear()
                    if errors:
                        break
                    await ws.unsubscribe(request_id)
                    if subscribe_pause:
                        await asyncio.sleep(subscribe_pause)
            finally:
                await ws.close()
            if errors:
                raise RuntimeError(f"websocket error: {errors[0]}")

        asyncio.run(run())
        missing = [s for s in wanted if s not in books]
        if missing and strict:                # strict=False: the caller sees the gaps as absent keys
            raise TimeoutError(
                f"no websocket snapshot for {len(missing)} of {len(wanted)} markets "
                f"within {timeout:g}s: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        return books
