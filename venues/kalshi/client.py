"""KalshiClient — read-side VenueClient for Kalshi (public API, no creds).

Quotes and market info both come from the batch endpoint
GET /markets?tickers=a,b,c (100 tickers per request — the pattern that
took the old arb-notebook quote sweep from minutes to seconds). All
prices normalized from the *_dollars string fields to floats.

book(): v1 SYNTHESIZES a one-level book from the batch record's
yes_bid/yes_ask + *_size_fp fields — honest level-1 depth, no dust
threshold (penny-jumper dust is a Polymarket phenomenon). Upgrade path:
/markets/{ticker}/orderbook when depth beyond the touch matters here.
"""
import json
import time
import urllib.parse
import urllib.request

from contracts.market import Book, MarketInfo, Quote
from venues.base import VenueClient

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


class KalshiClient(VenueClient):
    name = "kalshi"

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url

    # -- transport (monkeypatch seam for tests) -------------------------
    def _get_json(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())

    def _fetch_markets(self, tickers: list[str]) -> dict[str, dict]:
        """Batch market records keyed by ticker, chunked + self-paced."""
        out: dict[str, dict] = {}
        for i in range(0, len(tickers), self.chunk_size):
            if i:
                time.sleep(self.rate_delay)
            chunk = tickers[i:i + self.chunk_size]
            resp = self._get_json("/markets", {
                "tickers": ",".join(chunk), "limit": len(chunk)})
            for m in resp.get("markets") or []:
                out[m["ticker"]] = m
        return out

    # -- VenueClient ----------------------------------------------------

    @staticmethod
    def _px(m: dict, field: str) -> float | None:
        """*_dollars string -> float. An EMPTY side comes back as bid
        0.00 / ask 1.00 (no order can legally rest there — min tick is
        0.01) — normalize to None so absence reads as absence, never as
        a tradable price (live-observed 2026-08-03)."""
        v = m.get(f"{field}_dollars")
        if v in (None, ""):
            return None
        px = float(v)
        if px <= 0.0 or px >= 1.0:
            return None
        return px

    @staticmethod
    def _sz(m: dict, field: str) -> float | None:
        v = m.get(f"{field}_size_fp")
        return float(v) if v not in (None, "") else None

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {
            t: Quote(bid=self._px(m, "yes_bid"), ask=self._px(m, "yes_ask"),
                     bid_size=self._sz(m, "yes_bid"),
                     ask_size=self._sz(m, "yes_ask"))
            for t, m in self._fetch_markets(symbols).items()
        }

    def book(self, symbol: str) -> Book:
        m = self._fetch_markets([symbol]).get(symbol)
        if m is None:
            return Book()
        bid, ask = self._px(m, "yes_bid"), self._px(m, "yes_ask")
        bids = ((bid, self._sz(m, "yes_bid") or 0.0),) if bid else ()
        asks = ((ask, self._sz(m, "yes_ask") or 0.0),) if ask else ()
        return Book(bids=bids, asks=asks, dust_min=0.0)

    def settlement(self, symbol: str) -> float | None:
        """YES-space settlement of a settled market, else None — the
        same conservative contract as the polymarket client: None on
        any doubt, because the caller may only retire a ledger position
        on a verified value. `determined` counts (result is final,
        payout pending — the venue may already have zeroed the position
        row); a result other than yes/no maps to None."""
        try:
            m = self._fetch_markets([symbol]).get(symbol) or {}
        except Exception:
            return None
        if m.get("status") not in ("settled", "finalized", "determined"):
            return None
        return {"yes": 1.0, "no": 0.0}.get((m.get("result") or "").lower())

    def market_info(self, symbols: list[str]) -> dict[str, MarketInfo]:
        out: dict[str, MarketInfo] = {}
        for t, m in self._fetch_markets(symbols).items():
            ranges = m.get("price_ranges") or []
            tick = float(ranges[0]["step"]) if ranges else 0.01
            out[t] = MarketInfo(
                venue=self.name, symbol=t,
                status=m.get("status") or "",
                close_time=m.get("close_time"),
                tick_size=tick,
                # fractional trading exists but whole shares are policy
                # (fractional churn, 2026-07-18) — min_qty stays 1
                min_qty=1.0,
                fee_coefficient=None,   # kalshi taker fee is the formula
                                        # ceil(7*P*(1-P))/100, lives with
                                        # the fee math in venues/kalshi
                raw=m)
        return out
        # reward_programs: base {} — Kalshi has no liquidity program.
