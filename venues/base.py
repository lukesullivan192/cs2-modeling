"""VenueClient — the abstract read-side surface every venue implements.

Adding a venue (e.g. Polymarket International) = subclass this, place
it in venues/<name>/client.py, normalize everything to the contracts
(YES-space dollars, floats 0-1), and nothing upstream changes. The
strategy and runner only ever see this interface.

Method contract (identical across venues):

    quotes(symbols)          bulk BBO, one venue round-trip per ~100
                             symbols. Size-less results (venue bulk
                             endpoints without depth) are CHANGE
                             DETECTORS only — see contracts.Quote.
    book(symbol)             dust-aware depth for ONE market; the only
                             source of decision prices (mid, clips,
                             touch).
    market_info(symbols)     venue-fed per-market economics (fees, tick
                             size, min qty, status, close time). Never
                             hardcode any of these.
    reward_programs(symbols) ACTIVE liquidity-program config per market.
                             Base returns {} — venues without incentive
                             programs (Kalshi) simply don't override.

Clients self-pace between chunked requests (`rate_delay`) — the
2026-07-21 Cloudflare ban was one unthrottled 200-request burst. The
runner's risk gate paces across calls; the client paces within one.
"""
import time
from abc import ABC, abstractmethod

from contracts.market import Book, MarketInfo, Quote, RewardInfo  # noqa: F401


class VenueClient(ABC):
    name: str = ""                # "kalshi", "polymarket", ...
    rate_delay: float = 0.25      # seconds between chunked requests
    chunk_size: int = 100         # symbols per bulk request

    @abstractmethod
    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Bulk BBO for `symbols`. Missing/dead markets are absent from
        the result — absence is data, never a zero-filled row."""

    @abstractmethod
    def book(self, symbol: str) -> Book:
        """Depth for one market, YES-space, best-first, with the
        venue's dust threshold baked into the Book."""

    @abstractmethod
    def market_info(self, symbols: list[str]) -> dict[str, MarketInfo]:
        """Venue-fed per-market economics for `symbols`."""

    def reward_programs(self, symbols: list[str]) -> dict[str, RewardInfo]:
        """ACTIVE liquidity-program per market. Default: no programs."""
        return {}

    def books(self, symbols: list[str],
              progress=None) -> dict[str, Book]:
        """Depth for MANY markets. Neither venue has a bulk books
        endpoint (probed live 2026-08-03 — markets.list carries quotes
        only), so the default is a paced per-symbol loop: one request
        per market, `rate_delay` apart. Use for the slow-tick full
        refresh; fast ticks should book() only the markets whose bulk
        quote moved. A failed market is ABSENT from the result (logged
        by the venue transport), never an empty Book.

        `progress`, when given, is called as progress(done, total,
        symbol) before each fetch and once more as progress(total,
        total, "") when the sweep finishes — the runner renders it as
        an in-place progress bar."""
        out: dict[str, Book] = {}
        total = len(symbols)
        for i, s in enumerate(symbols):
            if i:
                time.sleep(self.rate_delay)
            if progress is not None:
                progress(i, total, s)
            try:
                out[s] = self.book(s)
            except Exception:
                continue        # absence is the signal; caller decides
        if progress is not None and total:
            progress(total, total, "")
        return out
