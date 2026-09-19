"""
order.py

One normalized Order shape that both venues collapse into, so the rest of the
code never branches on Kalshi-vs-Polymarket field names. The full API response
is kept on `raw` — model only the fields we act on, read the rest off `raw`.
"""
import datetime
import uuid
from dataclasses import dataclass, field


KALSHI_STATUS = {
    "resting": "resting",
    "executed": "filled",
    "canceled": "canceled",
}


class CapitalError(RuntimeError):
    """An order would spend past the executor's capital seed. Capital is a
    HARD invariant: every BUY commits it at submit time, same-venue netting
    returns it, and any attempt to breach the seed — quote, hedge, or
    unwind alike — crashes rather than trades. In the 2026-07-15 incident
    hedges and unwinds spent unmetered; this check would have stopped the
    loop ~$20 in."""


class CapitalAccount:
    """Capital ledger over one venue's session orders. Mixed into each
    executor: the executor OWNS its capital — it does the addition (every
    submitted order reserves/spends), the subtraction (same-venue YES+NO
    netting hands $1/share straight back), and the enforcement (order()
    refuses to breach the seed by raising CapitalError). Anything else
    (e.g. the runner) reads through the getters, never keeps its own copy.

    The stored number is CASH: what may be spent right now. Every filled buy
    runs it down (shares x price + fee), every close brings it back (the
    netting credit), a reserved open order holds its cost until settled —
    all read from the ledger, so cash = _base - committed() where _base is
    set once (`executor.cash = seed` at a fresh start, or the saved cash on
    resume). `capital` is an alias of `cash` for older callers.

    `cash` defaults to unlimited so bare clients (pnl.py, probes) work; a
    trading runner must set a real number."""

    _base: float = float("inf")

    @property
    def cash(self) -> float:
        """Dollars that may be spent right now."""
        return self._base - self.committed()

    @cash.setter
    def cash(self, value: float) -> None:
        self._base = value + self.committed()

    @property
    def capital(self) -> float:
        return self.cash

    @capital.setter
    def capital(self, value: float) -> None:
        self.cash = value

    def committed(self) -> float:
        """Cash currently spent (fills, plus the fees the venue charged on
        them) or reserved (resting remainders), minus the netting credit:
        YES + NO filled at this venue in the same market nets to a $1/share
        payout the venue returns immediately, so only a losing round trip's
        LOSS stays committed. Computed straight from the tracked orders —
        there is no counter to drift."""
        spent = 0.0
        legs: dict = {}
        for o in self.orders.values():
            px = o.outcome_price
            # An open order is reserved as if it fills: its remainder counts
            # both as cash out AND toward netting, so a pending close of held
            # shares reserves its NET cost (about nothing), the same number
            # check_capital charged it. (2026-09-14: eight unresolved closes
            # booked gross pushed committed 7.84 over capital and killed a run.)
            shares = o.filled + (o.remaining if o.is_open else 0.0)
            spent += shares * px + (max(o.fee, o.expected_fee) if o.is_open else o.fee)
            m = legs.setdefault(o.market, {"yes": 0.0, "no": 0.0})
            m[o.side] += shares
        for m in legs.values():
            spent -= min(m["yes"], m["no"])   # x $1/share returned by netting
        return spent

    def netting_credit(self, order: "Order") -> float:
        """Dollars this order hands straight back by netting against shares
        already held on the OTHER side of the same market ($1/share). A close
        (sell YES = buy NO) therefore costs ~nothing, matching committed()."""
        held = {"yes": 0.0, "no": 0.0}
        for o in self.orders.values():
            if o.market == order.market:
                held[o.side] += o.filled
        other = "no" if order.side == "yes" else "yes"
        unnetted = max(0.0, held[other] - held[order.side])
        return min(order.count, unnetted)

    def available_capital(self) -> float:
        return self.cash

    def check_capital(self, order: "Order") -> None:
        """Raise CapitalError unless `order` (shares x price plus the fee the
        caller expects, minus what it nets against) fits in cash. Called by
        order() before anything reaches the venue."""
        cost = order.count * order.outcome_price + order.expected_fee - self.netting_credit(order)
        if cost <= 1e-9:
            return          # a close: the shares it nets against pay for it, no cash needed
        headroom = self.available_capital()
        if cost > headroom + 1e-9:
            raise CapitalError(
                f"{order.venue} {order.side.upper()} x{order.count} @ {order.price} "
                f"on {order.market} costs {cost:.2f} but cash is {headroom:.2f} "
                f"(committed {self.committed():.2f})")

POLYMARKET_STATUS = {
    "ORDER_STATE_OPEN": "resting",
    "ORDER_STATE_NEW": "resting",
    "ORDER_STATE_PENDING_NEW": "resting",
    "ORDER_STATE_PARTIALLY_FILLED": "resting",   # still on the book
    "ORDER_STATE_FILLED": "filled",
    "ORDER_STATE_CANCELED": "canceled",
    "ORDER_STATE_EXPIRED": "canceled",
    "ORDER_STATE_REJECTED": "canceled",
    "ORDER_STATE_REPLACED": "canceled",
}


@dataclass
class Order:
    venue: str          # "kalshi" | "polymarket"
    market: str         # ticker | slug
    side: str           # "yes" | "no"
    price: float        # YES-space venue price — BOTH venues quote every order
                        # (incl. NO buys / BUY_SHORT) in YES-space. Cash per
                        # contract for a NO buy is 1 - price.
    count: float
    filled: float
    status: str         # "resting" | "filled" | "canceled"
    order_id: str
    raw: dict = field(repr=False)
    expiration_time: int = None                                  # unix seconds, None = rests until canceled
    id: str = field(default_factory=lambda: str(uuid.uuid4()))   # our own id
    tag: str = ""       # strategy label (e.g. quote tier); session-only, never sent to a venue
    fee: float = 0.0    # cash the venue has charged on this order's fills so far (from the
                        # venue's commissionNotionalTotalCollected); real money out, so
                        # committed() counts it
    expected_fee: float = 0.0   # what the caller expects the venue to charge on the whole
                                # order (its fee model); reserved by check_capital and
                                # committed() until the venue's actual fee replaces it
    tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL"   # Polymarket time-in-force; FOK for
                                                  # basket legs (2026-09-13); GTD when
                                                  # expiration_time is set

    @property
    def outcome_price(self):
        """Cash per contract of the outcome bought (price is YES-space on
        both venues, so a NO buy costs the complement)."""
        return self.price if self.side == "yes" else round(1 - self.price, 4)

    @property
    def remaining(self):
        return self.count - self.filled

    @property
    def is_open(self):
        return self.status == "resting"

    @property
    def is_filled(self):
        return self.status == "filled"

    def order_to_kalshi_body(self, time_in_force="good_till_canceled",
                             self_trade_prevention_type="maker"):
        body = {
            "ticker": self.market,
            "client_order_id": self.id,
            "side": "bid" if self.side == "yes" else "ask",   # bid = buy YES, ask = sell YES
            "count": f"{self.count:.2f}",
            "price": f"{self.price:.2f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
        }
        if self.expiration_time is not None:
            body["expiration_time"] = self.expiration_time
        return body

    def _poly_price_str(self) -> str:
        """YES-space price as the venue's decimal string. Golf/awards
        longshots tick at 0.001 — the old ':.2f' turned a 0.003 bid into
        '0.00' and the venue rejected every attempt (2026-08-14). Keep
        the proven 2-decimal form for cent prices; extend only when the
        price actually has sub-penny digits."""
        s = f"{self.price:.4f}"
        while s.endswith("0") and len(s.split(".")[1]) > 2:
            s = s[:-1]
        return s

    def order_to_polymarket_body(self):
        # Polymarket US expects the YES-space price for EVERY intent, including
        # BUY_SHORT — self.price is already YES-space, so it passes through.
        # (Sending the NO-space price here made shorts cross our own YES bid
        # and get killed asynchronously by self-trade prevention.)
        body = {
            "marketSlug": self.market,
            "intent": "ORDER_INTENT_BUY_LONG" if self.side == "yes" else "ORDER_INTENT_BUY_SHORT",
            "type": "ORDER_TYPE_LIMIT",
            "price": {"value": self._poly_price_str(), "currency": "USD"},
            # Fractional shares: minimumTradeQty is 0.01 on ~all markets (snapshot
            # 2026-09-12); step 4 sent float quantities with FOK successfully.
            "quantity": round(float(self.count), 2),
            "tif": self.tif,
        }
        if self.expiration_time is not None:
            gtt = datetime.datetime.fromtimestamp(self.expiration_time, datetime.timezone.utc)
            body["tif"] = "TIME_IN_FORCE_GOOD_TILL_DATE"
            body["goodTillTime"] = gtt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return body

    @classmethod
    def from_kalshi(cls, d):
        side = d["outcome_side"]
        price = d["yes_price_dollars"]   # Order.price is always YES-space
        return cls(
            venue="kalshi",
            market=d["ticker"],
            side=side,
            price=float(price),
            count=float(d["initial_count_fp"]),
            filled=float(d["fill_count_fp"]),
            status=KALSHI_STATUS[d["status"]],
            order_id=d["order_id"],
            raw=d,
        )

    @classmethod
    def from_polymarket(cls, d):
        side = "yes" if d["outcomeSide"] == "OUTCOME_SIDE_YES" else "no"
        filled = float(d.get("cumQuantity") or 0)
        avg_px = (d.get("avgPx") or {}).get("value")
        price = float(avg_px) if avg_px else float(d["price"]["value"])
        return cls(
            venue="polymarket",
            market=d["marketSlug"],
            side=side,
            price=price,
            count=float(d["quantity"]),
            filled=filled,
            status=POLYMARKET_STATUS[d["state"]],
            order_id=d["id"],
            raw=d,
        )
