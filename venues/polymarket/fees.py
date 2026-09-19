"""Polymarket US fee math.

TAKER fee only — resting limit orders (makers) pay NOTHING on either
end (Nick, 2026-08-04), which is why the flatten exit bound has no fee
term; this curve prices taker paths (hedge covers, future graph routes
through Poly). Mirrors Kalshi's curve: coefficient * P * (1 - P) per
share (Nick, 2026-08-03, per venue docs). The coefficient is VENUE-FED
per market (MarketInfo.fee_coefficient from markets.list, e.g. 0.06) —
never hardcoded. No rounding step: Kalshi's ceil-to-cent is a
documented Kalshi behavior; none is documented here."""


def taker_fee(price: float, coefficient: float) -> float:
    """Fee per share for a taker trade at outcome price `price`."""
    if not 0 < price < 1:
        raise ValueError(f"price must be in (0, 1), got {price}")
    if coefficient < 0:
        raise ValueError(f"coefficient must be >= 0, got {coefficient}")
    return coefficient * price * (1 - price)
