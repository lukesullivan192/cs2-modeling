"""Kalshi fee math (moved from the old strategy, 2026-08-03)."""
import math


def taker_fee(price: float) -> float:
    """Kalshi taker fee per contract, in dollars: ceil(7 * P * (1-P))
    cents (the general-market formula, rounded up to the next cent —
    conservative, since Kalshi rounds per order, not per contract).
    Symmetric in P <-> 1-P, so the fee for a NO trade at (1 - yes_price)
    equals the fee computed at yes_price. Hedges cross Kalshi's book, so
    they always pay taker."""
    if not 0 < price < 1:
        raise ValueError(f"price must be in (0, 1), got {price}")
    return math.ceil(7 * price * (1 - price)) / 100
