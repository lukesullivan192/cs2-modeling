"""Per-venue plumbing: clients, executors, fee math, venue quirks.

Battle-tested modules MOVE here from ../code/ — they are not
rewritten (DESIGN.md §7). Venue facts that must not be lost:
Polymarket NO orders are priced in YES-space; book responses wrapped
in "marketData"; dust levels (<=1 share) never set BBO/mid/clips;
whole shares only; Kalshi taker fee = ceil(7*P*(1-P))/100; paginated
reads always (unpaginated reads caused phantom reconcile drift).

Planned extensions: Polymarket bulk quotes (markets.list with slug
list), bulk positions, bulk orders.
"""
