"""PolymarketExecutor — Polymarket-US trading client (moved from
../code/helpers/executor_polymarket.py, 2026-08-03, logic unchanged).

Same method surface as the Kalshi executor, minus the auth layer: the
PolymarketUS client signs every request internally.

Difference from Kalshi: Polymarket's create-order has no client-order-id
field, so we can't tag orders with our own uuid. Instead we capture
Polymarket's returned order id onto `order.order_id` at submit time and
match open orders on that.

Credentials (repo-root .env):
    POLYMARKET_KEY_ID
    POLYMARKET_SECRET_KEY

Smoke test (read-only): prints the account balance.
"""
import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv


from venues.order import CapitalAccount, Order, POLYMARKET_STATUS

# .env lives at the repo root (venues/polymarket/<file>.py -> parents[2]).
_REPO = Path(__file__).resolve().parents[2]
load_dotenv(_REPO / ".env")

log = logging.getLogger("exec.poly")

class PolymarketExecutor(CapitalAccount):
    """Minimal Polymarket-US client. Owns its capital ledger
    (CapitalAccount): set `.capital` to a real seed before live trading —
    order() refuses (CapitalError) anything the ledger can't afford.

    `client` is injectable (tests pass a simulator; production shares one
    PolymarketUS client with the read-side PolymarketClient if desired)."""

    def __init__(self, client=None):
        if client is None:
            from polymarket_us import PolymarketUS
            client = PolymarketUS(
                key_id=os.getenv("POLYMARKET_KEY_ID"),
                secret_key=os.getenv("POLYMARKET_SECRET_KEY"),
            )
        self.client = client
        self.orders = {}   # order.id (uuid) -> Order

    def check_balence(self):
        return self.client.account.balances()["balances"]

    def get_open_orders_polymarket(self):
        return self.client.orders.list()["orders"]

    def update_orders(self):
        """Sync every tracked order's status + filled from the venue. Orders
        still on the book come from the open-orders list; ones that left it are
        retrieved individually for their final state (filled vs canceled). Fills
        — partial or full — land on Order.filled, which is what positions and
        hedging are computed from."""
        open_by_id = {o["id"]: o for o in self.get_open_orders_polymarket()}
        for order in self.orders.values():
            if not order.order_id:
                continue    # never acked by the venue — nothing to sync from
            if not order.is_open:
                continue    # already final (filled / canceled): nothing to fetch
            d = open_by_id.get(order.order_id)
            if d is None and order.is_open:
                # Left the book since last sync — fetch its final state. The
                # venue can 404 here (state lag on a fresh order, or a terminal
                # order it already purged): tolerate a few misses, then stop
                # tracking it as live so the strategy can re-quote the level.
                try:
                    d = self.client.orders.retrieve(order.order_id).get("order")
                except Exception as e:
                    misses = order.raw.get("retrieve_misses", 0) + 1
                    order.raw["retrieve_misses"] = misses
                    log.warning("retrieve %s (%s %s) failed %d/3: %s",
                                order.order_id, order.market, order.side, misses, e)
                    if misses >= 3:
                        # Gone at the venue with unknown final state — assume
                        # canceled. If it actually FILLED, that fill is
                        # invisible to us: verify positions on the venue.
                        log.warning("dropping %s as canceled after 3 misses — "
                                    "verify no unseen fill on the venue",
                                    order.order_id)
                        order.status = "canceled"
                    continue
            if d is None:
                continue
            order.filled = float(d.get("cumQuantity") or 0)
            order.status = POLYMARKET_STATUS.get(d.get("state"), order.status)
            self._apply_fee(order, d)
        return self.orders

    def get_open_orders(self):
        self.update_orders()
        return {id: order for id, order in self.orders.items() if order.is_open}

    def order(self, order):
        self.check_capital(order)   # HARD stop: raises before the venue is touched
        body = order.order_to_polymarket_body()
        response = self.client.orders.create(body)

        order.order_id = response["id"]   # Polymarket's id — how we match open orders
        self.orders[order.id] = order
        # LOUD errors: a create can return 200 yet reject the order inside its
        # executions — without this, the order silently never appears on the
        # book and the strategy re-quotes it forever.
        for ex in response.get("executions") or []:
            if ex.get("type") == "EXECUTION_TYPE_REJECTED":
                order.status = "canceled"
                log.warning("polymarket REJECTED %s %s x%s @ %s: %s %s",
                            order.market, order.side, order.count, order.price,
                            ex.get("orderRejectReason", "?"), ex.get("text", ""))
        # The executions also carry the fill: for a fill-or-kill order the
        # outcome is final when create returns, so no retrieve is needed
        # (2026-09-14: ~80 per-order retrieves after one tick got the api host
        # banned by Cloudflare, and the unknown states read as zero fills).
        order.raw["sent_at"] = time.time()
        self._apply_executions(order, response.get("executions") or [])
        # The venue often answers create before the match is attached, so an
        # FOK/IOC order can be left "resting" here: it reserves exactly what its
        # fill would cost, and the caller settles it (see order_synchronous).
        return response

    def record(self, order):
        """Put an Order the venue already has (its order_id known) into the
        ledger without sending anything. For a caller that learned about an
        order after a failed create (see strategy/broker.py)."""
        if not order.order_id:
            raise ValueError("record() needs the venue's order id")
        self.orders[order.id] = order
        return order

    def refresh_order(self, order):
        """Read `order` back from the venue (GET /v1/order/{id}) and set its
        filled and status from what the venue says. One request, no waiting,
        no retry: a NotFoundError (the venue has not indexed a fresh order
        yet) or a timeout propagates to the caller. Raises if the venue
        reports more filled than was sent. Returns the venue's order dict."""
        d = self.client.orders.retrieve(order.order_id).get("order") or {}
        cum = float(d.get("cumQuantity") or 0)
        if cum > order.count + 1e-9:
            raise RuntimeError(f"{order.market} {order.side} {order.order_id}: venue cumQuantity {cum} "
                               f"> sent {order.count}")
        order.filled = cum
        order.status = POLYMARKET_STATUS.get(d.get("state"), order.status)
        self._apply_fee(order, d)
        order.raw["venue_state"] = d.get("state")
        return d

    def order_synchronous(self, order, max_block_time=10):
        """order() with synchronousExecution: the venue holds the request until
        the order reaches a final state (filled, canceled, rejected, expired)
        or `max_block_time` seconds pass, then answers with every execution.
        For a fill-or-kill order that is milliseconds, so the fill (or kill)
        is on the Order when this returns and nothing has to be looked up.
        Same capital check, ledger entry and execution handling as order().
        Status afterwards: "filled", "canceled" (kill, reject, or a partial
        fill with the rest killed: filled > 0), or "resting" only if the
        venue ran out of time, which the caller must then settle.
        Returns the Order."""
        self.check_capital(order)   # HARD stop: raises before the venue is touched
        body = order.order_to_polymarket_body()
        body["synchronousExecution"] = True
        body["maxBlockTime"] = str(int(max_block_time))
        response = self.client.orders.create(body)

        order.order_id = response["id"]
        self.orders[order.id] = order
        for ex in response.get("executions") or []:
            if ex.get("type") == "EXECUTION_TYPE_REJECTED":
                order.status = "canceled"
                order.raw["reject_reason"] = f"{ex.get('orderRejectReason', '?')} {ex.get('text', '')}".strip()
                log.warning("polymarket REJECTED %s %s x%s @ %s: %s",
                            order.market, order.side, order.count, order.price, order.raw["reject_reason"])
        order.raw["sent_at"] = time.time()
        order.raw["response"] = response
        self._apply_executions(order, response.get("executions") or [])
        if order.is_open:
            log.warning("order_synchronous %s %s %s: still %s after %ss; caller must settle it",
                        order.market, order.side, order.order_id, order.status, max_block_time)
        return order

    def close_position(self, market_slug, slippage_tolerance=None, tag="close"):
        """Sell EVERYTHING held in `market_slug` with the venue's close-position
        order (POST /v1/order/close-position), without waiting: the venue
        answers as soon as it has accepted the order, usually before matching.
        If the answer already carries the order object it is booked in the
        ledger exactly as close_position_synchronous books it and the Order
        is returned. If it does not, NOTHING is booked (there is no side,
        price or quantity to book), a warning names the venue order id, and
        None is returned: the caller must retrieve that id and book it, or
        the ledger still holds the shares the venue just sold. Prefer
        close_position_synchronous."""
        body = self._close_position_body(market_slug, slippage_tolerance)
        response = self.client.orders.close_position(body)
        order = self._book_close_position(market_slug, response, tag)
        if order is None:
            log.warning("close-position %s: venue answered id %s with no order object; NOT booked",
                        market_slug, response.get("id"))
        return order

    def close_position_synchronous(self, market_slug, max_block_time=10, slippage_tolerance=None, tag="close"):
        """close_position() with synchronousExecution: waits up to
        `max_block_time` seconds for the order to finish, then books the
        result in the ledger and returns the Order.

        The venue reports the order as a SELL (intent SELL_LONG for held YES,
        SELL_SHORT for held NO) at a YES-space price. The ledger books every
        close as a buy of the other side at that price (sell YES = buy NO), so
        committed() nets it against the held shares exactly as an ordinary
        close order would. No capital check: a close of held shares costs
        nothing; if the ledger did NOT hold the shares the venue sold, the
        entry shows up as a naked buy and verify_portfolio exposes it.

        Raises if the venue answers without an order object (nothing to book:
        the ledger would still hold the shares) or with an intent that is not
        a sell. `slippage_tolerance` is passed through as the venue's object
        ({"currentPrice": {...}, "bips": n, "ticks": n}) when given."""
        body = self._close_position_body(market_slug, slippage_tolerance)
        body["synchronousExecution"] = True
        body["maxBlockTime"] = str(int(max_block_time))
        response = self.client.orders.close_position(body)
        order = self._book_close_position(market_slug, response, tag)
        if order is None:
            raise RuntimeError(f"close-position on {market_slug} answered id {response.get('id')} "
                               f"with no order object; nothing booked, check the venue")
        return order

    @staticmethod
    def _close_position_body(market_slug, slippage_tolerance):
        body = {"marketSlug": market_slug, "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC"}
        if slippage_tolerance:
            body["slippageTolerance"] = slippage_tolerance
        return body

    def _book_close_position(self, market_slug, response, tag):
        """Book the close-position `response` in the ledger as a buy of the
        other side (see close_position_synchronous). Returns the Order, or
        None when the response carries no order object to book from."""
        executions = response.get("executions") or []
        venue_order = next((ex["order"] for ex in reversed(executions) if ex.get("order")), None)
        if venue_order is None:
            return None
        intent = venue_order.get("intent")
        if intent == "ORDER_INTENT_SELL_LONG":
            side = "no"      # sold YES = bought NO
        elif intent == "ORDER_INTENT_SELL_SHORT":
            side = "yes"     # sold NO = bought YES
        else:
            raise RuntimeError(f"close-position on {market_slug} came back with intent {intent}; not booked")
        avg_px = (venue_order.get("avgPx") or {}).get("value")
        price = float(avg_px) if avg_px else float(venue_order["price"]["value"])
        order = Order(venue="polymarket", market=market_slug, side=side, price=price,
                      count=float(venue_order.get("quantity") or 0),
                      filled=float(venue_order.get("cumQuantity") or 0),
                      status=POLYMARKET_STATUS.get(venue_order.get("state"), "resting"),
                      order_id=response.get("id") or venue_order.get("id", ""),
                      raw={"sent_at": time.time(), "response": response},
                      tag=tag, tif=venue_order.get("tif", "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL"))
        self.orders[order.id] = order
        self._apply_executions(order, executions)
        log.info("close-position %s: %s intent, filled %s/%s @ %s (%s)", market_slug, intent,
                 order.filled, order.count, order.price, order.status)
        return order

    def _apply_executions(self, order, executions):
        """Set filled/status from create-response or order-stream executions.
        Each execution carries the venue's order object (cumQuantity, state)
        after that execution; the last one wins."""
        for ex in executions:
            venue_order = ex.get("order") or {}
            if venue_order.get("cumQuantity") is not None:
                order.filled = float(venue_order["cumQuantity"])
            self._apply_fee(order, venue_order)
            state = POLYMARKET_STATUS.get(venue_order.get("state"))
            if state:
                order.status = state
            elif ex.get("type") in ("EXECUTION_TYPE_FILL",):
                order.filled = order.count if not venue_order else order.filled
                order.status = "filled"
            elif ex.get("type") in ("EXECUTION_TYPE_CANCELED", "EXECUTION_TYPE_EXPIRED",
                                    "EXECUTION_TYPE_REJECTED"):
                order.status = "canceled"

    @staticmethod
    def _apply_fee(order, venue_order):
        """The venue's running fee total for the order (commissionNotionalTotalCollected),
        when the venue order object carries it."""
        total = (venue_order.get("commissionNotionalTotalCollected") or {}).get("value")
        if total is not None:
            order.fee = float(total)

    def order_batch(self, orders):
        """Send up to 20 orders in ONE request (POST /v1/orders/batched). The
        response carries only the venue's order ids, in request order; fills
        and rejections arrive on the order stream, so call sync_orders_ws()
        afterwards. Every order is capital-checked first (all raise before
        anything is sent). If any entry fails the venue's shape validation the
        WHOLE batch is rejected, so keep a batch to one basket."""
        if not 1 <= len(orders) <= 20:
            raise ValueError(f"a batch holds 1-20 orders, got {len(orders)}")
        for o in orders:
            self.check_capital(o)
        body = {"orders": [o.order_to_polymarket_body() for o in orders]}
        response = self.client.post("/v1/orders/batched", body=body, authenticated=True)
        ids = response.get("createdOrderIds") or []
        if len(ids) != len(orders):
            raise RuntimeError(f"batched create returned {len(ids)} ids for {len(orders)} orders: {response}")
        for o, venue_id in zip(orders, ids):
            o.order_id = venue_id
            o.status = "resting"          # unknown until the order stream says otherwise
            self.orders[o.id] = o
        return response

    def sync_orders_ws(self, timeout=15.0):
        """Refresh every tracked order's filled/status from the private
        websocket's order snapshot: one connection, no per-order retrieves.
        Returns {venue_order_id: venue order dict} for everything the snapshot
        listed (tracked or not), so a caller can also audit the account."""
        snapshot = self._private_snapshot("orders", timeout)
        by_id = {o["id"]: o for o in snapshot if isinstance(o, dict) and o.get("id")}
        missing = []
        for order in self.orders.values():
            if not order.order_id:
                continue
            d = by_id.get(order.order_id)
            if d is None:
                missing.append(order.order_id)
                continue
            order.filled = float(d.get("cumQuantity") or 0)
            order.status = POLYMARKET_STATUS.get(d.get("state"), order.status)
            order.raw.pop("retrieve_misses", None)
        if missing:
            raise RuntimeError(f"order stream snapshot did not list {len(missing)} tracked orders: {missing[:10]}")
        return by_id

    def positions_ws(self, timeout=15.0):
        """{marketSlug: position dict} from the private websocket's position
        snapshot. Same source of truth as positions(), without pagination."""
        return self._private_snapshot("positions", timeout)

    def _private_snapshot(self, kind, timeout):
        """Connect to the private websocket, subscribe to `kind` ("orders" or
        "positions"), return the snapshot payload once its eof arrives."""
        result = {}

        async def run():
            done = asyncio.Event()
            errors = []
            ws = self.client.ws.private()

            def on_snapshot(message):
                payload = (message.get("orderSubscriptionSnapshot") or message.get("ordersSnapshot")
                           or message.get("positionSubscriptionSnapshot") or message.get("positionsSnapshot") or {})
                if kind == "orders":
                    result.setdefault("data", []).extend(payload.get("orders") or [])
                else:
                    result.setdefault("data", {}).update(payload.get("positions") or {})
                if payload.get("eof", True):
                    done.set()

            def on_error(error):
                errors.append(error)
                done.set()

            ws.on("order_snapshot" if kind == "orders" else "position_snapshot", on_snapshot)
            ws.on("error", on_error)
            ws.on("close", done.set)
            await ws.connect()
            try:
                if kind == "orders":
                    await ws.subscribe_orders("sync-orders")
                else:
                    await ws.subscribe_positions("sync-positions")
                try:
                    await asyncio.wait_for(done.wait(), timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"no {kind} snapshot from the private websocket within {timeout:g}s")
            finally:
                await ws.close()
            if errors:
                raise RuntimeError(f"private websocket error: {errors[0]}")

        asyncio.run(run())
        return result.get("data", [] if kind == "orders" else {})

    def _finalize_cancel(self, order_id):
        """The venue acked a cancel (or the order left the book on its own).
        Mark the tracked order terminal NOW so update_orders stops chasing
        the dead id (noise storm, 2026-08-24) — but FIRST retrieve its final
        state once: a taker can hit the order in the same breath as the
        cancel, and a fill recorded nowhere is a reconcile stop (3 unseen
        shares, 2026-08-25). Retrieve failing = venue already purged a
        no-fill cancel: keep last known fill, stay quiet."""
        for o in self.orders.values():
            if o.order_id != order_id or not o.is_open:
                continue
            try:
                d = self.client.orders.retrieve(order_id).get("order") or {}
            except Exception:
                d = {}
            if d.get("cumQuantity") is not None:
                o.filled = float(d["cumQuantity"])
            status = POLYMARKET_STATUS.get(d.get("state"))
            # a "resting" readback here is venue state-lag, not truth —
            # the cancel was acked, so the order only ends filled/canceled
            o.status = status if status in ("filled", "canceled") else "canceled"

    def cancel(self, order_id, market_slug):
        # Polymarket's cancel needs the market slug alongside the order id.
        # The direct route puts the id in the URL PATH — and on 2026-07-21
        # Cloudflare's WAF false-positived on one specific id
        # (BCJEXG5GJ75R), blocking its cancel/retrieve for 6 hours while the
        # runner retried every tick. Fallback: the market-scoped cancel-all
        # keeps the id out of the URL entirely. It also kills our other
        # resting orders in that market — acceptable: they re-quote next
        # tick, and an uncancelable order is the worse state.
        try:
            resp = self.client.orders.cancel(order_id, {"marketSlug": market_slug})
            self._finalize_cancel(order_id)
            return resp
        except Exception as e:
            # Already off the book (fill/cancel race)? Then the failure is
            # benign — finalize it (it may have FILLED, which is why the
            # cancel missed) and DON'T nuke the market's healthy quotes.
            open_ids = {o["id"] for o in self.get_open_orders_polymarket()}
            if order_id not in open_ids:
                log.warning("direct cancel %s failed (%.120s) but order is no "
                            "longer open — finalizing locally", order_id, e)
                self._finalize_cancel(order_id)
                return None
            log.warning("direct cancel %s blocked (%.120s) — falling back to "
                        "market-scoped cancel_all on %s", order_id, e, market_slug)
            resp = self.client.orders.cancel_all({"marketSlug": market_slug})
            canceled = set(resp.get("canceledOrderIds") or [])
            if order_id not in canceled:
                raise RuntimeError(
                    f"cancel_all on {market_slug} did not cancel {order_id} "
                    f"(canceled: {sorted(canceled)})") from e
            for o in list(self.orders.values()):
                if o.order_id in canceled and o.is_open:
                    self._finalize_cancel(o.order_id)
            return resp

    def cancel_orders_polymarket(self, skip_tags=("unwind",), skip_markets=()):
        """Cancel session-tracked orders still resting on the venue, except
        those tagged in `skip_tags` (unwind legs rest at our profit target on
        purpose) or resting in `skip_markets` (markets the caller is actively
        managing — their quotes are repriced per tick, and canceling a
        correctly priced quote just burns queue position). Returns canceled
        ids. See the risk gate's orphan sweep for usage."""
        canceled = []
        for order in self.get_open_orders().values():
            if order.tag in skip_tags or order.market in skip_markets:
                continue
            if not order.order_id:
                continue    # never acked by the venue — nothing to cancel
            try:
                self.cancel(order.order_id, order.market)
            except Exception as e:
                log.warning("refresh cancel failed (%s %s): %s",
                            order.market, order.order_id, e)
                continue
            # cancel() already finalized status/fill; don't overwrite —
            # a fill captured in the cancel race must stay "filled"
            canceled.append(order.order_id)
        return canceled

    def positions(self):
        # {marketSlug: {netPosition (signed: + YES, - NO), cost: {value}, ...}}.
        # PAGINATED: with enough position rows the endpoint returns a varying
        # subset per call, which read as phantom ReconcileError drift on
        # 2026-07-22 (same market flip-flopping +1/-1 between fetches).
        out: dict = {}
        cursor = None
        for _ in range(50):
            resp = self.client.portfolio.positions(
                {"cursor": cursor} if cursor else None)
            out.update(resp.get("positions") or {})
            cursor = resp.get("nextCursor")
            if not cursor:
                return out
        raise RuntimeError("polymarket positions pagination did not "
                           "terminate after 50 pages")


if __name__ == "__main__":
    print(PolymarketExecutor().check_balence())
