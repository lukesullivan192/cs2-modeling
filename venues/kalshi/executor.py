"""KalshiExecutor — authenticated Kalshi trading client (moved from
../code/helpers/executor_kalshi.py, 2026-08-03, logic unchanged).

Auth (per Kalshi docs):
    Base URL : https://external-api.kalshi.com/trade-api/v2
    Headers  : KALSHI-ACCESS-KEY        = api key id
               KALSHI-ACCESS-TIMESTAMP  = ms since epoch
               KALSHI-ACCESS-SIGNATURE  = base64( RSA-PSS-SHA256( ts + METHOD + path ) )
    The signed `path` includes `/trade-api/v2` but NOT the query string.

Credentials (repo-root .env):
    KALSHI_API_KEY                  = <your key id>
    KALSHI_RSA_PRIVATE_KEY_PATH    = <path to the RSA private-key .pem>
                                     (default: repo-root .key)

Smoke test (read-only, safe — no orders placed):
    ../.venv/bin/python -c "import sys; sys.path.insert(0,'.'); \
        from venues.kalshi.executor import KalshiExecutor; \
        print(KalshiExecutor().check_balence())"
"""
import base64
import logging
import os
import datetime
from pathlib import Path

import requests

from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from venues.order import CapitalAccount, KALSHI_STATUS

# .env / .key live at the repo root, two levels above this package.
_REPO = Path(__file__).resolve().parents[2]
load_dotenv(_REPO / ".env")

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
API_PREFIX = "/trade-api/v2"  # the signed path is prefixed with this

log = logging.getLogger("exec.kalshi")


class KalshiExecutor(CapitalAccount):
    """Minimal signed-request Kalshi client. Owns its capital ledger
    (CapitalAccount): set `.capital` to a real seed before live trading —
    order() refuses (CapitalError) anything the ledger can't afford."""

    def __init__(self):
        self.key_id = os.getenv("KALSHI_API_KEY")
        key_path = Path(os.getenv("KALSHI_RSA_PRIVATE_KEY_PATH",
                                  str(_REPO / ".key")))
        self.rsa_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )
        self.orders = {}   # order.id (uuid) -> Order


    def get_timestamp_str(self):
        current_time = datetime.datetime.now()
        timestamp = current_time.timestamp()
        current_time_milliseconds = int(timestamp * 1000)
        return str(current_time_milliseconds)


    def get_sig(self, timestamp, method, path):
        path_without_query = path.split('?')[0]
        msg_string = timestamp + method.upper() + path_without_query
        sig = self.rsa_key.sign(
            msg_string.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode()

    def get_headers(self, method, path):
        # `path` is the full signed path, e.g. API_PREFIX + "/portfolio/balance".
        timestamp_str = self.get_timestamp_str()
        sig = self.get_sig(timestamp=timestamp_str, method=method, path=path)
        return {
            'Content-Type': 'application/json',
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-SIGNATURE': sig,
            'KALSHI-ACCESS-TIMESTAMP': timestamp_str,
        }

    def check_balence(self):
        headers = self.get_headers("GET", API_PREFIX + "/portfolio/balance")
        response = requests.get(BASE_URL + "/portfolio/balance", headers=headers)
        return response.json()["balance_dollars"]

    def _get_paginated(self, path, key):
        """GET a cursor-paginated portfolio endpoint, following `cursor`
        until exhausted. The signature covers the bare path (Kalshi signs
        method+path, query excluded). Missing `key` raises LOUD — requests
        doesn't raise on 4xx, and silently syncing against nothing is how
        phantom drift happens."""
        out: list = []
        cursor = None
        for _ in range(50):
            headers = self.get_headers("GET", API_PREFIX + path)
            url = BASE_URL + path + (f"?cursor={cursor}" if cursor else "")
            response = requests.get(url, headers=headers)
            data = response.json()
            if key not in data:
                raise RuntimeError(f"kalshi GET {path} failed "
                                   f"({response.status_code}): {data}")
            if not data[key]:
                return out          # empty page = exhausted, cursor or not
            out.extend(data[key])
            cursor = data.get("cursor")
            if not cursor:
                return out
        raise RuntimeError(f"kalshi {path} pagination did not terminate "
                           "after 50 pages")

    def get_orders_kalshi(self):
        # Unfiltered: returns resting AND executed/canceled orders, each with
        # `status` and `fill_count_fp`. Paginated (2026-07-22: unpaginated
        # reads returned varying subsets once the account grew, causing
        # phantom ReconcileError drift).
        return self._get_paginated("/portfolio/orders", "orders")

    def update_orders(self):
        """Sync every tracked order's status + filled from the venue (matched by
        client_order_id, which is our Order.id). Fills — partial or full — land
        on Order.filled, which is what positions/hedging are computed from."""
        by_client_id = {o["client_order_id"]: o for o in self.get_orders_kalshi()}
        for order in self.orders.values():
            d = by_client_id.get(order.id)
            if d is None:
                continue    # not in this page / venue lag — keep last known state
            order.order_id = d.get("order_id") or order.order_id
            order.filled = float(d.get("fill_count_fp") or 0)
            order.status = KALSHI_STATUS.get(d["status"], order.status)
        return self.orders

    def get_open_orders(self):
        self.update_orders()
        return {id: order for id, order in self.orders.items() if order.is_open}


    def order(self, order):
        self.check_capital(order)   # HARD stop: raises before the venue is touched
        headers = self.get_headers("POST", API_PREFIX + "/portfolio/events/orders")
        body = order.order_to_kalshi_body()
        response = requests.post(BASE_URL + "/portfolio/events/orders", headers=headers, json=body)

        data = response.json()
        # Kalshi's exchange order id — needed to cancel this order later. The
        # V2 create response is flat ({"order_id": ...}); tolerate the legacy
        # nested {"order": {...}} shape too.
        order.order_id = (data.get("order_id")
                          or (data.get("order") or {}).get("order_id")
                          or order.order_id)
        self.orders[order.id] = order
        if not order.order_id:
            # LOUD: requests doesn't raise on 4xx, so a rejected create looks
            # like success while the order never reaches the book. Mark it
            # canceled so its capital reservation releases immediately —
            # otherwise it "rests" in the ledger forever. If the venue DID
            # create it despite the odd response, the next update_orders sync
            # (matched by client_order_id) restores the true status.
            order.status = "canceled"
            log.warning("kalshi order NOT acked (%s %s x%s @ %s, http %s): %s",
                        order.market, order.side, order.count, order.price,
                        response.status_code, data)
        return data

    def cancel(self, order_id, market_slug=None):
        # V2: DELETE /portfolio/events/orders/{order_id}; market_slug is ignored
        # (Kalshi identifies the order by id alone) — kept for a uniform API.
        path = API_PREFIX + "/portfolio/events/orders/" + order_id
        headers = self.get_headers("DELETE", path)
        response = requests.delete(BASE_URL + "/portfolio/events/orders/" + order_id,
                                   headers=headers)
        data = response.json()
        if "order_id" not in data:
            # LOUD: a silently failed cancel leaves the order resting and the
            # strategy re-emitting the same CANCEL every tick.
            log.warning("kalshi cancel NOT acked (%s, http %s): %s",
                        order_id, response.status_code, data)
        else:
            # acked: mark the tracked order now (same noise-storm
            # protection as the polymarket executor, 2026-08-24)
            for o in self.orders.values():
                if o.order_id == order_id and o.is_open:
                    o.status = "canceled"
        return data

    def cancel_orders_kalshi(self, skip_tags=("unwind",), skip_markets=()):
        """Cancel session-tracked orders still resting on the venue, except
        those tagged in `skip_tags` or resting in `skip_markets`.

        Unwind legs are skipped by default: they deliberately rest at our
        profit target, and canceling one makes the strategy hedge the
        re-exposed pair at taker prices — then re-emit the unwind — paying a
        crossed spread + fees every refresh cycle. Returns canceled ids.
        (The runner's orphan sweep does NOT call this — resting Kalshi
        orders are hedges, and hedges reduce risk; see the risk gate.)"""
        canceled = []
        for order in self.get_open_orders().values():
            if order.tag in skip_tags or order.market in skip_markets:
                continue
            if not order.order_id:
                continue    # never acked by the venue — nothing to cancel
            try:
                data = self.cancel(order.order_id)
            except Exception as e:
                log.warning("refresh cancel failed (%s %s): %s",
                            order.market, order.order_id, e)
                continue
            if "order_id" in data:   # acked (self.cancel logs the NOT-acked case)
                order.status = "canceled"
                canceled.append(order.order_id)
        return canceled

    def positions(self):
        # market_positions[]: {ticker, position_fp (signed: + YES, - NO),
        # total_traded_dollars, ...}. Paginated — see _get_paginated.
        return self._get_paginated("/portfolio/positions", "market_positions")


if __name__ == "__main__":
    # Read-only smoke: prints the account balance. (The old testing-ground
    # __main__ placed a real order — deliberately NOT carried over.)
    print(KalshiExecutor().check_balence())
