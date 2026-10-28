"""Direct-SDK adapter for the live runner.

The live runner is built around an MCP-shaped surface: it asks an adapter for
``call_tool(remote_name, arguments)`` and expects broker-neutral payloads
(a list of position dicts, a balance dict, a list of order dicts). Remote-MCP
brokers (Robinhood) satisfy that with :class:`~src.tools.mcp.MCPServerAdapter`.

Direct-SDK connectors (MT5) have no MCP server: they are plain Python calls
into the connector module. This adapter provides the same ``call_tool``
surface on top of :mod:`src.trading.service`, so the runner wiring, the
reconcile diff and the audit trail stay broker-agnostic.

Two responsibilities beyond the plain call:

* **Envelope unwrapping** - the service layer returns a tagged envelope
  (``{"status": "ok", "positions": [...]}``). The runner wants the payload.
* **Shape normalization** - the reconcile diff compares positions/orders
  across runs, so the rows must carry the generic keys it understands
  (signed ``qty``, ``symbol``, ``order_id``, ``status``). MT5 reports an
  unsigned ``volume`` plus a ``side``; the adapter derives the signed qty.

Writes are NOT normalized here: order placement goes through
:func:`src.trading.service.place_order`, which routes live profiles through
the mandate gate (``src.live.sdk_order_gate``) and paper profiles straight to
the broker sandbox. The adapter deliberately does not bypass that.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Runner operation -> the tool name the adapter exposes for it.
SDK_RUNNER_TOOLS: dict[str, str] = {
    "positions": "get_positions",
    "account": "get_account",
    "orders": "get_open_orders",
    "submit_order": "place_order",
    "cancel_order": "cancel_order",
}

#: MT5 position rows are unsigned; the reconcile diff compares signed qty.
_POSITION_QTY_KEYS = ("qty", "quantity", "position")


def _unwrap(envelope: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present payload key from a service envelope.

    Args:
        envelope: The service-layer response.
        *keys: Candidate payload keys, most specific first.

    Returns:
        The payload value, or the envelope itself when no key matches.
    """
    for key in keys:
        if key in envelope:
            return envelope[key]
    return envelope


def _is_error(envelope: Mapping[str, Any]) -> bool:
    """Whether the service envelope reports a failure."""
    return str(envelope.get("status", "")).lower() == "error"


def _signed_qty(row: Mapping[str, Any]) -> float:
    """Return the signed quantity of a position row.

    MT5 reports an unsigned ``volume`` plus a ``side``; the generic reconcile
    diff compares a signed qty. Rows that already carry a signed qty are
    passed through untouched.
    """
    for key in _POSITION_QTY_KEYS:
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return 0.0
    volume = row.get("volume")
    if volume is None:
        return 0.0
    try:
        volume = float(volume)
    except (TypeError, ValueError):
        return 0.0
    if str(row.get("side", "")).strip().lower() == "sell":
        return -abs(volume)
    return abs(volume)


def normalize_position(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one position row in the generic shape the runner diffs."""
    normalized = dict(row)
    normalized["qty"] = _signed_qty(row)
    return normalized


def normalize_order(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one order row in the generic shape the runner diffs.

    ``status`` is the generic field the reconcile uses to decide whether an
    order was durably confirmed. A resting broker order maps to
    ``"pending"``; anything else is reported by the broker's own state.
    """
    normalized = dict(row)
    if "order_id" not in normalized and row.get("id") is not None:
        normalized["order_id"] = row["id"]
    if not normalized.get("status"):
        state = str(row.get("state", "")).strip().lower()
        normalized["status"] = {
            "": "pending",
            "placed": "pending",
            "active": "pending",
            "partially_filled": "pending",
        }.get(state, state or "pending")
    if normalized.get("status") == "pending":
        normalized.setdefault("client_order_id", normalized.get("order_id"))
    return normalized


class SdkBrokerAdapter:
    """``call_tool`` surface over a direct-SDK connector.

    Args:
        connector: Connector key, e.g. ``"mt5"``.
        profile_id: Profile to operate on. Defaults to the selected profile.
        session_id: Originating session id, stamped onto audit events.
    """

    def __init__(
        self,
        connector: str,
        profile_id: str | None = None,
        session_id: str = "",
    ) -> None:
        self.connector = connector
        self.profile_id = profile_id
        self.session_id = session_id

    # -- helpers ------------------------------------------------------------

    def _profile(self) -> str | None:
        """Resolve the profile id to operate on."""
        if self.profile_id:
            return self.profile_id
        from src.trading.profiles import load_selected_profile_id

        return load_selected_profile_id()

    def _service(self) -> Any:
        """Import the trading service layer lazily."""
        from src.trading import service

        return service

    # -- reads --------------------------------------------------------------

    def read_positions(self) -> list[dict[str, Any]]:
        """Return the broker's open positions in the generic shape."""
        envelope = self._service().get_positions(self._profile())
        if not isinstance(envelope, Mapping):
            return []
        if _is_error(envelope):
            logger.warning("mt5 positions read failed: %s", envelope.get("error"))
            return []
        rows = _unwrap(envelope, "positions")
        if not isinstance(rows, (list, tuple)):
            return []
        return [normalize_position(row) for row in rows if isinstance(row, Mapping)]

    def read_balance(self) -> dict[str, Any]:
        """Return the broker balance in the generic shape."""
        envelope = self._service().get_account(self._profile())
        if not isinstance(envelope, Mapping):
            return {}
        if _is_error(envelope):
            logger.warning("mt5 account read failed: %s", envelope.get("error"))
            return {}
        account = _unwrap(envelope, "account")
        return dict(account) if isinstance(account, Mapping) else {}

    def read_open_orders(self) -> list[dict[str, Any]]:
        """Return the broker's resting orders in the generic shape."""
        envelope = self._service().get_open_orders(self._profile())
        if not isinstance(envelope, Mapping):
            return []
        if _is_error(envelope):
            logger.warning("mt5 orders read failed: %s", envelope.get("error"))
            return []
        rows = _unwrap(envelope, "open_orders", "orders")
        if not isinstance(rows, (list, tuple)):
            return []
        return [normalize_order(row) for row in rows if isinstance(row, Mapping)]

    # -- writes -------------------------------------------------------------

    def submit_order(self, order: Mapping[str, Any]) -> dict[str, Any]:
        """Place an order through the mandate-gated path.

        The runner is the *autonomous* surface, so its orders always go
        through the same fail-closed gate a live account uses
        (:func:`src.live.sdk_order_gate.execute_live_order`): a committed
        mandate, an unexpired consent, a clear kill switch, the notional /
        exposure / daily-count caps, and an audit event. This holds for a
        demo (paper) profile too - the caps are what make the autonomy
        bounded, and the connector's own identity guard keeps a paper profile
        from ever touching a real-money account.

        Args:
            order: The agent's order payload (``symbol``, ``side``, and one of
                ``notional`` / ``quantity``).

        Returns:
            The gate's response envelope.
        """
        symbol = str(order.get("symbol") or "").strip().upper()
        if not symbol:
            return {"status": "error", "error": "order is missing a symbol"}
        side = str(order.get("side") or "").strip().lower()
        if side not in ("buy", "sell"):
            return {"status": "error", "error": f"unsupported order side: {side!r}"}

        from src.live.enforcement import OrderIntent
        from src.live.sdk_order_gate import execute_live_order
        from src.trading.profiles import profile_by_id
        from src.trading.service import _order_classification, _sdk_module

        profile = profile_by_id(self._profile())
        module = _sdk_module(profile.connector)
        config = module.build_config(profile.config, {})

        notional = order.get("notional", order.get("notional_usd"))
        quantity = order.get("quantity", order.get("qty"))
        instrument_type, asset_class = _order_classification(profile.connector, symbol)
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            notional_usd=float(notional) if notional is not None else None,
            quantity=float(quantity) if quantity is not None else None,
            instrument_type=instrument_type,
            asset_class=asset_class,
        )
        place_kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": intent.quantity,
            "notional": intent.notional_usd,
            "order_type": str(order.get("order_type") or "market"),
            "limit_price": order.get("limit_price", order.get("price")),
            "time_in_force": str(order.get("time_in_force") or "day"),
        }
        return execute_live_order(
            broker=profile.connector,
            connector_module=module,
            config=config,
            intent=intent,
            place_kwargs=place_kwargs,
            session_id=self.session_id,
        )

    def cancel_order(self, order: Mapping[str, Any]) -> dict[str, Any]:
        """Cancel a resting order by its broker order id."""
        order_id = order.get("order_id") or order.get("id")
        if not order_id:
            return {"status": "error", "error": "order is missing an order_id"}
        return self._service().cancel_order(str(order_id), self._profile())

    # -- MCP-shaped surface -------------------------------------------------

    def call_tool(
        self,
        remote_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        local_name: str | None = None,
    ) -> Any:
        """Execute one runner operation.

        Args:
            remote_name: The tool name, e.g. ``"get_positions"``.
            arguments: Operation arguments.
            local_name: Ignored; accepted for signature parity with
                :class:`~src.tools.mcp.MCPServerAdapter`.

        Returns:
            The operation payload.
        """
        args = dict(arguments or {})
        if remote_name == SDK_RUNNER_TOOLS["positions"]:
            return self.read_positions()
        if remote_name == SDK_RUNNER_TOOLS["account"]:
            return self.read_balance()
        if remote_name == SDK_RUNNER_TOOLS["orders"]:
            return self.read_open_orders()
        if remote_name == SDK_RUNNER_TOOLS["submit_order"]:
            return self.submit_order(args)
        if remote_name == SDK_RUNNER_TOOLS["cancel_order"]:
            return self.cancel_order(args)
        return {"status": "error", "error": f"unknown tool: {remote_name}"}


__all__ = [
    "SDK_RUNNER_TOOLS",
    "SdkBrokerAdapter",
    "normalize_order",
    "normalize_position",
]
