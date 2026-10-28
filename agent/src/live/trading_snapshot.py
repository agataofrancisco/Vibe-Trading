"""Trading snapshot for the live runtime surface.

The runtime panel answers three questions about a broker channel: is the
runner on, what is currently open, and is the account making or losing money.
This module builds that snapshot from the connector's own reads so the panel
never has to know a broker's payload shape.

Every read is best-effort: a missing SDK, a closed terminal or an unconfigured
profile yields a snapshot with ``available: False`` and a reason, never an
exception. The runtime status endpoint polls this on a timer, so a throwing
read would take the whole panel down.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce a broker value to a float, falling back to *default*."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _payload(envelope: Any, *keys: str) -> Any:
    """Return the first present payload key from a connector envelope."""
    if not isinstance(envelope, Mapping):
        return None
    for key in keys:
        if key in envelope:
            return envelope[key]
    return None


def _is_error(envelope: Any) -> bool:
    """Whether a connector envelope reports a failure."""
    return isinstance(envelope, Mapping) and str(envelope.get("status", "")).lower() == "error"


def _unavailable(reason: str, profile_id: str | None = None) -> dict[str, Any]:
    """Build the "no trading data" snapshot."""
    snapshot: dict[str, Any] = {"available": False, "reason": reason}
    if profile_id:
        snapshot["profile_id"] = profile_id
    return snapshot


def _position_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalize broker position rows into the panel's shape."""
    if not isinstance(rows, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        volume = _num(row.get("volume"))
        side = str(row.get("side", "")).strip().lower() or "buy"
        profit = _num(row.get("profit"))
        swap = _num(row.get("swap"))
        result.append(
            {
                "ticket": row.get("ticket"),
                "symbol": str(row.get("symbol", "")),
                "side": side,
                "volume": volume,
                "price_open": _num(row.get("price_open")),
                "price_current": _num(row.get("price_current")),
                "profit": profit,
                "swap": swap,
                "total_profit": profit + swap,
                "notional": _num(row.get("market_value")),
            }
        )
    return result


def _order_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalize broker order rows into the panel's shape."""
    if not isinstance(rows, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        result.append(
            {
                "order_id": str(row.get("order_id", "") or row.get("id", "")),
                "symbol": str(row.get("symbol", "")),
                "volume": _num(row.get("volume")),
                "price_open": _num(row.get("price_open")),
                "state": str(row.get("state", "")),
            }
        )
    return result


def build_trading_snapshot(broker: str, profile_id: str | None = None) -> dict[str, Any]:
    """Build the trading snapshot for one broker channel.

    Args:
        broker: Connector key, e.g. ``"mt5"``.
        profile_id: Profile to read. Defaults to the selected profile.

    Returns:
        A snapshot dict. Always contains ``available``; on success it also
        carries ``account``, ``positions``, ``open_orders``, ``daily`` and
        ``totals``.
    """
    from src.trading.profiles import profile_by_id
    from src.trading.service import _sdk_module

    try:
        profile = profile_by_id(profile_id) if profile_id else None
    except ValueError as exc:
        return _unavailable(str(exc))
    if profile is None:
        from src.trading.profiles import load_selected_profile_id

        try:
            profile = profile_by_id(load_selected_profile_id())
        except ValueError as exc:
            return _unavailable(str(exc))
    if profile.connector != broker:
        return _unavailable(
            f"selected profile {profile.id!r} does not belong to broker {broker!r}",
            profile.id,
        )
    if profile.transport != "broker_sdk":
        return _unavailable(
            f"broker {broker!r} is not a direct-SDK connector", profile.id
        )

    try:
        module = _sdk_module(profile.connector)
        config = module.build_config(profile.config, {})
    except Exception as exc:  # noqa: BLE001 - the panel must never break
        return _unavailable(f"connector unavailable: {type(exc).__name__}", profile.id)

    account_envelope = module.get_account_snapshot(config)
    if _is_error(account_envelope):
        return _unavailable(
            str(account_envelope.get("error") or "account read failed"), profile.id
        )
    account = _payload(account_envelope, "account") or {}

    positions_envelope = module.get_positions(config)
    positions = _position_rows(_payload(positions_envelope, "positions"))

    orders_envelope = module.get_open_orders(config)
    open_orders = _order_rows(_payload(orders_envelope, "open_orders", "orders"))

    # Realized P&L today. Optional: a connector without a deal read simply
    # reports zero rather than failing the whole snapshot.
    realized = 0.0
    deals_count = 0
    daily_reader = getattr(module, "get_daily_deals", None)
    if callable(daily_reader):
        try:
            deals_envelope = daily_reader(config, days=1)
            if not _is_error(deals_envelope):
                deals = _payload(deals_envelope, "deals") or []
                if isinstance(deals, (list, tuple)):
                    closing = [
                        d
                        for d in deals
                        if isinstance(d, Mapping) and d.get("is_closing")
                    ]
                    deals_count = len(closing)
                    realized = _num(
                        _payload(deals_envelope, "realized_profit_usd"), 0.0
                    )
        except Exception as exc:  # noqa: BLE001 - optional enrichment only
            logger.debug("daily deals read failed for %s: %s", broker, exc)

    unrealized = sum(row["total_profit"] for row in positions)
    open_notional = sum(abs(row["notional"]) for row in positions)

    # Daily trade cap + loss stop come from the committed mandate, when there is
    # one. Both are informational here — the order gates enforce them.
    from src.live.daily_count import read_daily_count, read_daily_start_equity
    from src.live.mandate.store import load_mandate

    try:
        daily_count = read_daily_count(broker)
    except Exception:  # noqa: BLE001 - the cap is informational here
        daily_count = 0
    mandate = load_mandate(broker)
    daily_cap = None
    daily_loss_limit = None
    daily_loss_pct = None
    daily_loss_halted = False
    if mandate is not None:
        cap = getattr(mandate.hard_caps, "max_trades_per_day", None)
        daily_cap = cap if isinstance(cap, int) else None
        limit = getattr(mandate.hard_caps, "max_daily_loss_pct", 0.0) or 0.0
        if limit > 0:
            daily_loss_limit = float(limit)
            try:
                start_equity = read_daily_start_equity(broker)
            except Exception:  # noqa: BLE001 - informational only
                start_equity = None
            equity_now = _num(account.get("equity"), 0.0)
            if start_equity and start_equity > 0 and equity_now > 0:
                daily_loss_pct = max(
                    0.0, (start_equity - equity_now) / start_equity
                )
                daily_loss_halted = daily_loss_pct >= daily_loss_limit

    return {
        "available": True,
        "profile_id": profile.id,
        "environment": profile.environment,
        "account": {
            "login": account.get("login"),
            "name": account.get("name"),
            "server": account.get("server"),
            "currency": account.get("currency"),
            "balance": _num(account.get("balance")),
            "equity": _num(account.get("equity")),
            "margin": _num(account.get("margin")),
            "margin_free": _num(account.get("margin_free")),
            "leverage": _num(account.get("leverage"), 0.0),
            "is_demo": account.get("is_demo"),
        },
        "positions": positions,
        "open_orders": open_orders,
        "daily": {
            "trades": daily_count,
            "cap": daily_cap,
            "closed_trades": deals_count,
            "realized_pnl": realized,
            "loss_limit_pct": daily_loss_limit,
            "loss_pct": daily_loss_pct,
            "loss_stop_hit": daily_loss_halted,
        },
        "totals": {
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "net_pnl": unrealized + realized,
            "open_positions": len(positions),
            "open_orders": len(open_orders),
            "open_notional": open_notional,
        },
    }


__all__ = ["build_trading_snapshot"]
