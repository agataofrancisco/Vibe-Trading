"""Per-broker daily order counter + daily equity anchor (UTC calendar day).

Shared by every live order path (the MCP ``LiveOrderGuardTool`` keeps its own
in-class copy for now; the direct-SDK gate uses these helpers). The counter is
advisory defense-in-depth — the broker enforces the real ceiling — so any
read failure reads as ``0`` (fail-open on the count only, never on the order).

``daily_anchor.json`` stores the UTC day's starting equity: the denominator of
the mandate's ``max_daily_loss_pct`` hard stop. It is written once per day, on
the first order-gate evaluation, and kept separate from the order counter so a
rejected order never creates a counter record.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import BinaryIO, Iterator

try:  # POSIX advisory lock (Linux/macOS).
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows advisory byte-range lock.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

from src.live.paths import broker_dir

_COUNTER_FILENAME = "trade_counter.json"
_ANCHOR_FILENAME = "daily_anchor.json"
_LOCK_FILENAME = ".order_submit.lock"


class DailyOrderLockUnavailable(RuntimeError):
    """Raised when another process holds the broker's order permit lock."""


def _counter_path(broker: str):
    return broker_dir(broker) / _COUNTER_FILENAME


def _anchor_path(broker: str):
    return broker_dir(broker) / _ANCHOR_FILENAME


def _utc_today() -> str:
    """Return today's UTC calendar date as ``YYYY-MM-DD``."""
    return datetime.now(timezone.utc).date().isoformat()


def _try_lock(handle: BinaryIO) -> None:
    """Acquire a non-blocking cross-process advisory lock."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is not None:  # pragma: no cover - exercised on Windows CI
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    raise OSError("no supported advisory lock backend")


def _unlock(handle: BinaryIO) -> None:
    """Release the platform advisory lock."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - exercised on Windows CI
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def daily_order_lock(broker: str) -> Iterator[None]:
    """Hold one non-blocking order-admission lock for ``broker``.

    The caller must keep this lock through the final daily-count check, broker
    submission, and durable count increment. Lock contention fails closed.

    Raises:
        DailyOrderLockUnavailable: If the lock cannot be acquired immediately.
    """
    path = broker_dir(broker) / _LOCK_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = path.open("a+b")
    except OSError as exc:
        raise DailyOrderLockUnavailable(
            f"the {broker} order-submission lock is unavailable"
        ) from exc
    try:
        _try_lock(handle)
    except OSError as exc:
        handle.close()
        raise DailyOrderLockUnavailable(
            f"another {broker} order submission is already in progress"
        ) from exc
    try:
        yield
    finally:
        try:
            _unlock(handle)
        finally:
            handle.close()


def _read_raw(broker: str) -> dict:
    """Return today's counter record (``{}`` on miss, corruption or rollover)."""
    path = _counter_path(broker)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("date") != _utc_today():
        return {}
    return raw


def _atomic_write(path, payload: dict) -> None:
    """Atomically persist ``payload`` as JSON at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_raw(broker: str, payload: dict) -> None:
    """Atomically persist today's counter record."""
    _atomic_write(_counter_path(broker), payload)


def _read_anchor_raw(broker: str) -> dict:
    """Return today's equity-anchor record (``{}`` on miss/rollover)."""
    path = _anchor_path(broker)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("date") != _utc_today():
        return {}
    return raw


def read_daily_count(broker: str) -> int:
    """Return today's order count for ``broker`` (UTC rollover; 0 on any miss)."""
    try:
        return int(_read_raw(broker).get("count", 0))
    except (TypeError, ValueError):
        return 0


def increment_daily_count(broker: str) -> int:
    """Persist ``broker``'s incremented count for today (atomic). Returns new count."""
    count = read_daily_count(broker) + 1
    _write_raw(broker, {"date": _utc_today(), "count": count})
    return count


def read_daily_start_equity(broker: str) -> float | None:
    """Return today's persisted starting equity, or ``None`` when unset."""
    value = _read_anchor_raw(broker).get("equity")
    try:
        equity = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return equity if equity > 0 else None


def ensure_daily_start_equity(broker: str, equity: float | None) -> float | None:
    """Return today's starting equity, persisting ``equity`` on first use.

    The anchor lives in its own ``daily_anchor.json`` (not the order counter, so
    a rejected order never creates a counter record). It is written once per UTC
    day and never rewritten, so a mid-day restart still measures the day's loss
    against the equity the day opened with. A non-positive or unusable
    ``equity`` leaves the anchor unset.

    Args:
        broker: Broker key.
        equity: Current account equity in USD, used only when no anchor exists.

    Returns:
        The anchor equity, or ``None`` when it cannot be established.
    """
    existing = read_daily_start_equity(broker)
    if existing is not None:
        return existing
    if equity is None:
        return None
    try:
        value = float(equity)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    _atomic_write(
        _anchor_path(broker), {"date": _utc_today(), "equity": value}
    )
    return value
