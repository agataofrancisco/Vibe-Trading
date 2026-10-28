"""Daily order-counter serialization tests with no broker transport."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.live.paths as live_paths
from src.live.daily_count import daily_order_lock

pytestmark = pytest.mark.unit


def _child_lock_attempt(repo_root: Path, home: Path) -> subprocess.CompletedProcess[str]:
    script = """
from src.live.daily_count import DailyOrderLockUnavailable, daily_order_lock
try:
    with daily_order_lock("alpaca"):
        print("acquired")
except DailyOrderLockUnavailable:
    print("blocked")
"""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(repo_root / "agent")
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )


def test_daily_order_lock_is_cross_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock held here denies another process, then becomes available."""
    runtime_root = tmp_path / ".vibe-trading"
    monkeypatch.setattr(live_paths, "get_runtime_root", lambda: runtime_root)
    repo_root = Path(__file__).resolve().parents[2]

    with daily_order_lock("alpaca"):
        blocked = _child_lock_attempt(repo_root, tmp_path)
    acquired = _child_lock_attempt(repo_root, tmp_path)

    assert blocked.stdout.strip() == "blocked"
    assert acquired.stdout.strip() == "acquired"


# --------------------------------------------------------------------------- #
# Daily starting-equity anchor (denominator of the daily-loss stop)           #
# --------------------------------------------------------------------------- #


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime_root = tmp_path / ".vibe-trading"
    monkeypatch.setattr(live_paths, "get_runtime_root", lambda: runtime_root)
    return runtime_root


def test_daily_start_equity_anchor_is_stable_for_the_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.live.daily_count import (
        ensure_daily_start_equity,
        increment_daily_count,
        read_daily_start_equity,
    )

    runtime_root = _runtime(tmp_path, monkeypatch)
    assert read_daily_start_equity("alpaca") is None

    assert ensure_daily_start_equity("alpaca", 10_000.0) == 10_000.0
    # A later call must never re-anchor, even if equity has moved.
    assert ensure_daily_start_equity("alpaca", 9_000.0) == 10_000.0

    increment_daily_count("alpaca")
    assert read_daily_start_equity("alpaca") == 10_000.0

    # The anchor and the counter live in separate files so a rejected order
    # never creates a counter record.
    broker_dir = runtime_root / "live" / "alpaca"
    assert (broker_dir / "daily_anchor.json").is_file()
    assert (broker_dir / "trade_counter.json").is_file()


def test_daily_start_equity_rejects_unusable_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.live.daily_count import ensure_daily_start_equity, read_daily_start_equity

    runtime_root = _runtime(tmp_path, monkeypatch)
    assert ensure_daily_start_equity("alpaca", None) is None
    assert ensure_daily_start_equity("alpaca", 0.0) is None
    assert ensure_daily_start_equity("alpaca", -5.0) is None
    assert read_daily_start_equity("alpaca") is None
    assert not (runtime_root / "live" / "alpaca" / "daily_anchor.json").is_file()

