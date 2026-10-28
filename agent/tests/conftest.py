"""Shared fixtures and sys.path setup for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure agent/ is on sys.path so imports like `backtest.*` and `src.*` work.
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Clear the cached EnvConfig before each test so monkeypatch.setenv works."""
    from src.config.accessor import reset_env_config
    reset_env_config()
    yield
    reset_env_config()


#: Opt-in switches that make the Jev integration reach the network. A
#: developer's local ``.env`` (loaded into ``os.environ`` by
#: ``_ensure_dotenv()``) would otherwise leak into the test process and turn
#: order-guard tests into live HTTP round-trips — slow and flaky. Tests that
#: want the network behaviour set the variable explicitly via
#: ``monkeypatch.setenv`` inside the test body.
_JEV_OPT_IN_VARS = (
    "JEV_ADVISORY",
    "JEV_ORDER_GUARD",
    "JEV_LIVE_VETO",
)


@pytest.fixture(autouse=True)
def _isolate_jev_opt_in(monkeypatch):
    """Force the Jev integration to its default (no-network) behaviour."""
    for name in _JEV_OPT_IN_VARS:
        monkeypatch.delenv(name, raising=False)
