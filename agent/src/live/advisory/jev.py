"""Jev pre-trade advisory provider.

Wires the Jev structured-decision model into the order guard's advisory
layer. The advisory layer is observational by design: a Jev verdict is
recorded in the audit trail and surfaced to the agent, but it never blocks an
order on its own. The mandate gate remains the sole allow/deny authority.

Jev is a good fit here because it answers a *fixed* question ("is this risk
acceptable?") with a calibrated probability in milliseconds, without the
prose an LLM would generate.
"""

from __future__ import annotations

import logging
from typing import Any

from src.live.advisory import (
    AdvisoryContext,
    AdvisoryResult,
    PreTradeAdvisoryInterface,
    Verdict,
)

logger = logging.getLogger(__name__)

#: Provider id recorded in the audit trail.
JEV_PROVIDER_ID = "jev"

#: Timeout for the advisory call. The advisory layer sits on the order path,
#: so a slow Jev must never stall an order: on timeout the provider returns
#: ``REVIEW_UNAVAILABLE`` and the order proceeds.
JEV_ADVISORY_TIMEOUT_SECONDS = 5.0

#: Environment variable that opts the Jev advisory provider in. Truthy values
#: (case-insensitive): ``"1"``, ``"true"``, ``"yes"``. The provider also
#: requires the project's own ``VIBE_TRADING_ENABLE_ADVISORY=1`` switch.
JEV_ADVISORY_ENV = "JEV_ADVISORY"
_TRUTHY = frozenset({"1", "true", "yes"})


def jev_advisory_enabled() -> bool:
    """Whether the operator opted the Jev advisory provider in."""
    import os

    return os.environ.get(JEV_ADVISORY_ENV, "").strip().lower() in _TRUTHY


class JevAdvisoryProvider(PreTradeAdvisoryInterface):
    """Ask Jev whether a proposed order carries acceptable risk.

    The provider is fail-open: any missing configuration, transport error or
    malformed verdict becomes :attr:`Verdict.REVIEW_UNAVAILABLE`, which the
    orchestrator treats as "no opinion" — the order proceeds unaffected.
    """

    @property
    def provider_id(self) -> str:
        """Unique identifier for this advisory provider."""
        return JEV_PROVIDER_ID

    def review(self, context: AdvisoryContext) -> AdvisoryResult:
        """Assess the risk of the order described by *context*.

        Args:
            context: Normalized pre-trade context.

        Returns:
            An :class:`AdvisoryResult` carrying the Jev verdict.
        """
        try:
            from src.providers.jev import (
                answer_probability,
                answer_value,
                get_jev_client,
            )
        except Exception as exc:  # noqa: BLE001 - fail-open
            return self._unavailable(f"jev client import failed: {exc}")

        try:
            client = get_jev_client()
        except Exception as exc:  # noqa: BLE001 - fail-open
            return self._unavailable(f"jev client unavailable: {exc}")

        if not client.is_configured:
            return self._unavailable("jev is not configured")

        state = (
            f"Proposed order: {context.side} {context.symbol} "
            f"notional ${context.notional_usd:.2f}. "
            f"Account equity: ${context.account_equity:.2f}. "
            f"Open positions: {context.open_position_count}. "
            f"Total exposure: ${context.total_exposure_usd:.2f}. "
            f"Funding cap: ${context.funding_usd:.2f}. "
            f"Funding utilization: {context.utilization_ratio:.2%}."
        )

        try:
            result = client.decide_from_context(
                state, "risk", timeout=JEV_ADVISORY_TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 - fail-open
            logger.warning("jev advisory review failed: %s", exc)
            return self._unavailable(f"jev request failed: {exc}")

        verdict_value = answer_value(result, "acceptable_risk")
        probability = answer_probability(result, "acceptable_risk")
        risk_level = answer_value(result, "risk_level")
        if verdict_value is None and probability is None:
            return self._unavailable("jev returned no usable verdict")

        if verdict_value == "yes":
            verdict = Verdict.APPROVE
        elif verdict_value == "no":
            verdict = Verdict.REJECT
        else:
            verdict = Verdict.APPROVE_WITH_CONCERNS

        concerns: tuple[str, ...] = ()
        if verdict is not Verdict.APPROVE:
            concerns = (
                f"jev flagged {context.side} {context.symbol} as "
                f"{verdict_value or 'uncertain'} risk"
                + (f" (risk level {risk_level})" if risk_level else ""),
            )

        summary = (
            f"jev risk verdict: {verdict_value or 'unknown'}"
            f" (probability {probability:.2f})"
            if probability is not None
            else f"jev risk verdict: {verdict_value or 'unknown'}"
        )
        detail: dict[str, Any] = {
            "acceptable_risk": verdict_value,
            "probability": probability,
            "risk_level": risk_level,
        }
        return AdvisoryResult(
            verdict=verdict,
            confidence=probability,
            summary=summary,
            concerns=concerns,
            provider=self.provider_id,
            detail=detail,
        )

    @staticmethod
    def _unavailable(reason: str) -> AdvisoryResult:
        """Build the fail-open "no opinion" result."""
        return AdvisoryResult(
            verdict=Verdict.REVIEW_UNAVAILABLE,
            summary=reason,
            provider=JEV_PROVIDER_ID,
        )


_JEV_PROVIDER_REGISTERED = False


def ensure_jev_advisory_registered() -> bool:
    """Register the Jev advisory provider once per process.

    The provider is only registered when Jev is actually configured **and**
    the operator opted in via ``JEV_ADVISORY=1``. An unconfigured provider
    would return ``REVIEW_UNAVAILABLE`` on every order, and because the
    orchestrator aggregates worst-case, that "no opinion" would mask a real
    verdict from another provider.

    Returns:
        ``True`` when the provider is registered, ``False`` when Jev is not
        configured, not opted in, or not importable.
    """
    global _JEV_PROVIDER_REGISTERED
    if _JEV_PROVIDER_REGISTERED:
        return True
    if not jev_advisory_enabled():
        return False
    try:
        from src.providers.jev import get_jev_client
    except Exception:  # noqa: BLE001 - jev is optional
        return False
    try:
        if not get_jev_client().is_configured:
            return False
    except Exception:  # noqa: BLE001 - jev is optional
        return False
    try:
        from src.live.advisory import get_advisory_providers, register_advisory_provider
    except Exception:  # noqa: BLE001 - advisory layer optional
        return False
    if any(
        getattr(provider, "provider_id", "") == JEV_PROVIDER_ID
        for provider in get_advisory_providers()
    ):
        _JEV_PROVIDER_REGISTERED = True
        return True
    register_advisory_provider(JevAdvisoryProvider())
    _JEV_PROVIDER_REGISTERED = True
    return True


__all__ = [
    "JEV_ADVISORY_ENV",
    "JEV_PROVIDER_ID",
    "JevAdvisoryProvider",
    "ensure_jev_advisory_registered",
    "jev_advisory_enabled",
]
