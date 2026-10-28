"""Jev decision tool: fast, structured, type-safe decisions without prose.

Jev is a System One model from TypeSafe AI. It answers structured questions
with probabilities in milliseconds, which makes it a good fit for the
fast-path checks (should we trade? is this risky?) that would otherwise burn
an LLM round-trip.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.agent.tools import BaseTool
from src.providers.jev import JevError, get_jev_client


def _json_error(error: str, *, error_type: str = "runtime") -> str:
    """Return a standard JSON error envelope."""
    return json.dumps(
        {"status": "error", "error_type": error_type, "error": error},
        ensure_ascii=False,
    )


class JevDecideTool(BaseTool):
    """Make fast structured decisions using the Jev model."""

    name = "jev_decide"
    description = (
        "Make fast, structured decisions using Jev (TypeSafe AI System One model). "
        "Best for quick yes/no or scored judgements: should I trade, is this risky, "
        "should I escalate to a human. Returns answers with probabilities in "
        "milliseconds instead of generating long prose. Use the LLM for open-ended "
        "reasoning and research; use Jev when you need a fast, auditable verdict."
    )
    parameters = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": (
                    "Context shared by every question: market data, account state, "
                    "the proposed action, recent indicators. Plain text or JSON."
                ),
            },
            "decision_type": {
                "type": "string",
                "enum": ["trade", "risk", "route", "custom"],
                "description": (
                    "trade = should we place an order; risk = is this risk "
                    "acceptable; route = should this escalate to a human; "
                    "custom = use the questions you pass in."
                ),
            },
            "questions": {
                "type": "object",
                "description": (
                    "Custom question definitions, used only when decision_type is "
                    "'custom'. Each value accepts a 'type' of choice, score, or noul."
                ),
            },
            "allowed_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Candidate actions for a 'trade' decision, e.g. "
                    "['buy', 'sell', 'hold']."
                ),
            },
        },
        "required": ["state", "decision_type"],
    }
    repeatable = True
    is_readonly = True

    def execute(
        self,
        state: str,
        decision_type: str = "trade",
        questions: Optional[Dict[str, Any]] = None,
        allowed_actions: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        """Run a Jev decision and return the raw answer envelope as JSON."""
        try:
            client = get_jev_client()
        except Exception as exc:  # noqa: BLE001 - never raise into the agent loop
            return _json_error(f"Jev client unavailable: {exc}", error_type="dependency")

        try:
            if decision_type == "custom" and questions:
                result = client.decide(state, questions)
            else:
                options: Dict[str, Any] = {}
                if allowed_actions:
                    options["allowed_actions"] = allowed_actions
                result = client.decide_from_context(state, decision_type, **options)
        except JevError as exc:
            return _json_error(str(exc))
        except Exception as exc:  # noqa: BLE001 - fail closed, never raise
            return _json_error(f"Jev request failed: {exc}")

        return json.dumps(
            {"status": "ok", "model": "jev", "decision_type": decision_type, "result": result},
            ensure_ascii=False,
        )
