"""Jev (TypeSafe AI) client: fast, structured, type-safe decisions.

Jev is a "System One" model: instead of generating prose it answers a fixed
set of structured questions about a shared state and returns each answer with
a probability. That makes it a good fit for the hot path of an automated
trading loop, where a single yes/no verdict ("should we trade?") is worth far
more than a paragraph of reasoning.

Transport
---------
Jev is served through the same OpenRouter gateway the rest of the project
already uses, so a single ``OPENROUTER_API_KEY`` covers both the LLM and the
Jev model. The model id is ``typesafe/jev-router``.

Question types
--------------
``choice``  - pick one of ``options``; the answer carries a probability.
``score``   - a numeric value inside ``range``; the answer carries a probability.
``noul``    - a tri-state yes/no/uncertain verdict with a probability.

The client always returns the same envelope::

    {"status": "ok", "model": ..., "result": {"answers": {<qname>: {...}}}}

so callers can read ``result.answers.<qname>.probability`` without knowing
which transport produced the answer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)

#: Default OpenRouter-hosted Jev model.
DEFAULT_JEV_MODEL = "typesafe/jev-router"

#: Default gateway root. Jev rides the same OpenRouter key as the LLM.
DEFAULT_JEV_BASE_URL = "https://openrouter.ai/api/v1"

#: Decisions API endpoint (structured decision models: Jev v1, Mercury Decide,
#: Mammoth 1...). These models do NOT accept chat completions.
DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

#: Model prefixes served by the Decisions API. Selecting one of these through
#: ``JEV_MODEL`` automatically switches the client to the decisions transport.
_DECISIONS_MODEL_PREFIXES = (
    "baseten/mammoth-",
    "baseten/elephant-",
    "inception/mercury-decide",
    "typesafe/jev-1.",
)

#: Request timeout (seconds). Jev is a fast model; keep this tight so a
#: decision can never stall the trading loop for long.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Output cap for a decision. The answer is a small JSON object, so a few
#: hundred tokens is plenty. Capping it also keeps the request affordable on
#: gateways that bill against the requested ceiling rather than the usage.
DEFAULT_MAX_TOKENS = 512

_NOUL_ANSWERS = ("yes", "no", "uncertain")


class JevError(RuntimeError):
    """Raised when a Jev request cannot be completed."""


def _question_type(spec: Any) -> str:
    """Return the normalized question type for a question spec."""
    if isinstance(spec, str):
        return spec.strip().lower() or "noul"
    if isinstance(spec, Mapping):
        raw = spec.get("type") or "noul"
        return str(raw).strip().lower() or "noul"
    return "noul"


def _question_options(spec: Any) -> list[str]:
    """Return the option list for a ``choice`` question."""
    if not isinstance(spec, Mapping):
        return []
    options = spec.get("options")
    if not isinstance(options, (list, tuple)):
        return []
    return [str(option) for option in options]


def _question_range(spec: Any) -> tuple[float, float]:
    """Return the ``(low, high)`` bounds for a ``score`` question."""
    if not isinstance(spec, Mapping):
        return (0.0, 1.0)
    bounds = spec.get("range")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        return (0.0, 1.0)
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError):
        return (0.0, 1.0)
    return (low, high) if low <= high else (high, low)


def _build_answer_schema(
    questions: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Build a JSON schema for the answers plus a per-question type map.

    Args:
        questions: Question name -> question spec.

    Returns:
        Tuple of ``(json_schema, type_map)``.
    """
    properties: Dict[str, Any] = {}
    required: list[str] = []
    type_map: Dict[str, str] = {}

    for name, spec in questions.items():
        key = str(name)
        kind = _question_type(spec)
        type_map[key] = kind
        if kind == "choice":
            options = _question_options(spec)
            if not options:
                options = list(_NOUL_ANSWERS)
            properties[key] = {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": options},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["value", "probability"],
                "additionalProperties": False,
            }
        elif kind == "score":
            low, high = _question_range(spec)
            properties[key] = {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": low, "maximum": high},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["value", "probability"],
                "additionalProperties": False,
            }
        else:  # noul - tri-state verdict
            properties[key] = {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": list(_NOUL_ANSWERS)},
                    "probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["value", "probability"],
                "additionalProperties": False,
            }
        required.append(key)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, type_map


def _render_questions(questions: Mapping[str, Any]) -> str:
    """Render the question set as an explicit instruction block."""
    lines: list[str] = []
    for name, spec in questions.items():
        kind = _question_type(spec)
        if kind == "choice":
            options = _question_options(spec)
            lines.append(
                f'- "{name}" (choice): one of {options}. '
                'Return {"value": <the chosen option>, "probability": <0-1>}.'
            )
        elif kind == "score":
            low, high = _question_range(spec)
            lines.append(
                f'- "{name}" (score): a number between {low} and {high}. '
                'Return {"value": <the number>, "probability": <0-1>}.'
            )
        else:
            lines.append(
                f'- "{name}" (noul): one of {list(_NOUL_ANSWERS)}. '
                'Return {"value": <the verdict>, "probability": <0-1>}.'
            )
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You are Jev, a System One decision model. You evaluate the given state "
    "and answer each structured question with a verdict and a calibrated "
    "probability. You do not explain, argue, or add prose. You output only "
    "the JSON object described by the schema."
)


def _extract_content(payload: Mapping[str, Any]) -> str:
    """Pull the assistant message content out of a chat-completion payload."""
    choices = payload.get("choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise JevError("Jev response contained no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise JevError("Jev response choice was malformed")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise JevError("Jev response contained no message")
    content = message.get("content")
    if isinstance(content, list):
        # Some gateways return content blocks instead of a plain string.
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping)
        ]
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise JevError("Jev response contained empty content")
    return content


def _parse_answers(content: str) -> Dict[str, Any]:
    """Parse the model output into an ``answers`` mapping."""
    text = content.strip()
    if text.startswith("```"):
        # Strip a fenced code block if the gateway wrapped the JSON.
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        # Fall back to the first JSON object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise JevError(f"Jev response was not valid JSON: {text[:200]}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise JevError("Jev response was not a JSON object")
    return dict(parsed)


def _normalize_answer(value: Any) -> Dict[str, Any]:
    """Normalize one answer to ``{"value": ..., "probability": ...}``."""
    if isinstance(value, Mapping):
        probability = value.get("probability")
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            probability = 0.5
        probability = max(0.0, min(1.0, probability))
        return {"value": value.get("value"), "probability": probability}
    if isinstance(value, bool):
        return {"value": "yes" if value else "no", "probability": 0.5}
    if isinstance(value, (int, float)):
        return {"value": value, "probability": 0.5}
    return {"value": value, "probability": 0.5}


class JevClient:
    """Client for the Jev decision model.

    Args:
        api_key: API key. Defaults to ``JEV_API_KEY`` then ``OPENROUTER_API_KEY``.
        base_url: Gateway root. Defaults to ``JEV_BASE_URL`` then
            ``OPENROUTER_BASE_URL`` then the OpenRouter default.
        model: Model id. Defaults to ``JEV_MODEL`` then ``typesafe/jev-router``.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("JEV_API_KEY", "")
            or os.getenv("OPENROUTER_API_KEY", "")
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("JEV_BASE_URL", "")
            or os.getenv("OPENROUTER_BASE_URL", "")
            or DEFAULT_JEV_BASE_URL
        ).rstrip("/")
        self.model = (
            model or os.getenv("JEV_MODEL", "") or DEFAULT_JEV_MODEL
        ).strip()
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Whether a credential is available for a request."""
        return bool(self.api_key)

    def decide(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Answer structured questions about a shared state.

        Args:
            state: Context every question is asked against.
            questions: Question name -> question spec. The spec ``type`` is
                one of ``choice``, ``score`` or ``noul``.
            model: Optional model override for this call.

        Returns:
            ``{"status": "ok", "model": ..., "result": {"answers": {...}}}``.

        Raises:
            JevError: When the request fails or the response is unusable.
        """
        if not isinstance(questions, Mapping) or not questions:
            raise JevError("Jev requires at least one question")
        if not self.api_key:
            raise JevError(
                "Jev is not configured: set JEV_API_KEY or OPENROUTER_API_KEY"
            )

        schema, type_map = _build_answer_schema(questions)
        prompt = (
            f"State:\n{state}\n\n"
            f"Questions:\n{_render_questions(questions)}\n\n"
            "Answer every question. Output only the JSON object."
        )

        payload: Dict[str, Any] = {
            "model": (model or self.model),
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "jev_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
        }
        max_tokens = kwargs.get("max_tokens") or DEFAULT_MAX_TOKENS
        payload["max_tokens"] = int(max_tokens)
        timeout = float(kwargs.get("timeout") or self.timeout)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text[:400]
            except Exception:  # noqa: BLE001 - best-effort detail only
                detail = ""
            raise JevError(
                f"Jev request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise JevError(f"Jev request failed: {exc}") from exc
        except ValueError as exc:
            raise JevError(f"Jev response was not valid JSON: {exc}") from exc

        answers = _parse_answers(_extract_content(body))
        normalized = {
            name: _normalize_answer(value) for name, value in answers.items()
        }
        return {
            "status": "ok",
            "model": payload["model"],
            "question_types": type_map,
            "result": {"answers": normalized},
        }

    def decide_from_context(
        self,
        context: str,
        decision_type: str = "trade",
        **options: Any,
    ) -> Dict[str, Any]:
        """Run a predefined decision question set.

        Args:
            context: The state the decision is made against.
            decision_type: One of ``trade``, ``risk``, ``route``.
            **options: Extra options. ``allowed_actions`` supplies the
                candidate actions for a ``trade`` decision and ``reasons``
                supplies the candidate reasons for a ``route`` decision.

        Returns:
            The same envelope as :meth:`decide`.
        """
        if decision_type == "trade":
            questions: Dict[str, Any] = {
                "should_trade": {"type": "noul"},
                "confidence": {"type": "score", "range": [0, 1]},
                "urgency": {"type": "score", "range": [1, 5]},
            }
            allowed = options.get("allowed_actions")
            if allowed:
                questions["action"] = {"type": "choice", "options": list(allowed)}
        elif decision_type == "risk":
            questions = {
                "acceptable_risk": {"type": "noul"},
                "risk_level": {"type": "score", "range": [1, 10]},
            }
        elif decision_type == "route":
            questions = {
                "route_to_human": {"type": "noul"},
                "reason": {
                    "type": "choice",
                    "options": list(
                        options.get(
                            "reasons", ["complex", "risky", "unclear", "normal"]
                        )
                    ),
                },
            }
        else:
            questions = {"decision": {"type": "noul"}}

        return self.decide(context, questions, **options)


# --------------------------------------------------------------------------- #
# Decisions API transport (Jev v1, Mercury Decide, Mammoth 1, ...)
# --------------------------------------------------------------------------- #

#: Human instructions for the question names this codebase already uses, so the
#: decisions models get full sentences instead of bare snake_case keys.
_KNOWN_INSTRUCTIONS: Dict[str, str] = {
    "should_trade": (
        "Should a trading system open a new position right now, based on the "
        "price action in the state?"
    ),
    "acceptable_risk": (
        "Is the risk of a new position acceptable under the stated risk budget "
        "and with no existing open positions?"
    ),
    "action": "Which action should the trading system take right now?",
    "confidence": "How confident are you in this directional call?",
    "risk_level": "How risky is the current market condition?",
    "urgency": "How urgent is acting on this setup?",
}

_KNOWN_NOUL_CRITERIA: Dict[str, Dict[str, str]] = {
    "should_trade": {
        "true": "The recent price action has a clear directional edge worth acting on now.",
        "false": "The price action is choppy, exhausted, or lacks a clear edge, so no new position should be opened.",
    },
    "acceptable_risk": {
        "true": "Volatility and range are normal enough that the stated risk budget is reasonable.",
        "false": "Volatility is extreme or the setup is too uncertain for the stated risk budget.",
    },
}

_KNOWN_CHOICE_CRITERIA: Dict[str, Dict[str, str]] = {
    "action": {
        "buy": "The evidence points to further upside.",
        "sell": "The evidence points to further downside.",
        "hold": "There is no clear directional edge; stay flat.",
    },
}


def _decisions_questions(questions: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate the internal question specs to Decisions API questions."""
    out: Dict[str, Any] = {}
    for name, spec in questions.items():
        key = str(name)
        kind = _question_type(spec)
        spec_map = spec if isinstance(spec, Mapping) else {}
        instructions = str(
            spec_map.get("instructions") or _KNOWN_INSTRUCTIONS.get(key) or ""
        ).strip()
        criteria = spec_map.get("criteria")

        if kind == "choice":
            options = _question_options(spec)
            if not criteria:
                known = _KNOWN_CHOICE_CRITERIA.get(key) or {}
                if known and all(option in known for option in options):
                    criteria = {option: known[option] for option in options}
                else:
                    criteria = {
                        option: f"The best choice for this judgment is {option!r}."
                        for option in options
                    }
            if not instructions:
                instructions = f"Choose exactly one of: {', '.join(options)}."
        elif kind == "score":
            low, high = _question_range(spec)
            if not criteria:
                criteria = ["Low", "Medium", "High"]
            if not instructions:
                instructions = f"Rate the degree on a scale from {low} to {high}."
        else:  # noul
            if not criteria:
                criteria = _KNOWN_NOUL_CRITERIA.get(key) or {
                    "true": "The condition holds.",
                    "false": "The condition does not hold.",
                }
            if not instructions:
                instructions = f"Does {key.replace('_', ' ')!r} hold given the state?"

        out[key] = {
            "type": kind,
            "instructions": instructions,
            "criteria": criteria,
        }
    return out


def _decisions_answer(
    name: str, answer: Mapping[str, Any], spec: Any
) -> Dict[str, Any]:
    """Map one Decisions API answer to the internal ``{value, probability}``."""
    kind = _question_type(spec)

    if kind == "choice":
        chosen = answer.get("choice")
        probabilities = answer.get("probabilities")
        probability = 0.5
        if isinstance(probabilities, Mapping) and chosen is not None:
            try:
                probability = float(probabilities.get(str(chosen), 0.5))
            except (TypeError, ValueError):
                probability = 0.5
        return {"value": chosen, "probability": max(0.0, min(1.0, probability))}

    if kind == "score":
        probabilities = answer.get("probabilities")
        levels = len(probabilities) if isinstance(probabilities, Mapping) and probabilities else 3
        try:
            raw = float(answer.get("score"))
        except (TypeError, ValueError):
            raw = (levels - 1) / 2.0
        ratio = raw / (levels - 1) if levels > 1 else 0.5
        low, high = _question_range(spec)
        value = low + (high - low) * max(0.0, min(1.0, ratio))
        try:
            probability = float(answer.get("confidence", 0.5))
        except (TypeError, ValueError):
            probability = 0.5
        return {
            "value": round(value, 6),
            "probability": max(0.0, min(1.0, probability)),
        }

    # noul: the API returns P(yes) only. A 0.5 gate is the documented default;
    # the reported probability is the probability of the returned verdict.
    try:
        yes_probability = float(answer.get("noul"))
    except (TypeError, ValueError):
        yes_probability = 0.5
    yes_probability = max(0.0, min(1.0, yes_probability))
    value = "yes" if yes_probability >= 0.5 else "no"
    probability = yes_probability if value == "yes" else 1.0 - yes_probability
    return {"value": value, "probability": probability}


class DecisionsClient(JevClient):
    """Client for OpenRouter's Decisions API (probability outputs, no text).

    Returns the exact same envelope as :class:`JevClient`, so the runner, the
    advisory layer and the backtest harness need no changes. Selected
    automatically when ``JEV_MODEL`` names a decisions model, or explicitly via
    ``JEV_PROTOCOL=decisions``. The endpoint can be overridden with
    ``JEV_DECISIONS_URL``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        decisions_url: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key, base_url=base_url, model=model, timeout=timeout
        )
        self.decisions_url = (
            decisions_url
            or os.getenv("JEV_DECISIONS_URL", "")
            or DEFAULT_DECISIONS_URL
        ).rstrip("/")

    def decide(
        self,
        state: str,
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Answer structured questions via the Decisions API."""
        if not isinstance(questions, Mapping) or not questions:
            raise JevError("Jev requires at least one question")
        if not self.api_key:
            raise JevError(
                "Jev is not configured: set JEV_API_KEY or OPENROUTER_API_KEY"
            )

        payload: Dict[str, Any] = {
            "model": (model or self.model),
            "questions": _decisions_questions(questions),
            "state": state,
        }
        timeout = float(kwargs.get("timeout") or self.timeout)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self.decisions_url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text[:400]
            except Exception:  # noqa: BLE001 - best-effort detail only
                detail = ""
            raise JevError(
                f"Decisions request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise JevError(f"Decisions request failed: {exc}") from exc
        except ValueError as exc:
            raise JevError(f"Decisions response was not valid JSON: {exc}") from exc

        raw_answers = body.get("answers")
        if not isinstance(raw_answers, Mapping):
            raise JevError("Decisions response contained no answers")

        normalized: Dict[str, Any] = {}
        for name, spec in questions.items():
            key = str(name)
            raw = raw_answers.get(key)
            if isinstance(raw, Mapping):
                normalized[key] = _decisions_answer(key, raw, spec)

        return {
            "status": "ok",
            "model": body.get("model") or payload["model"],
            "provider": body.get("provider"),
            "usage": body.get("usage"),
            "question_types": {
                str(name): _question_type(spec) for name, spec in questions.items()
            },
            "result": {"answers": normalized},
        }


def _is_decisions_model(model: str) -> bool:
    """Return whether ``model`` is served by the Decisions API."""
    normalized = (model or "").strip().lower()
    return any(
        normalized.startswith(prefix) for prefix in _DECISIONS_MODEL_PREFIXES
    )


_jev_client: Optional[JevClient] = None


def read_answer(result: Mapping[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Read one answer out of a Jev envelope.

    Args:
        result: The envelope returned by :meth:`JevClient.decide`.
        name: Question name.

    Returns:
        The ``{"value": ..., "probability": ...}`` dict, or ``None``.
    """
    if not isinstance(result, Mapping):
        return None
    answers = result.get("result")
    if not isinstance(answers, Mapping):
        return None
    answers = answers.get("answers")
    if not isinstance(answers, Mapping):
        return None
    answer = answers.get(name)
    return answer if isinstance(answer, Mapping) else None


def answer_value(result: Mapping[str, Any], name: str) -> Optional[str]:
    """Return the verdict of one answer as a lower-case string."""
    answer = read_answer(result, name)
    if not answer:
        return None
    value = answer.get("value")
    if value is None:
        return None
    return str(value).strip().lower()


def answer_probability(result: Mapping[str, Any], name: str) -> Optional[float]:
    """Return the calibrated probability attached to one answer."""
    answer = read_answer(result, name)
    if not answer:
        return None
    probability = answer.get("probability")
    try:
        return float(probability)
    except (TypeError, ValueError):
        return None


def get_jev_client() -> JevClient:
    """Return the process-wide decision client, creating it on first use.

    Chooses the transport from the environment: ``JEV_PROTOCOL=decisions`` (or
    a ``JEV_MODEL`` served by the Decisions API, e.g. ``inception/mercury-decide:free``)
    yields :class:`DecisionsClient`; otherwise the chat-completions
    :class:`JevClient` is used.
    """
    global _jev_client
    if _jev_client is None:
        protocol = os.getenv("JEV_PROTOCOL", "").strip().lower()
        model = os.getenv("JEV_MODEL", "")
        if protocol == "decisions" or (
            protocol not in ("chat", "completions") and _is_decisions_model(model)
        ):
            _jev_client = DecisionsClient()
        else:
            _jev_client = JevClient()
    return _jev_client


def reset_jev_client() -> None:
    """Drop the cached client so the next call re-reads the environment."""
    global _jev_client
    _jev_client = None
