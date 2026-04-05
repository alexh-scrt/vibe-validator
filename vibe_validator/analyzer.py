"""Analyzer module for vibe_validator.

This module is responsible for orchestrating the viability analysis pipeline:

1. Accept an :class:`~vibe_validator.models.IdeaRequest`.
2. Build the OpenAI messages via :mod:`vibe_validator.prompts`.
3. Call the OpenAI Chat Completions API.
4. Parse and validate the JSON response into a
   :class:`~vibe_validator.models.ViabilityReport`.
5. Raise descriptive, typed exceptions on failure so callers can handle
   errors gracefully.

The OpenAI client is built lazily on first use so that the module can be
imported in environments where ``OPENAI_API_KEY`` is not yet set (e.g. during
testing with mocks).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI
from pydantic import ValidationError

from vibe_validator.models import IdeaRequest, ViabilityReport
from vibe_validator.prompts import build_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AnalyzerError(Exception):
    """Base exception for all analyzer failures."""


class OpenAIClientError(AnalyzerError):
    """Raised when the OpenAI API call itself fails (network, auth, etc.)."""


class ResponseParseError(AnalyzerError):
    """Raised when the LLM response cannot be decoded as valid JSON."""


class ReportValidationError(AnalyzerError):
    """Raised when the parsed JSON fails Pydantic model validation."""


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------


def _get_openai_client() -> OpenAI:
    """Build and return an :class:`openai.OpenAI` client instance.

    Reads configuration from environment variables set via ``.env``.

    Returns
    -------
    OpenAI
        A configured OpenAI client.

    Raises
    ------
    OpenAIClientError
        If ``OPENAI_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise OpenAIClientError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> str:
    """Extract a JSON object string from raw LLM output.

    The LLM is instructed to return only JSON, but sometimes wraps the
    response in markdown code fences or adds a short preamble.  This helper
    attempts to locate the first ``{`` and last ``}`` in the string and return
    just that slice.

    Parameters
    ----------
    raw:
        The raw string returned by the LLM.

    Returns
    -------
    str
        The extracted JSON string.

    Raises
    ------
    ResponseParseError
        If no JSON object boundaries can be found in the response.
    """
    if not raw or not raw.strip():
        raise ResponseParseError(
            "Cannot extract JSON from an empty or whitespace-only string."
        )

    # Strip markdown fences if present: ```json ... ``` or ``` ... ```
    stripped = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ResponseParseError(
            f"Could not locate a JSON object in the LLM response. "
            f"Raw response (first 500 chars): {raw[:500]!r}"
        )

    return stripped[start : end + 1]


def _parse_response(raw_content: str) -> dict[str, Any]:
    """Decode the raw LLM output into a Python dictionary.

    Parameters
    ----------
    raw_content:
        The raw content string from the OpenAI chat completion message.

    Returns
    -------
    dict[str, Any]
        The decoded JSON payload.

    Raises
    ------
    ResponseParseError
        If the content cannot be decoded as JSON or is not a dict.
    """
    json_str = _extract_json(raw_content)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(
            f"LLM returned invalid JSON: {exc}. "
            f"Extracted string (first 500 chars): {json_str[:500]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ResponseParseError(
            f"Expected a JSON object at the top level, got {type(data).__name__}."
        )
    return data


def _validate_report(data: dict[str, Any]) -> ViabilityReport:
    """Validate the parsed dictionary against the :class:`ViabilityReport` model.

    Parameters
    ----------
    data:
        The decoded JSON payload from the LLM.

    Returns
    -------
    ViabilityReport
        A fully validated viability report.

    Raises
    ------
    ReportValidationError
        If the data does not satisfy the Pydantic model constraints.
    """
    try:
        return ViabilityReport.model_validate(data)
    except ValidationError as exc:
        error_summary = "; ".join(
            f"{'.' .join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ReportValidationError(
            f"LLM response failed model validation ({exc.error_count()} error(s)): "
            f"{error_summary}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_idea(
    request: IdeaRequest,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.7,
    timeout: float = 60.0,
) -> ViabilityReport:
    """Analyse a side-hustle app idea and return a structured viability report.

    This is the primary entry point for the analyzer.  It:

    1. Builds the prompt messages for the OpenAI API.
    2. Calls the chat completions endpoint.
    3. Extracts, parses, and validates the JSON response.
    4. Returns a fully validated :class:`ViabilityReport` instance.

    Parameters
    ----------
    request:
        The validated idea submission from the user.
    client:
        Optional pre-constructed :class:`openai.OpenAI` client.  If not
        provided, one is built from environment variables.  Useful for
        injecting mocks in tests.
    model:
        OpenAI model name to use.  Defaults to the ``OPENAI_MODEL``
        environment variable, falling back to ``"gpt-4o"``.
    max_tokens:
        Maximum tokens for the completion.  Defaults to the
        ``OPENAI_MAX_TOKENS`` environment variable, falling back to ``2048``.
    temperature:
        Sampling temperature (0.0 – 2.0).  Defaults to ``0.7`` for a
        balance between creativity and reliability.
    timeout:
        Request timeout in seconds.  Defaults to ``60.0``.

    Returns
    -------
    ViabilityReport
        The parsed and validated viability report.

    Raises
    ------
    OpenAIClientError
        If the OpenAI API call fails due to auth, connection, timeout,
        or API-level errors.
    ResponseParseError
        If the LLM response cannot be decoded as valid JSON.
    ReportValidationError
        If the decoded JSON does not satisfy the :class:`ViabilityReport`
        schema.
    """
    if client is None:
        client = _get_openai_client()

    resolved_model: str = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    resolved_max_tokens: int = max_tokens or int(
        os.getenv("OPENAI_MAX_TOKENS", "2048")
    )

    messages = build_messages(request.idea)

    logger.info(
        "Sending idea to OpenAI model=%s max_tokens=%d idea_length=%d",
        resolved_model,
        resolved_max_tokens,
        len(request.idea),
    )

    t_start = time.monotonic()

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=resolved_max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
    except AuthenticationError as exc:
        raise OpenAIClientError(
            "OpenAI authentication failed — check your OPENAI_API_KEY."
        ) from exc
    except APITimeoutError as exc:
        raise OpenAIClientError(
            f"OpenAI API request timed out after {timeout:.0f}s. "
            "Try again or increase the timeout."
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIClientError(
            f"Could not connect to the OpenAI API: {exc}"
        ) from exc
    except APIStatusError as exc:
        # Map common status codes to actionable messages.
        status_code = exc.status_code
        if status_code == 401:
            detail = "authentication failed — check your OPENAI_API_KEY"
        elif status_code == 429:
            detail = "rate limit exceeded — please wait and try again"
        elif status_code == 503:
            detail = "service temporarily unavailable — check status.openai.com"
        else:
            detail = exc.message or str(exc)
        raise OpenAIClientError(
            f"OpenAI API returned an error (status {status_code}): {detail}"
        ) from exc

    elapsed = time.monotonic() - t_start

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    raw_content: str = choice.message.content or ""

    logger.debug(
        "OpenAI response received finish_reason=%s content_length=%d elapsed=%.2fs",
        finish_reason,
        len(raw_content),
        elapsed,
    )

    if finish_reason == "length":
        logger.warning(
            "OpenAI response was truncated (finish_reason='length') after %.2fs. "
            "Consider increasing OPENAI_MAX_TOKENS.",
            elapsed,
        )

    if not raw_content.strip():
        raise ResponseParseError(
            "OpenAI returned an empty response content. "
            f"finish_reason={finish_reason!r}"
        )

    data = _parse_response(raw_content)
    report = _validate_report(data)

    logger.info(
        "Viability report generated successfully score=%s competitors=%d elapsed=%.2fs",
        report.viability_score.value,
        len(report.competitors),
        elapsed,
    )

    return report
