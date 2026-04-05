"""Unit tests for vibe_validator.analyzer.

All OpenAI API calls are mocked so that tests run without a real API key and
execute quickly.  Tests cover:

- Successful end-to-end parsing of a well-formed LLM response.
- JSON extraction from markdown-fenced responses.
- Error handling for auth failures, connection errors, and API errors.
- Empty response content handling.
- JSON decode errors.
- Pydantic validation failures.
- Custom client injection (important for testability).
- Environment variable resolution for model and max_tokens.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, AuthenticationError

from vibe_validator.analyzer import (
    AnalyzerError,
    OpenAIClientError,
    ResponseParseError,
    ReportValidationError,
    _extract_json,
    _parse_response,
    _validate_report,
    analyze_idea,
)
from vibe_validator.models import IdeaRequest, ViabilityReport, ViabilityScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_report_dict() -> dict[str, Any]:
    """Return a minimal valid report payload matching the ViabilityReport schema."""
    return {
        "idea_summary": "An AI tool that validates side-hustle app ideas.",
        "viability_score": "high",
        "viability_rationale": (
            "The market for founder tooling is large and this fills a genuine gap "
            "for non-technical builders who want instant signal."
        ),
        "market_size": {
            "tier": "medium",
            "tam": "$5 billion globally by 2027",
            "sam": "$800 million for English-speaking markets",
            "growth_rate": "~15% CAGR through 2028",
            "notes": "Based on analyst estimates for the productivity SaaS sector.",
        },
        "competitors": [
            {
                "name": "IdeaFlip",
                "description": "Visual brainstorming tool for early-stage ideation.",
                "url": "https://ideaflip.com",
                "differentiator": "Focus on viability scoring rather than brainstorming.",
            }
        ],
        "monetization_models": [
            {
                "model_name": "Freemium SaaS",
                "description": "Free tier with 3 validations/month; $19/month for unlimited.",
                "estimated_arpu": "$15 – $30 / user / month",
                "pros": ["Low barrier to entry"],
                "cons": ["High churn risk at free tier"],
            }
        ],
        "key_risks": ["Dependence on OpenAI API uptime and pricing."],
        "key_opportunities": ["Massive wave of non-technical founders shipping with AI."],
        "starter_prompt": (
            "Build a FastAPI web app called Vibe Validator. It should accept a text "
            "description of a side-hustle app idea and return a structured viability "
            "report including market size, competitors, monetisation models, and risks. "
            "Use HTMX for the frontend, TailwindCSS for styling, and OpenAI GPT-4o for "
            "the analysis. Store no user data."
        ),
    }


def _make_mock_client(content: str, finish_reason: str = "stop") -> MagicMock:
    """Build a mock OpenAI client that returns the given content string."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _valid_idea_request() -> IdeaRequest:
    return IdeaRequest(idea="An AI tool that validates side-hustle app ideas for founders.")


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json_object(self) -> None:
        raw = '{"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_json_with_leading_text(self) -> None:
        raw = 'Here is the JSON: {"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_json_with_markdown_fence(self) -> None:
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_json_with_plain_fence(self) -> None:
        raw = '```\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_nested_json_object(self) -> None:
        raw = '{"outer": {"inner": 42}}'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == 42

    def test_no_json_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="Could not locate"):
            _extract_json("No JSON here at all.")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ResponseParseError):
            _extract_json("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ResponseParseError):
            _extract_json("   \n\t  ")


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json_dict(self) -> None:
        raw = json.dumps({"hello": "world"})
        result = _parse_response(raw)
        assert result == {"hello": "world"}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="invalid JSON"):
            _parse_response("{not valid json}")

    def test_json_array_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="Expected a JSON object"):
            _parse_response("[1, 2, 3]")

    def test_nested_structure_preserved(self) -> None:
        payload = {"a": {"b": [1, 2, 3]}}
        raw = json.dumps(payload)
        result = _parse_response(raw)
        assert result["a"]["b"] == [1, 2, 3]

    def test_response_with_markdown_fence(self) -> None:
        payload = {"key": "value"}
        raw = f'```json\n{json.dumps(payload)}\n```'
        result = _parse_response(raw)
        assert result == payload


# ---------------------------------------------------------------------------
# _validate_report tests
# ---------------------------------------------------------------------------


class TestValidateReport:
    def test_valid_data_returns_report(self) -> None:
        data = _valid_report_dict()
        report = _validate_report(data)
        assert isinstance(report, ViabilityReport)
        assert report.viability_score == ViabilityScore.HIGH

    def test_invalid_data_raises(self) -> None:
        data = {"idea_summary": "Too short"}
        with pytest.raises(ReportValidationError, match="model validation"):
            _validate_report(data)

    def test_error_message_contains_field_info(self) -> None:
        data = _valid_report_dict()
        data.pop("viability_score")
        with pytest.raises(ReportValidationError) as exc_info:
            _validate_report(data)
        assert "viability_score" in str(exc_info.value)


# ---------------------------------------------------------------------------
# analyze_idea — success path
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaSuccess:
    def test_returns_viability_report(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)

        assert isinstance(report, ViabilityReport)
        assert report.viability_score == ViabilityScore.HIGH

    def test_openai_create_called_once(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        mock_client.chat.completions.create.assert_called_once()

    def test_messages_passed_to_api(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        assert any(m["role"] == "system" for m in messages)
        assert any(m["role"] == "user" for m in messages)

    def test_custom_model_passed_to_api(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client, model="gpt-3.5-turbo")

        call_kwargs = mock_client.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert model_used == "gpt-3.5-turbo"

    def test_custom_max_tokens_passed_to_api(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client, max_tokens=512)

        call_kwargs = mock_client.chat.completions.create.call_args
        max_tokens_used = call_kwargs.kwargs.get("max_tokens")
        assert max_tokens_used == 512

    def test_markdown_fenced_response_parsed_correctly(self) -> None:
        payload = _valid_report_dict()
        fenced = f'```json\n{json.dumps(payload)}\n```'
        mock_client = _make_mock_client(fenced)
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.viability_score == ViabilityScore.HIGH

    def test_truncated_response_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload), finish_reason="length")
        request = _valid_idea_request()

        with caplog.at_level(logging.WARNING, logger="vibe_validator.analyzer"):
            analyze_idea(request, client=mock_client)

        assert any("truncated" in record.message for record in caplog.records)

    def test_model_defaults_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model")
        assert model_used == "gpt-4-turbo"

    def test_max_tokens_defaults_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "1024")
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        max_tokens_used = call_kwargs.kwargs.get("max_tokens")
        assert max_tokens_used == 1024


# ---------------------------------------------------------------------------
# analyze_idea — error paths
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaErrors:
    def test_authentication_error_raises_openai_client_error(self) -> None:
        mock_client = MagicMock()
        mock_request = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Incorrect API key",
            response=MagicMock(status_code=401),
            body={},
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="authentication failed"):
            analyze_idea(request, client=mock_client)

    def test_connection_error_raises_openai_client_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="connect"):
            analyze_idea(request, client=mock_client)

    def test_api_status_error_raises_openai_client_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={},
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="429"):
            analyze_idea(request, client=mock_client)

    def test_empty_content_raises_response_parse_error(self) -> None:
        mock_client = _make_mock_client("")
        request = _valid_idea_request()

        with pytest.raises(ResponseParseError, match="empty response"):
            analyze_idea(request, client=mock_client)

    def test_invalid_json_raises_response_parse_error(self) -> None:
        mock_client = _make_mock_client("This is not JSON at all!")
        request = _valid_idea_request()

        with pytest.raises(ResponseParseError):
            analyze_idea(request, client=mock_client)

    def test_valid_json_invalid_schema_raises_report_validation_error(self) -> None:
        bad_payload = {"idea_summary": "Too short", "completely": "wrong"}
        mock_client = _make_mock_client(json.dumps(bad_payload))
        request = _valid_idea_request()

        with pytest.raises(ReportValidationError):
            analyze_idea(request, client=mock_client)

    def test_missing_api_key_raises_openai_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "")
        request = _valid_idea_request()

        # No client injected — will try to build one from env
        with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
            analyze_idea(request)  # no client kwarg

    def test_analyzer_error_is_base_for_all_errors(self) -> None:
        assert issubclass(OpenAIClientError, AnalyzerError)
        assert issubclass(ResponseParseError, AnalyzerError)
        assert issubclass(ReportValidationError, AnalyzerError)


# ---------------------------------------------------------------------------
# analyze_idea — edge cases
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaEdgeCases:
    def test_response_with_extra_keys_accepted(self) -> None:
        """Extra keys in the JSON payload should not cause validation failure."""
        payload = _valid_report_dict()
        payload["extra_key"] = "should be ignored"
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert isinstance(report, ViabilityReport)

    def test_low_viability_score(self) -> None:
        payload = _valid_report_dict()
        payload["viability_score"] = "low"
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.viability_score == ViabilityScore.LOW

    def test_multiple_competitors_in_response(self) -> None:
        payload = _valid_report_dict()
        payload["competitors"] = [
            {
                "name": "IdeaFlip",
                "description": "Visual brainstorming tool.",
                "url": "https://ideaflip.com",
                "differentiator": "Focus on viability scoring.",
            },
            {
                "name": "Validate.io",
                "description": "Market validation platform for early-stage startups.",
                "url": None,
                "differentiator": "Provide AI-driven scoring instead of manual surveys.",
            },
        ]
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert len(report.competitors) == 2

    def test_idea_content_appears_in_user_message(self) -> None:
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        idea_text = "An AI tool that validates side-hustle app ideas for founders."
        request = IdeaRequest(idea=idea_text)

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert idea_text in user_message["content"]
