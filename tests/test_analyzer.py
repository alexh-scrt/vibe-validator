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
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, AuthenticationError

from vibe_validator.analyzer import (
    AnalyzerError,
    OpenAIClientError,
    ReportValidationError,
    ResponseParseError,
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
                "estimated_arpu": "$15 \u2013 $30 / user / month",
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
    """Return a valid IdeaRequest for use in tests."""
    return IdeaRequest(idea="An AI tool that validates side-hustle app ideas for founders.")


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for the _extract_json internal helper."""

    def test_plain_json_object(self) -> None:
        """A plain JSON object string must be returned unchanged."""
        raw = '{"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_json_with_leading_text(self) -> None:
        """Leading text before the JSON object must be stripped."""
        raw = 'Here is the JSON: {"key": "value"}'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_json_with_markdown_fence_json(self) -> None:
        """Markdown ```json fences must be removed before extraction."""
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_json_with_plain_fence(self) -> None:
        """Plain markdown ``` fences must be removed before extraction."""
        raw = '```\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result == '{"key": "value"}'

    def test_nested_json_object(self) -> None:
        """Nested JSON objects must be extracted correctly."""
        raw = '{"outer": {"inner": 42}}'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == 42

    def test_json_with_whitespace_padding(self) -> None:
        """Surrounding whitespace must not prevent extraction."""
        raw = '   \n  {"key": "value"}  \n  '
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_no_json_raises(self) -> None:
        """A string with no JSON object must raise ResponseParseError."""
        with pytest.raises(ResponseParseError, match="Could not locate"):
            _extract_json("No JSON here at all.")

    def test_empty_string_raises(self) -> None:
        """An empty string must raise ResponseParseError."""
        with pytest.raises(ResponseParseError):
            _extract_json("")

    def test_whitespace_only_raises(self) -> None:
        """A whitespace-only string must raise ResponseParseError."""
        with pytest.raises(ResponseParseError):
            _extract_json("   \n\t  ")

    def test_only_closing_brace_raises(self) -> None:
        """A string with only '}' must raise ResponseParseError."""
        with pytest.raises(ResponseParseError):
            _extract_json("}")

    def test_large_json_payload(self) -> None:
        """A large JSON payload must be extracted without truncation."""
        payload = {"key": "v" * 5000}
        raw = json.dumps(payload)
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert len(parsed["key"]) == 5000

    def test_json_with_array_value(self) -> None:
        """A JSON object containing arrays must be extracted correctly."""
        raw = '{"items": [1, 2, 3]}'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for the _parse_response internal helper."""

    def test_valid_json_dict(self) -> None:
        """A valid JSON dict string must be returned as a Python dict."""
        raw = json.dumps({"hello": "world"})
        result = _parse_response(raw)
        assert result == {"hello": "world"}

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON must raise ResponseParseError."""
        with pytest.raises(ResponseParseError, match="invalid JSON"):
            _parse_response("{not valid json}")

    def test_json_array_raises(self) -> None:
        """A top-level JSON array must raise ResponseParseError."""
        with pytest.raises(ResponseParseError, match="Expected a JSON object"):
            _parse_response("[1, 2, 3]")

    def test_nested_structure_preserved(self) -> None:
        """Nested structures must survive the parse round-trip."""
        payload = {"a": {"b": [1, 2, 3]}}
        raw = json.dumps(payload)
        result = _parse_response(raw)
        assert result["a"]["b"] == [1, 2, 3]

    def test_response_with_markdown_fence(self) -> None:
        """Markdown-fenced responses must be parsed correctly."""
        payload = {"key": "value"}
        raw = f'```json\n{json.dumps(payload)}\n```'
        result = _parse_response(raw)
        assert result == payload

    def test_null_value_in_dict(self) -> None:
        """JSON null values must be preserved as Python None."""
        raw = '{"url": null}'
        result = _parse_response(raw)
        assert result["url"] is None

    def test_boolean_values_preserved(self) -> None:
        """JSON boolean values must be preserved correctly."""
        raw = '{"active": true, "deleted": false}'
        result = _parse_response(raw)
        assert result["active"] is True
        assert result["deleted"] is False

    def test_numeric_values_preserved(self) -> None:
        """JSON numeric values must be preserved correctly."""
        raw = '{"count": 42, "score": 3.14}'
        result = _parse_response(raw)
        assert result["count"] == 42
        assert abs(result["score"] - 3.14) < 1e-9


# ---------------------------------------------------------------------------
# _validate_report tests
# ---------------------------------------------------------------------------


class TestValidateReport:
    """Tests for the _validate_report internal helper."""

    def test_valid_data_returns_report(self) -> None:
        """Valid data must produce a ViabilityReport instance."""
        data = _valid_report_dict()
        report = _validate_report(data)
        assert isinstance(report, ViabilityReport)
        assert report.viability_score == ViabilityScore.HIGH

    def test_invalid_data_raises(self) -> None:
        """Invalid data must raise ReportValidationError."""
        data = {"idea_summary": "Too short"}
        with pytest.raises(ReportValidationError, match="model validation"):
            _validate_report(data)

    def test_error_message_contains_field_info(self) -> None:
        """The ReportValidationError message must identify the failing field."""
        data = _valid_report_dict()
        data.pop("viability_score")
        with pytest.raises(ReportValidationError) as exc_info:
            _validate_report(data)
        assert "viability_score" in str(exc_info.value)

    def test_error_count_in_message(self) -> None:
        """The error message must include the count of validation errors."""
        data = {}  # Empty dict — many errors
        with pytest.raises(ReportValidationError) as exc_info:
            _validate_report(data)
        # Message format: "... (N error(s)):"
        assert "error" in str(exc_info.value)

    def test_valid_low_score_report(self) -> None:
        """A report with a 'low' viability_score must pass validation."""
        data = _valid_report_dict()
        data["viability_score"] = "low"
        report = _validate_report(data)
        assert report.viability_score == ViabilityScore.LOW

    def test_valid_medium_score_report(self) -> None:
        """A report with a 'medium' viability_score must pass validation."""
        data = _valid_report_dict()
        data["viability_score"] = "medium"
        report = _validate_report(data)
        assert report.viability_score == ViabilityScore.MEDIUM


# ---------------------------------------------------------------------------
# analyze_idea — success path
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaSuccess:
    """Tests for the happy path of analyze_idea."""

    def test_returns_viability_report(self) -> None:
        """A successful call must return a ViabilityReport."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)

        assert isinstance(report, ViabilityReport)
        assert report.viability_score == ViabilityScore.HIGH

    def test_openai_create_called_once(self) -> None:
        """The OpenAI completions endpoint must be called exactly once."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        mock_client.chat.completions.create.assert_called_once()

    def test_messages_contain_system_and_user_roles(self) -> None:
        """The messages list passed to the API must include system and user roles."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        roles = {m["role"] for m in messages}
        assert "system" in roles
        assert "user" in roles

    def test_custom_model_passed_to_api(self) -> None:
        """A custom model name must be forwarded to the API call."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client, model="gpt-3.5-turbo")

        call_kwargs = mock_client.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert model_used == "gpt-3.5-turbo"

    def test_custom_max_tokens_passed_to_api(self) -> None:
        """A custom max_tokens value must be forwarded to the API call."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client, max_tokens=512)

        call_kwargs = mock_client.chat.completions.create.call_args
        max_tokens_used = call_kwargs.kwargs.get("max_tokens")
        assert max_tokens_used == 512

    def test_default_temperature_is_07(self) -> None:
        """The default temperature must be 0.7."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        temperature_used = call_kwargs.kwargs.get("temperature")
        assert temperature_used == 0.7

    def test_custom_temperature_passed_to_api(self) -> None:
        """A custom temperature must be forwarded to the API call."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client, temperature=0.2)

        call_kwargs = mock_client.chat.completions.create.call_args
        temperature_used = call_kwargs.kwargs.get("temperature")
        assert temperature_used == 0.2

    def test_markdown_fenced_response_parsed_correctly(self) -> None:
        """A markdown-fenced JSON response must be parsed to a valid report."""
        payload = _valid_report_dict()
        fenced = f'```json\n{json.dumps(payload)}\n```'
        mock_client = _make_mock_client(fenced)
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.viability_score == ViabilityScore.HIGH

    def test_truncated_response_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A truncated response (finish_reason='length') must emit a warning log."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload), finish_reason="length")
        request = _valid_idea_request()

        with caplog.at_level(logging.WARNING, logger="vibe_validator.analyzer"):
            analyze_idea(request, client=mock_client)

        assert any("truncated" in record.message for record in caplog.records)

    def test_model_defaults_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When model is not specified, OPENAI_MODEL env var must be used."""
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo")
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model")
        assert model_used == "gpt-4-turbo"

    def test_max_tokens_defaults_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When max_tokens is not specified, OPENAI_MAX_TOKENS env var must be used."""
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "1024")
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        max_tokens_used = call_kwargs.kwargs.get("max_tokens")
        assert max_tokens_used == 1024

    def test_model_defaults_to_gpt4o_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When OPENAI_MODEL is not set, the model must default to 'gpt-4o'."""
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model")
        assert model_used == "gpt-4o"

    def test_max_tokens_defaults_to_2048_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When OPENAI_MAX_TOKENS is not set, max_tokens must default to 2048."""
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        max_tokens_used = call_kwargs.kwargs.get("max_tokens")
        assert max_tokens_used == 2048

    def test_response_format_json_object_requested(self) -> None:
        """The API call must request JSON object response format."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        response_format = call_kwargs.kwargs.get("response_format")
        assert response_format == {"type": "json_object"}


# ---------------------------------------------------------------------------
# analyze_idea — error paths
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaErrors:
    """Tests for error handling in analyze_idea."""

    def test_authentication_error_raises_openai_client_error(self) -> None:
        """An OpenAI AuthenticationError must be raised as OpenAIClientError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Incorrect API key",
            response=MagicMock(status_code=401),
            body={},
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="authentication failed"):
            analyze_idea(request, client=mock_client)

    def test_connection_error_raises_openai_client_error(self) -> None:
        """An OpenAI APIConnectionError must be raised as OpenAIClientError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="connect"):
            analyze_idea(request, client=mock_client)

    def test_api_status_error_429_raises_openai_client_error(self) -> None:
        """A 429 APIStatusError must be raised as OpenAIClientError with status code."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={},
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="429"):
            analyze_idea(request, client=mock_client)

    def test_api_status_error_500_raises_openai_client_error(self) -> None:
        """A 500 APIStatusError must be raised as OpenAIClientError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500),
            body={},
        )
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="500"):
            analyze_idea(request, client=mock_client)

    def test_empty_content_raises_response_parse_error(self) -> None:
        """An empty response content must raise ResponseParseError."""
        mock_client = _make_mock_client("")
        request = _valid_idea_request()

        with pytest.raises(ResponseParseError, match="empty response"):
            analyze_idea(request, client=mock_client)

    def test_whitespace_only_content_raises_response_parse_error(self) -> None:
        """A whitespace-only response content must raise ResponseParseError."""
        mock_client = _make_mock_client("   \n   ")
        request = _valid_idea_request()

        with pytest.raises(ResponseParseError):
            analyze_idea(request, client=mock_client)

    def test_invalid_json_raises_response_parse_error(self) -> None:
        """A non-JSON response must raise ResponseParseError."""
        mock_client = _make_mock_client("This is not JSON at all!")
        request = _valid_idea_request()

        with pytest.raises(ResponseParseError):
            analyze_idea(request, client=mock_client)

    def test_valid_json_invalid_schema_raises_report_validation_error(self) -> None:
        """Valid JSON that fails schema validation must raise ReportValidationError."""
        bad_payload = {"idea_summary": "Too short", "completely": "wrong"}
        mock_client = _make_mock_client(json.dumps(bad_payload))
        request = _valid_idea_request()

        with pytest.raises(ReportValidationError):
            analyze_idea(request, client=mock_client)

    def test_missing_api_key_raises_openai_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing OPENAI_API_KEY must raise OpenAIClientError when no client injected."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "")
        request = _valid_idea_request()

        with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
            analyze_idea(request)  # no client kwarg

    def test_analyzer_error_is_base_for_openai_client_error(self) -> None:
        """OpenAIClientError must be a subclass of AnalyzerError."""
        assert issubclass(OpenAIClientError, AnalyzerError)

    def test_analyzer_error_is_base_for_response_parse_error(self) -> None:
        """ResponseParseError must be a subclass of AnalyzerError."""
        assert issubclass(ResponseParseError, AnalyzerError)

    def test_analyzer_error_is_base_for_report_validation_error(self) -> None:
        """ReportValidationError must be a subclass of AnalyzerError."""
        assert issubclass(ReportValidationError, AnalyzerError)

    def test_all_errors_are_exceptions(self) -> None:
        """All custom exception types must be subclasses of Exception."""
        assert issubclass(AnalyzerError, Exception)
        assert issubclass(OpenAIClientError, Exception)
        assert issubclass(ResponseParseError, Exception)
        assert issubclass(ReportValidationError, Exception)


# ---------------------------------------------------------------------------
# analyze_idea — edge cases
# ---------------------------------------------------------------------------


class TestAnalyzeIdeaEdgeCases:
    """Edge-case tests for analyze_idea."""

    def test_response_with_extra_keys_accepted(self) -> None:
        """Extra keys in the JSON payload must not cause validation failure."""
        payload = _valid_report_dict()
        payload["extra_key"] = "should be ignored"
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert isinstance(report, ViabilityReport)

    def test_low_viability_score_parsed(self) -> None:
        """A 'low' viability_score in the response must be parsed correctly."""
        payload = _valid_report_dict()
        payload["viability_score"] = "low"
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.viability_score == ViabilityScore.LOW

    def test_medium_viability_score_parsed(self) -> None:
        """A 'medium' viability_score in the response must be parsed correctly."""
        payload = _valid_report_dict()
        payload["viability_score"] = "medium"
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.viability_score == ViabilityScore.MEDIUM

    def test_multiple_competitors_in_response(self) -> None:
        """Multiple competitors in the response must all be parsed correctly."""
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
        """The idea text must appear in the user message sent to the API."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        idea_text = "An AI tool that validates side-hustle app ideas for founders."
        request = IdeaRequest(idea=idea_text)

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        user_message = next(m for m in messages if m["role"] == "user")
        assert idea_text in user_message["content"]

    def test_system_message_is_non_empty(self) -> None:
        """The system message content must not be empty."""
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        analyze_idea(request, client=mock_client)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[1]
        system_message = next(m for m in messages if m["role"] == "system")
        assert len(system_message["content"]) > 100

    def test_null_url_in_competitor_accepted(self) -> None:
        """A null URL in a competitor must be accepted and parsed as None."""
        payload = _valid_report_dict()
        payload["competitors"] = [
            {
                "name": "SomeProduct",
                "description": "A product without a URL.",
                "url": None,
                "differentiator": "We have a URL, they don't.",
            }
        ]
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert report.competitors[0].url is None

    def test_multiple_monetization_models_parsed(self) -> None:
        """Multiple monetization models in the response must all be parsed."""
        payload = _valid_report_dict()
        payload["monetization_models"] = [
            {
                "model_name": "Freemium SaaS",
                "description": "Free tier limited; paid tier unlimited.",
                "estimated_arpu": "$19/month",
                "pros": ["Easy adoption"],
                "cons": ["High support costs"],
            },
            {
                "model_name": "Usage-based",
                "description": "Charge per validation report generated.",
                "estimated_arpu": "$5 per report",
                "pros": ["Scales with value"],
                "cons": ["Unpredictable revenue"],
            },
        ]
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert len(report.monetization_models) == 2

    def test_multiple_risks_and_opportunities_parsed(self) -> None:
        """Multiple risks and opportunities must all be stored in the report."""
        payload = _valid_report_dict()
        payload["key_risks"] = ["Risk A", "Risk B", "Risk C"]
        payload["key_opportunities"] = ["Opportunity X", "Opportunity Y"]
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        report = analyze_idea(request, client=mock_client)
        assert len(report.key_risks) == 3
        assert len(report.key_opportunities) == 2

    def test_injected_client_is_used_instead_of_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a client is injected, no OPENAI_API_KEY lookup should occur."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        payload = _valid_report_dict()
        mock_client = _make_mock_client(json.dumps(payload))
        request = _valid_idea_request()

        # Should not raise even though OPENAI_API_KEY is missing
        report = analyze_idea(request, client=mock_client)
        assert isinstance(report, ViabilityReport)
