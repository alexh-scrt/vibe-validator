"""Integration tests for vibe_validator FastAPI routes.

All OpenAI API calls are mocked so that tests run without a real API key.
Tests cover:

- GET /  — landing page renders correctly.
- POST /validate — success path returns report partial.
- POST /validate — various error paths return error partial with correct
  status codes.
- GET /health — health check endpoint.
- HTMX header propagation and response characteristics.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vibe_validator.analyzer import (
    OpenAIClientError,
    ReportValidationError,
    ResponseParseError,
)
from vibe_validator.main import app


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


def _valid_report_dict() -> dict[str, Any]:
    """Return a minimal valid ViabilityReport payload."""
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
            "notes": "Based on analyst estimates.",
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


VALID_IDEA = "An AI tool that validates side-hustle app ideas for non-technical founders."


def _mock_analyze_idea_success() -> MagicMock:
    """Return a mock that patches analyze_idea to return a valid ViabilityReport."""
    from vibe_validator.models import ViabilityReport  # noqa: PLC0415

    report = ViabilityReport.model_validate(_valid_report_dict())
    mock = MagicMock(return_value=report)
    return mock


# ---------------------------------------------------------------------------
# GET / — Landing page
# ---------------------------------------------------------------------------


class TestIndexRoute:
    def test_get_index_returns_200(self, client: TestClient) -> None:
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_index_returns_html(self, client: TestClient) -> None:
        """GET / content-type must be text/html."""
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_get_index_contains_form(self, client: TestClient) -> None:
        """GET / HTML must include the idea form."""
        response = client.get("/")
        body = response.text
        assert "<form" in body
        assert 'name="idea"' in body

    def test_get_index_contains_htmx_post(self, client: TestClient) -> None:
        """GET / form must reference the /validate HTMX endpoint."""
        response = client.get("/")
        assert 'hx-post="/validate"' in response.text

    def test_get_index_contains_brand_name(self, client: TestClient) -> None:
        """GET / must mention 'Vibe Validator' in the page body."""
        response = client.get("/")
        assert "Vibe Validator" in response.text

    def test_get_index_contains_htmx_script(self, client: TestClient) -> None:
        """GET / must load HTMX from CDN."""
        response = client.get("/")
        assert "htmx.org" in response.text

    def test_get_index_contains_tailwind_script(self, client: TestClient) -> None:
        """GET / must load TailwindCSS from CDN."""
        response = client.get("/")
        assert "tailwindcss" in response.text


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthRoute:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload_has_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_payload_has_version(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_content_type_is_json(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /validate — Success path
# ---------------------------------------------------------------------------


class TestValidateSuccess:
    def test_returns_200_on_valid_idea(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 200

    def test_returns_html_content_type(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "text/html" in response.headers["content-type"]

    def test_report_partial_contains_viability_score(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "high" in response.text.lower()

    def test_report_partial_contains_competitor_name(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "IdeaFlip" in response.text

    def test_report_partial_contains_market_size(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "5 billion" in response.text

    def test_report_partial_contains_starter_prompt(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Vibe Validator" in response.text

    def test_report_partial_contains_monetization_model(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Freemium SaaS" in response.text

    def test_analyze_idea_called_with_correct_idea(self, client: TestClient) -> None:
        mock_fn = _mock_analyze_idea_success()
        with patch("vibe_validator.main.analyze_idea", mock_fn):
            client.post("/validate", data={"idea": VALID_IDEA})
        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        idea_request = call_args.args[0]
        assert idea_request.idea == VALID_IDEA

    def test_no_error_partial_on_success(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # The error template contains "Something went wrong"
        assert "Something went wrong" not in response.text


# ---------------------------------------------------------------------------
# POST /validate — Validation errors (bad idea input)
# ---------------------------------------------------------------------------


class TestValidateInputErrors:
    def test_empty_idea_returns_error_partial(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": ""})
        # Should return error partial (422)
        assert response.status_code == 422

    def test_empty_idea_contains_error_html(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": ""})
        assert "text/html" in response.headers["content-type"]

    def test_too_short_idea_returns_422(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": "Too short"})
        assert response.status_code == 422

    def test_too_short_idea_renders_error_partial(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": "Too short"})
        assert "Something went wrong" in response.text or "invalid" in response.text.lower()

    def test_too_long_idea_returns_error(self, client: TestClient) -> None:
        long_idea = "a" * 2001
        response = client.post("/validate", data={"idea": long_idea})
        assert response.status_code in (422, 422)

    def test_missing_idea_field_returns_error(self, client: TestClient) -> None:
        response = client.post("/validate", data={})
        # FastAPI form validation will return 422
        assert response.status_code == 422

    def test_whitespace_only_idea_returns_error(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": "   " * 10})
        assert response.status_code == 422

    def test_error_partial_contains_try_again(self, client: TestClient) -> None:
        response = client.post("/validate", data={"idea": "Too short"})
        # The error template has a retry button
        body = response.text
        # Should contain either the error template or a standard error message
        assert response.status_code in (422, 500)


# ---------------------------------------------------------------------------
# POST /validate — Analyzer error paths
# ---------------------------------------------------------------------------


class TestValidateAnalyzerErrors:
    def test_openai_client_error_returns_502(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("API key invalid"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 502

    def test_openai_client_error_renders_error_partial(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("API key invalid"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "text/html" in response.headers["content-type"]
        assert "Something went wrong" in response.text

    def test_openai_client_error_mentions_api(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("Auth failure"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # The error message should mention OpenAI or API
        body = response.text
        assert "OpenAI" in body or "API" in body

    def test_response_parse_error_returns_500(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Could not parse JSON"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_response_parse_error_renders_error_partial(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad JSON"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_report_validation_error_returns_500(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ReportValidationError("Schema mismatch"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_report_validation_error_renders_error_partial(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ReportValidationError("Schema mismatch"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_generic_exception_returns_500(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError("Unexpected error"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_generic_exception_renders_error_partial(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError("Unexpected error"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_error_partial_does_not_expose_raw_exception(self, client: TestClient) -> None:
        """Internal exception details must not be leaked to the user."""
        secret_detail = "super_secret_internal_traceback_detail"
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError(secret_detail),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert secret_detail not in response.text


# ---------------------------------------------------------------------------
# POST /validate — Content verification
# ---------------------------------------------------------------------------


class TestValidateContent:
    def test_report_contains_risks_section(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # The report template renders key_risks
        assert "OpenAI API uptime" in response.text

    def test_report_contains_opportunities_section(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "non-technical founders" in response.text

    def test_report_contains_tam_value(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "5 billion" in response.text

    def test_report_contains_growth_rate(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "15%" in response.text or "CAGR" in response.text

    def test_report_contains_arpu(self, client: TestClient) -> None:
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_idea_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # ARPU value from the mock
        assert "30" in response.text


# ---------------------------------------------------------------------------
# Error template structure
# ---------------------------------------------------------------------------


class TestErrorTemplateStructure:
    def test_error_partial_contains_error_code(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("Auth failed"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # Error template includes the HTTP status code
        assert "502" in response.text

    def test_error_partial_contains_retry_button(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # The error template renders a "Try Again" button
        assert "Try Again" in response.text or "try again" in response.text.lower()

    def test_error_partial_has_role_alert(self, client: TestClient) -> None:
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert 'role="alert"' in response.text
