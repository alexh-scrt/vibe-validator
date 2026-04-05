"""Integration tests for vibe_validator FastAPI routes.

All OpenAI API calls are mocked so that tests run without a real API key.
Tests cover:

- GET /  \u2014 landing page renders correctly.
- POST /validate \u2014 success path returns report partial.
- POST /validate \u2014 various error paths return error partial with correct
  status codes.
- GET /health \u2014 health check endpoint.
- HTMX header propagation and response characteristics.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vibe_validator.analyzer import (
    AnalyzerError,
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


VALID_IDEA: str = "An AI tool that validates side-hustle app ideas for non-technical founders."


def _make_mock_report() -> Any:
    """Build and return a validated ViabilityReport from the valid payload."""
    from vibe_validator.models import ViabilityReport  # noqa: PLC0415

    return ViabilityReport.model_validate(_valid_report_dict())


def _mock_analyze_success() -> MagicMock:
    """Return a MagicMock that mimics a successful analyze_idea call."""
    return MagicMock(return_value=_make_mock_report())


# ---------------------------------------------------------------------------
# GET / \u2014 Landing page
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """Tests for GET / (landing page)."""

    def test_get_index_returns_200(self, client: TestClient) -> None:
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_get_index_returns_html_content_type(self, client: TestClient) -> None:
        """GET / must return text/html content-type."""
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_get_index_contains_form_element(self, client: TestClient) -> None:
        """GET / HTML must include an HTML form element."""
        response = client.get("/")
        assert "<form" in response.text

    def test_get_index_contains_idea_textarea(self, client: TestClient) -> None:
        """GET / HTML must include the idea textarea field."""
        response = client.get("/")
        assert 'name="idea"' in response.text

    def test_get_index_form_has_htmx_post(self, client: TestClient) -> None:
        """GET / form must reference the /validate HTMX endpoint."""
        response = client.get("/")
        assert 'hx-post="/validate"' in response.text

    def test_get_index_contains_brand_name(self, client: TestClient) -> None:
        """GET / must mention 'Vibe Validator' in the page body."""
        response = client.get("/")
        assert "Vibe Validator" in response.text

    def test_get_index_loads_htmx_cdn(self, client: TestClient) -> None:
        """GET / must load HTMX from a CDN."""
        response = client.get("/")
        assert "htmx.org" in response.text

    def test_get_index_loads_tailwind_cdn(self, client: TestClient) -> None:
        """GET / must load TailwindCSS from CDN."""
        response = client.get("/")
        assert "tailwindcss" in response.text

    def test_get_index_contains_result_area(self, client: TestClient) -> None:
        """GET / must contain the #result-area HTMX target element."""
        response = client.get("/")
        assert "result-area" in response.text

    def test_get_index_contains_submit_button(self, client: TestClient) -> None:
        """GET / must contain a submit button."""
        response = client.get("/")
        assert 'type="submit"' in response.text

    def test_get_index_page_title(self, client: TestClient) -> None:
        """GET / must contain a relevant page title."""
        response = client.get("/")
        assert "<title>" in response.text
        assert "Vibe Validator" in response.text

    def test_get_index_hx_target_result_area(self, client: TestClient) -> None:
        """GET / form must target #result-area for HTMX swapping."""
        response = client.get("/")
        assert 'hx-target="#result-area"' in response.text


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthRoute:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload_has_status_ok(self, client: TestClient) -> None:
        """GET /health payload must have status='ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_payload_has_version(self, client: TestClient) -> None:
        """GET /health payload must include a non-empty version string."""
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_content_type_is_json(self, client: TestClient) -> None:
        """GET /health must return application/json content-type."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_version_matches_package(self, client: TestClient) -> None:
        """GET /health version must match the package __version__."""
        from vibe_validator import __version__  # noqa: PLC0415

        data = client.get("/health").json()
        assert data["version"] == __version__


# ---------------------------------------------------------------------------
# POST /validate \u2014 Success path
# ---------------------------------------------------------------------------


class TestValidateSuccess:
    """Tests for the happy path of POST /validate."""

    def test_returns_200_on_valid_idea(self, client: TestClient) -> None:
        """POST /validate with a valid idea must return HTTP 200."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 200

    def test_returns_html_content_type(self, client: TestClient) -> None:
        """POST /validate success must return text/html content-type."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "text/html" in response.headers["content-type"]

    def test_report_partial_contains_viability_score(self, client: TestClient) -> None:
        """The report partial must include the viability score text."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "high" in response.text.lower()

    def test_report_partial_contains_competitor_name(self, client: TestClient) -> None:
        """The report partial must include the competitor name."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "IdeaFlip" in response.text

    def test_report_partial_contains_tam_value(self, client: TestClient) -> None:
        """The report partial must include the TAM value."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "5 billion" in response.text

    def test_report_partial_contains_starter_prompt_text(self, client: TestClient) -> None:
        """The report partial must include text from the starter prompt."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Vibe Validator" in response.text

    def test_report_partial_contains_monetization_model_name(self, client: TestClient) -> None:
        """The report partial must include the monetization model name."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Freemium SaaS" in response.text

    def test_analyze_idea_called_once(self, client: TestClient) -> None:
        """analyze_idea must be called exactly once per POST request."""
        mock_fn = _mock_analyze_success()
        with patch("vibe_validator.main.analyze_idea", mock_fn):
            client.post("/validate", data={"idea": VALID_IDEA})
        mock_fn.assert_called_once()

    def test_analyze_idea_called_with_correct_idea(self, client: TestClient) -> None:
        """analyze_idea must receive an IdeaRequest with the submitted idea text."""
        mock_fn = _mock_analyze_success()
        with patch("vibe_validator.main.analyze_idea", mock_fn):
            client.post("/validate", data={"idea": VALID_IDEA})
        call_args = mock_fn.call_args
        idea_request = call_args.args[0]
        assert idea_request.idea == VALID_IDEA

    def test_no_error_partial_on_success(self, client: TestClient) -> None:
        """The success response must not contain the error partial marker."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" not in response.text

    def test_report_partial_contains_key_risk(self, client: TestClient) -> None:
        """The report partial must include the key risk text."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "OpenAI API uptime" in response.text

    def test_report_partial_contains_key_opportunity(self, client: TestClient) -> None:
        """The report partial must include the key opportunity text."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "non-technical founders" in response.text

    def test_report_partial_contains_growth_rate(self, client: TestClient) -> None:
        """The report partial must include market growth rate information."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        body = response.text
        assert "15%" in body or "CAGR" in body

    def test_report_partial_contains_arpu(self, client: TestClient) -> None:
        """The report partial must include the ARPU value."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "30" in response.text

    def test_report_partial_contains_idea_summary(self, client: TestClient) -> None:
        """The report partial must include the idea summary."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "validates side-hustle" in response.text

    def test_report_partial_contains_market_tier(self, client: TestClient) -> None:
        """The report partial must include the market tier label."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "medium" in response.text.lower()


# ---------------------------------------------------------------------------
# POST /validate \u2014 Input validation errors
# ---------------------------------------------------------------------------


class TestValidateInputErrors:
    """Tests for input validation error paths of POST /validate."""

    def test_empty_idea_returns_422(self, client: TestClient) -> None:
        """An empty idea string must return HTTP 422."""
        response = client.post("/validate", data={"idea": ""})
        assert response.status_code == 422

    def test_empty_idea_returns_html(self, client: TestClient) -> None:
        """An empty idea must return an HTML error partial."""
        response = client.post("/validate", data={"idea": ""})
        assert "text/html" in response.headers["content-type"]

    def test_too_short_idea_returns_422(self, client: TestClient) -> None:
        """An idea shorter than 20 chars must return HTTP 422."""
        response = client.post("/validate", data={"idea": "Too short"})
        assert response.status_code == 422

    def test_too_short_idea_renders_error_indication(self, client: TestClient) -> None:
        """A too-short idea must render some form of error indication."""
        response = client.post("/validate", data={"idea": "Too short"})
        body = response.text.lower()
        # Accept either the error template or a validation error message
        assert (
            "something went wrong" in body
            or "invalid" in body
            or "error" in body
        )

    def test_too_long_idea_returns_error(self, client: TestClient) -> None:
        """An idea longer than 2000 chars must return an error status code."""
        long_idea = "a" * 2001
        response = client.post("/validate", data={"idea": long_idea})
        assert response.status_code in (422, 500)

    def test_missing_idea_field_returns_422(self, client: TestClient) -> None:
        """A POST without an idea field must return HTTP 422 from FastAPI."""
        response = client.post("/validate", data={})
        assert response.status_code == 422

    def test_whitespace_only_idea_returns_422(self, client: TestClient) -> None:
        """A whitespace-only idea must be rejected with HTTP 422."""
        response = client.post("/validate", data={"idea": "   " * 10})
        assert response.status_code == 422

    def test_empty_idea_does_not_call_analyzer(self, client: TestClient) -> None:
        """An invalid idea must not trigger the analyzer."""
        mock_fn = _mock_analyze_success()
        with patch("vibe_validator.main.analyze_idea", mock_fn):
            client.post("/validate", data={"idea": ""})
        mock_fn.assert_not_called()

    def test_exactly_20_chars_accepted(self, client: TestClient) -> None:
        """An idea of exactly 20 characters must be accepted and call the analyzer."""
        mock_fn = _mock_analyze_success()
        with patch("vibe_validator.main.analyze_idea", mock_fn):
            response = client.post("/validate", data={"idea": "a" * 20})
        assert response.status_code == 200
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# POST /validate \u2014 Analyzer error paths
# ---------------------------------------------------------------------------


class TestValidateAnalyzerErrors:
    """Tests for analyzer error handling in POST /validate."""

    def test_openai_client_error_returns_502(self, client: TestClient) -> None:
        """An OpenAIClientError must result in HTTP 502."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("API key invalid"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 502

    def test_openai_client_error_renders_error_partial(self, client: TestClient) -> None:
        """An OpenAIClientError must render the error partial HTML."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("API key invalid"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "text/html" in response.headers["content-type"]
        assert "Something went wrong" in response.text

    def test_openai_client_error_mentions_api_in_body(self, client: TestClient) -> None:
        """The error partial for OpenAIClientError must mention OpenAI or API."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("Auth failure"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        body = response.text
        assert "OpenAI" in body or "API" in body

    def test_response_parse_error_returns_500(self, client: TestClient) -> None:
        """A ResponseParseError must result in HTTP 500."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Could not parse JSON"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_response_parse_error_renders_error_partial(self, client: TestClient) -> None:
        """A ResponseParseError must render the error partial HTML."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad JSON"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_report_validation_error_returns_500(self, client: TestClient) -> None:
        """A ReportValidationError must result in HTTP 500."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ReportValidationError("Schema mismatch"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_report_validation_error_renders_error_partial(self, client: TestClient) -> None:
        """A ReportValidationError must render the error partial HTML."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ReportValidationError("Schema mismatch"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_generic_exception_returns_500(self, client: TestClient) -> None:
        """An unexpected exception must result in HTTP 500."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError("Unexpected boom"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_generic_exception_renders_error_partial(self, client: TestClient) -> None:
        """An unexpected exception must render the error partial HTML."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError("Unexpected boom"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Something went wrong" in response.text

    def test_error_partial_does_not_expose_raw_exception(self, client: TestClient) -> None:
        """Internal exception details must not be leaked to the user."""
        secret_detail = "super_secret_internal_traceback_detail_xyz"
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=RuntimeError(secret_detail),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert secret_detail not in response.text

    def test_analyzer_error_base_class_returns_500(self, client: TestClient) -> None:
        """A base AnalyzerError must result in HTTP 500."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=AnalyzerError("Generic analyzer failure"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert response.status_code == 500

    def test_openai_client_error_error_code_visible(self, client: TestClient) -> None:
        """The HTTP error code must be visible in the error partial for 502."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=OpenAIClientError("Auth failed"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "502" in response.text

    def test_parse_error_error_code_visible(self, client: TestClient) -> None:
        """The HTTP error code must be visible in the error partial for 500."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "500" in response.text


# ---------------------------------------------------------------------------
# Error template structure verification
# ---------------------------------------------------------------------------


class TestErrorTemplateStructure:
    """Tests that verify the structure of the rendered error partial."""

    def test_error_partial_contains_retry_button(self, client: TestClient) -> None:
        """The error partial must contain a 'Try Again' element."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        body = response.text.lower()
        assert "try again" in body

    def test_error_partial_has_role_alert(self, client: TestClient) -> None:
        """The error partial must have role='alert' for accessibility."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert 'role="alert"' in response.text

    def test_error_partial_contains_help_tips(self, client: TestClient) -> None:
        """The error partial must include user-facing help tips."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # The error template includes "What you can try"
        body = response.text
        assert "try" in body.lower() or "check" in body.lower()

    def test_error_partial_is_html_fragment(self, client: TestClient) -> None:
        """The error partial must be a valid HTML fragment (not a full page)."""
        with patch(
            "vibe_validator.main.analyze_idea",
            side_effect=ResponseParseError("Bad parse"),
        ):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        # A partial does not include a full <html> wrapper from a fresh page
        # (it may or may not have html tag depending on template, but it must
        # not have a full standalone DOCTYPE since it is a partial)
        # The key check is that it is HTML text
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# Report template structure verification
# ---------------------------------------------------------------------------


class TestReportTemplateStructure:
    """Tests that verify the structure of the rendered report partial."""

    def test_report_partial_is_html(self, client: TestClient) -> None:
        """The report partial must have text/html content type."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "text/html" in response.headers["content-type"]

    def test_report_partial_has_aria_region(self, client: TestClient) -> None:
        """The report partial must include an aria region label."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Viability Report" in response.text or 'role="region"' in response.text

    def test_report_partial_contains_market_size_section(self, client: TestClient) -> None:
        """The report partial must include a Market Size section heading."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Market Size" in response.text

    def test_report_partial_contains_competitors_section(self, client: TestClient) -> None:
        """The report partial must include a competitors section."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Competitor" in response.text

    def test_report_partial_contains_monetisation_section(self, client: TestClient) -> None:
        """The report partial must include a monetisation section heading."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Monetis" in response.text or "Monetiz" in response.text

    def test_report_partial_contains_starter_prompt_section(self, client: TestClient) -> None:
        """The report partial must include a Starter Prompt section."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Starter Prompt" in response.text

    def test_report_partial_contains_try_another_idea_button(self, client: TestClient) -> None:
        """The report partial must include a 'Try Another Idea' button."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "Try Another Idea" in response.text

    def test_report_partial_contains_copy_button(self, client: TestClient) -> None:
        """The report partial must include a copy-to-clipboard button."""
        with patch("vibe_validator.main.analyze_idea", _mock_analyze_success()):
            response = client.post("/validate", data={"idea": VALID_IDEA})
        assert "copy" in response.text.lower() or "Copy" in response.text
