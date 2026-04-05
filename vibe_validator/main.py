"""FastAPI application entrypoint for vibe_validator.

This module wires up the FastAPI application with:

- ``GET /``  — serves the main landing page (``index.html``).
- ``POST /validate`` — HTMX-driven endpoint that accepts the idea form,
  calls the analyzer, and returns either ``report.html`` or ``error.html``
  as an HTML partial for inline page replacement.
- ``GET /health`` — lightweight health-check used by load balancers / tests.

The module also exposes a ``run()`` entry point used by the
``vibe-validator`` console script declared in ``pyproject.toml``.

Usage::

    # Development (auto-reload)
    uvicorn vibe_validator.main:app --reload

    # Via console script
    vibe-validator
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from vibe_validator import __version__
from vibe_validator.analyzer import (
    AnalyzerError,
    OpenAIClientError,
    ReportValidationError,
    ResponseParseError,
    analyze_idea,
)
from vibe_validator.models import IdeaRequest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load .env before anything else so environment variables are available
# when the module is imported by Uvicorn.
load_dotenv(override=False)

# Configure logging with a consistent format for all modules.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template configuration
# ---------------------------------------------------------------------------

_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vibe Validator",
    description=(
        "AI-powered viability reports for side-hustle app ideas. "
        "Instant market analysis, competitor surfacing, monetisation "
        "suggestions, and a ready-to-paste starter prompt."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Human-friendly labels for user-facing error messages.
_USER_FRIENDLY_ERRORS: dict[type[Exception], str] = {
    OpenAIClientError: (
        "We could not reach the OpenAI API. "
        "Please ensure a valid OPENAI_API_KEY is configured and try again."
    ),
    ResponseParseError: (
        "The AI returned a response we could not understand. "
        "This occasionally happens with complex ideas — please try rephrasing "
        "your idea or try again in a moment."
    ),
    ReportValidationError: (
        "The AI returned a report that did not match the expected structure. "
        "Please try again. If the issue persists, try simplifying or "
        "rephrasing your idea."
    ),
}


def _is_htmx_request(request: Request) -> bool:
    """Return ``True`` if the request originated from HTMX.

    HTMX sets the ``HX-Request`` header on every request it makes.

    Parameters
    ----------
    request:
        The incoming FastAPI ``Request`` object.

    Returns
    -------
    bool
        ``True`` when the ``HX-Request`` header is present and truthy.
    """
    return request.headers.get("HX-Request", "").lower() == "true"


def _render_error(
    request: Request,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    *,
    log_level: str = "warning",
) -> HTMLResponse:
    """Render the ``error.html`` partial and return an ``HTMLResponse``.

    Parameters
    ----------
    request:
        The current request (passed to the template context).
    message:
        Human-readable error message to display in the template.
    status_code:
        HTTP status code for the response. Defaults to 500.
    log_level:
        Logger level to use when recording the error (``'warning'``,
        ``'error'``, or ``'exception'``). Defaults to ``'warning'``.

    Returns
    -------
    HTMLResponse
        Rendered error partial with the appropriate status code.
    """
    log_fn = getattr(logger, log_level, logger.warning)
    log_fn("Rendering error partial: status=%d message=%r", status_code, message)

    context: dict[str, Any] = {
        "request": request,
        "error_message": message,
        "error_code": status_code,
    }
    return templates.TemplateResponse(
        "error.html",
        context,
        status_code=status_code,
    )


def _sanitise_idea(raw: str) -> str:
    """Strip and normalise whitespace from the submitted idea string.

    Parameters
    ----------
    raw:
        Raw idea string from the form submission.

    Returns
    -------
    str
        Trimmed idea string.
    """
    return raw.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Landing page",
    tags=["pages"],
    include_in_schema=False,
)
async def index(request: Request) -> HTMLResponse:
    """Serve the main landing page with the idea submission form.

    Parameters
    ----------
    request:
        The incoming HTTP request.

    Returns
    -------
    HTMLResponse
        Rendered ``index.html`` template.
    """
    logger.debug("Serving index page")
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post(
    "/validate",
    response_class=HTMLResponse,
    summary="Validate an app idea",
    tags=["analysis"],
    responses={
        200: {"description": "Report HTML partial rendered successfully"},
        422: {"description": "Idea validation error — error HTML partial"},
        502: {"description": "OpenAI API unreachable — error HTML partial"},
        500: {"description": "Server or parse error — error HTML partial"},
    },
)
async def validate(
    request: Request,
    idea: str = Form(
        ...,
        min_length=1,
        description="Free-text description of the app idea to validate.",
    ),
) -> HTMLResponse:
    """Validate a side-hustle app idea and return a viability report partial.

    This endpoint is designed to be called by HTMX from the landing page
    form.  It accepts the ``idea`` field via an HTML form POST, runs the
    AI analysis pipeline, and returns an HTML partial that HTMX swaps into
    ``#result-area``.

    On success, the ``report.html`` partial is rendered with the
    :class:`~vibe_validator.models.ViabilityReport` in context.

    On any failure (validation error, OpenAI error, parse error) the
    ``error.html`` partial is rendered with a human-readable message.

    Parameters
    ----------
    request:
        The incoming HTTP request (used for template context and header
        inspection).
    idea:
        The idea text submitted via the HTML form.

    Returns
    -------
    HTMLResponse
        Either ``report.html`` (HTTP 200) or ``error.html`` (HTTP 4xx/5xx)
        rendered as an HTML partial.
    """
    # ------------------------------------------------------------------
    # 1. Sanitise and validate the incoming form data via Pydantic.
    # ------------------------------------------------------------------
    sanitised_idea = _sanitise_idea(idea)

    try:
        idea_request = IdeaRequest(idea=sanitised_idea)
    except ValidationError as exc:
        # Collect all field-level error messages into a single readable string.
        messages = "; ".join(
            e["msg"] for e in exc.errors()
        )
        logger.info(
            "Idea validation failed for submission (length=%d): %s",
            len(sanitised_idea),
            messages,
        )
        return _render_error(
            request,
            (
                f"Your idea description is invalid: {messages}. "
                "Please ensure it is between 20 and 2\u202f000 characters and "
                "contains meaningful content."
            ),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            log_level="info",
        )

    # ------------------------------------------------------------------
    # 2. Run the AI analysis pipeline.
    # ------------------------------------------------------------------
    logger.info(
        "Starting viability analysis (idea_length=%d)",
        len(idea_request.idea),
    )

    try:
        report = analyze_idea(idea_request)

    except OpenAIClientError as exc:
        logger.error("OpenAI client error during /validate: %s", exc)
        # Include a sanitised version of the technical detail for
        # transparency without leaking internal tracebacks.
        technical_hint = str(exc)
        if len(technical_hint) > 200:
            technical_hint = technical_hint[:200] + "\u2026"
        return _render_error(
            request,
            (
                "We could not reach the OpenAI API. Please check that a valid "
                f"OPENAI_API_KEY is configured and try again. "
                f"Detail: {technical_hint}"
            ),
            status_code=status.HTTP_502_BAD_GATEWAY,
            log_level="error",
        )

    except ResponseParseError as exc:
        logger.error("Response parse error during /validate: %s", exc)
        return _render_error(
            request,
            _USER_FRIENDLY_ERRORS[ResponseParseError],
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_level="error",
        )

    except ReportValidationError as exc:
        logger.error("Report validation error during /validate: %s", exc)
        return _render_error(
            request,
            _USER_FRIENDLY_ERRORS[ReportValidationError],
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_level="error",
        )

    except AnalyzerError as exc:
        # Catch-all for any other AnalyzerError subclasses.
        logger.exception("Unexpected AnalyzerError in /validate: %s", exc)
        return _render_error(
            request,
            "An unexpected error occurred during analysis. Please try again.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_level="error",
        )

    except Exception as exc:  # noqa: BLE001
        # Absolute safety net — never expose raw tracebacks to the UI.
        logger.exception("Unhandled exception in /validate: %s", exc)
        return _render_error(
            request,
            "An unexpected server error occurred. Please try again later.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            log_level="error",
        )

    # ------------------------------------------------------------------
    # 3. Render and return the report partial.
    # ------------------------------------------------------------------
    logger.info(
        "Viability analysis complete (idea_length=%d, score=%s, competitors=%d)",
        len(idea_request.idea),
        report.viability_score.value,
        len(report.competitors),
    )

    return templates.TemplateResponse(
        "report.html",
        {"request": request, "report": report},
        status_code=status.HTTP_200_OK,
    )


@app.get(
    "/health",
    summary="Health check",
    tags=["meta"],
    response_model=None,
)
async def health_check() -> JSONResponse:
    """Return a lightweight health-check payload.

    Used by load balancers, container orchestrators, and smoke tests to
    confirm that the server is running and responsive.

    Returns
    -------
    JSONResponse
        ``{"status": "ok", "version": "<version>"}`` with HTTP 200.
    """
    return JSONResponse(
        content={"status": "ok", "version": __version__},
        status_code=status.HTTP_200_OK,
    )


# ---------------------------------------------------------------------------
# Console script entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the Uvicorn development server.

    This function is invoked by the ``vibe-validator`` console script
    declared in ``pyproject.toml``.  Configuration is read from environment
    variables with sensible defaults.

    Environment variables
    ---------------------
    APP_HOST:
        Host to bind to (default: ``"0.0.0.0"``).
    APP_PORT:
        Port to listen on (default: ``8000``).
    APP_ENV:
        Set to ``"development"`` to enable auto-reload (default: ``"development"``).
    """
    import uvicorn  # noqa: PLC0415 — deferred import; uvicorn is optional at module level

    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    env: str = os.getenv("APP_ENV", "development").lower()
    reload: bool = env == "development"

    logger.info(
        "Starting Vibe Validator v%s on %s:%d (reload=%s)",
        __version__,
        host,
        port,
        reload,
    )

    uvicorn.run(
        "vibe_validator.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    run()
