"""vibe_validator — AI-powered side-hustle viability analysis tool.

This package exposes the FastAPI application instance so it can be imported
and run directly::

    uvicorn vibe_validator:app --reload

The ``app`` name is populated lazily by ``vibe_validator.main`` to avoid
circular imports during the build-up of subsequent phases.  In Phase 1 we
provide a minimal, self-contained placeholder that returns a health-check
response so the server can start immediately.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["app", "__version__"]

__version__: str = "0.1.0"

logger = logging.getLogger(__name__)


def _build_placeholder_app() -> "FastAPI":
    """Build and return a minimal FastAPI placeholder application.

    This placeholder is used during Phase 1 before the full route definitions
    are wired up in Phase 5.  It exposes a single ``/health`` endpoint so
    operators can confirm the server is running.

    Returns
    -------
    FastAPI
        A configured FastAPI application instance.
    """
    try:
        from fastapi import FastAPI  # noqa: PLC0415
        from fastapi.responses import JSONResponse  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fastapi is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    _app = FastAPI(
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

    @_app.get("/health", tags=["meta"], summary="Health check")
    async def health_check() -> JSONResponse:
        """Return a simple health-check payload.

        Returns
        -------
        JSONResponse
            JSON body ``{"status": "ok", "version": "<version>"}``.
        """
        return JSONResponse(content={"status": "ok", "version": __version__})

    return _app


def _load_app() -> "FastAPI":
    """Attempt to load the full application from ``vibe_validator.main``.

    If ``vibe_validator.main`` is not yet available (e.g. during early
    development phases) the placeholder application is returned instead.

    Returns
    -------
    FastAPI
        Either the fully-configured application or the placeholder.
    """
    try:
        main_module = importlib.import_module("vibe_validator.main")
        return main_module.app  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        logger.debug(
            "vibe_validator.main not available yet — using placeholder app."
        )
        return _build_placeholder_app()


# Public ``app`` symbol consumed by Uvicorn and other importers.
app: "FastAPI" = _load_app()
