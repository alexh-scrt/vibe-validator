"""Smoke tests for Phase 1 — project scaffold and configuration.

These tests verify that the package is importable, exposes the expected
public symbols, and that the placeholder application responds correctly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_package_imports() -> None:
    """vibe_validator package must be importable without errors."""
    import vibe_validator  # noqa: PLC0415

    assert vibe_validator is not None


def test_version_string() -> None:
    """__version__ must be a non-empty string."""
    from vibe_validator import __version__  # noqa: PLC0415

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_app_is_fastapi_instance() -> None:
    """The exported ``app`` must be a FastAPI application."""
    from fastapi import FastAPI  # noqa: PLC0415

    from vibe_validator import app  # noqa: PLC0415

    assert isinstance(app, FastAPI)


def test_health_endpoint_returns_200() -> None:
    """GET /health must return HTTP 200 with status ok."""
    from vibe_validator import app  # noqa: PLC0415

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_payload() -> None:
    """GET /health payload must contain 'status' and 'version' keys."""
    from vibe_validator import app  # noqa: PLC0415

    client = TestClient(app)
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "version" in data
    assert isinstance(data["version"], str)
