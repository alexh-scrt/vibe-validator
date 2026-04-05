"""Pydantic models for vibe_validator.

This module defines the data contract for the entire application:

- :class:`IdeaRequest` — the incoming idea submission from the user.
- :class:`MarketSizeEstimate` — structured market-size data returned by the LLM.
- :class:`Competitor` — a single competing product surfaced by the analysis.
- :class:`MonetizationModel` — a suggested revenue model with rationale.
- :class:`ViabilityReport` — the complete structured report produced by the analyzer.

All models use Pydantic v2 semantics (``model_validator``, ``field_validator``,
``model_config``, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ViabilityScore(str, Enum):
    """Overall viability rating for the submitted idea.

    Values progress from lowest to highest confidence that the idea has
    real-world traction potential.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketSizeTier(str, Enum):
    """Broad market-size tier used when precise TAM figures are unavailable."""

    NICHE = "niche"  # < $100 M TAM
    SMALL = "small"  # $100 M – $1 B
    MEDIUM = "medium"  # $1 B – $10 B
    LARGE = "large"  # $10 B – $100 B
    MASSIVE = "massive"  # > $100 B


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class MarketSizeEstimate(BaseModel):
    """Estimated market-size data for the analysed idea.

    Attributes
    ----------
    tier:
        A broad categorical size tier when TAM figures are speculative.
    total_addressable_market:
        Human-readable TAM string (e.g. ``"$4.5 B globally by 2027"``).
    serviceable_addressable_market:
        Human-readable SAM string representing the realistic reachable slice.
    growth_rate:
        Indicative CAGR or year-on-year growth description (e.g. ``"~18 % CAGR"``).
    notes:
        Any additional context, caveats, or data sources the LLM provides.
    """

    tier: MarketSizeTier = Field(
        ...,
        description="Broad categorical market-size tier.",
        examples=["medium"],
    )
    total_addressable_market: str = Field(
        ...,
        alias="tam",
        description="Total addressable market as a human-readable string.",
        examples=["$4.5 billion globally by 2027"],
        min_length=1,
        max_length=300,
    )
    serviceable_addressable_market: str = Field(
        ...,
        alias="sam",
        description="Serviceable addressable market as a human-readable string.",
        examples=["$850 million for English-speaking SaaS founders"],
        min_length=1,
        max_length=300,
    )
    growth_rate: str = Field(
        ...,
        description="Indicative annual growth rate or CAGR description.",
        examples=["~18% CAGR through 2028"],
        min_length=1,
        max_length=200,
    )
    notes: str = Field(
        default="",
        description="Optional caveats or additional context from the analysis.",
        max_length=1000,
    )

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }


class Competitor(BaseModel):
    """A single existing product that competes with or resembles the idea.

    Attributes
    ----------
    name:
        Product or company name.
    description:
        Short description of what the competitor does and how it overlaps.
    url:
        Optional public URL for the competitor's website or product page.
    differentiator:
        How the user's idea could differentiate from this competitor.
    """

    name: str = Field(
        ...,
        description="Product or company name.",
        examples=["Notion"],
        min_length=1,
        max_length=200,
    )
    description: str = Field(
        ...,
        description="Short description of what this competitor does.",
        examples=["All-in-one workspace combining notes, wikis, and project management."],
        min_length=1,
        max_length=600,
    )
    url: str | None = Field(
        default=None,
        description="Optional public URL for the competitor.",
        examples=["https://notion.so"],
    )
    differentiator: str = Field(
        ...,
        description="How the submitted idea could stand out from this competitor.",
        examples=["Focus exclusively on solo founders with opinionated templates."],
        min_length=1,
        max_length=600,
    )

    model_config = {
        "str_strip_whitespace": True,
    }

    @field_validator("url", mode="before")
    @classmethod
    def normalise_url(cls, value: object) -> str | None:
        """Accept empty string as ``None`` and strip whitespace from URLs.

        Parameters
        ----------
        value:
            Raw URL value from the LLM response.

        Returns
        -------
        str | None
            Cleaned URL string or ``None`` if blank.
        """
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return str(value)


class MonetizationModel(BaseModel):
    """A suggested revenue / monetisation approach for the idea.

    Attributes
    ----------
    model_name:
        Short label for the monetisation model (e.g. ``"Freemium SaaS"``).
    description:
        Explanation of how this model would work for the specific idea.
    estimated_arpu:
        Rough average revenue per user per month (human-readable string).
    pros:
        List of advantages of this model for the idea.
    cons:
        List of trade-offs or risks.
    """

    model_name: str = Field(
        ...,
        description="Short label for the monetisation model.",
        examples=["Freemium SaaS"],
        min_length=1,
        max_length=150,
    )
    description: str = Field(
        ...,
        description="How this model would work for the specific idea.",
        examples=[
            "Offer a free tier with limited validations per month and charge "
            "$19/month for unlimited access."
        ],
        min_length=1,
        max_length=800,
    )
    estimated_arpu: str = Field(
        ...,
        description="Approximate average revenue per user per month.",
        examples=["$15 – $30 / user / month"],
        min_length=1,
        max_length=200,
    )
    pros: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list,
        description="Advantages of this monetisation model.",
        max_length=10,
    )
    cons: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list,
        description="Trade-offs or risks of this monetisation model.",
        max_length=10,
    )

    model_config = {
        "str_strip_whitespace": True,
    }


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class IdeaRequest(BaseModel):
    """Payload submitted by the user describing their app idea.

    Attributes
    ----------
    idea:
        Free-text description of the side-hustle app idea to be validated.
        Must be between 20 and 2 000 characters so the LLM has enough
        context without being flooded with noise.
    """

    idea: str = Field(
        ...,
        description=(
            "Free-text description of the app idea to validate. "
            "Be as specific as possible for a better analysis."
        ),
        examples=[
            "An AI tool that analyses a founder's idea and produces an instant "
            "viability report with market size, competitors, and a starter prompt "
            "for Lovable or Claude."
        ],
        min_length=20,
        max_length=2000,
    )

    model_config = {
        "str_strip_whitespace": True,
        "json_schema_extra": {
            "examples": [
                {
                    "idea": (
                        "A mobile app that lets dog owners book on-demand "
                        "veterinary video consultations within minutes."
                    )
                }
            ]
        },
    }

    @field_validator("idea", mode="after")
    @classmethod
    def idea_not_blank(cls, value: str) -> str:
        """Ensure the idea string contains non-whitespace content.

        Parameters
        ----------
        value:
            Already-stripped idea string (``str_strip_whitespace`` is active).

        Returns
        -------
        str
            The validated idea string.

        Raises
        ------
        ValueError
            If the stripped idea is empty.
        """
        if not value:
            raise ValueError("idea must contain non-whitespace characters.")
        return value


# ---------------------------------------------------------------------------
# Primary response model
# ---------------------------------------------------------------------------


class ViabilityReport(BaseModel):
    """Complete structured viability report for a submitted app idea.

    This is the canonical response model produced by the analyzer and rendered
    by the Jinja2 report template.

    Attributes
    ----------
    idea_summary:
        One- or two-sentence restatement / clarification of the idea as
        understood by the LLM.
    viability_score:
        Overall viability rating (low / medium / high).
    viability_rationale:
        Short paragraph explaining why this score was assigned.
    market_size:
        Structured market-size estimate.
    competitors:
        List of similar or competing products (1 – 5 items).
    monetization_models:
        Suggested revenue models (1 – 3 items).
    key_risks:
        Bullet-point list of the biggest risks or unknowns.
    key_opportunities:
        Bullet-point list of the strongest tailwinds or opportunities.
    starter_prompt:
        Ready-to-paste prompt the user can drop into Lovable, Claude, or
        Base44 to start building immediately.
    """

    idea_summary: str = Field(
        ...,
        description="One- or two-sentence restatement of the idea as understood by the LLM.",
        min_length=10,
        max_length=500,
    )
    viability_score: ViabilityScore = Field(
        ...,
        description="Overall viability rating for the idea.",
        examples=["medium"],
    )
    viability_rationale: str = Field(
        ...,
        description="Short paragraph explaining the viability score.",
        min_length=20,
        max_length=1000,
    )
    market_size: MarketSizeEstimate = Field(
        ...,
        description="Structured market-size estimate.",
    )
    competitors: list[Competitor] = Field(
        ...,
        description="Similar or competing products surfaced by the analysis.",
        min_length=1,
        max_length=5,
    )
    monetization_models: list[MonetizationModel] = Field(
        ...,
        description="Suggested revenue / monetisation approaches.",
        min_length=1,
        max_length=3,
    )
    key_risks: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        ...,
        description="Biggest risks or unknowns the founder should be aware of.",
        min_length=1,
        max_length=8,
    )
    key_opportunities: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        ...,
        description="Strongest tailwinds or opportunities the founder can leverage.",
        min_length=1,
        max_length=8,
    )
    starter_prompt: str = Field(
        ...,
        description=(
            "Ready-to-paste prompt for Lovable, Claude, or Base44 so the "
            "founder can start building immediately."
        ),
        min_length=50,
        max_length=3000,
    )

    model_config = {
        "str_strip_whitespace": True,
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def validate_list_diversity(self) -> "ViabilityReport":
        """Ensure competitor names are unique within the report.

        Returns
        -------
        ViabilityReport
            The validated model instance.

        Raises
        ------
        ValueError
            If duplicate competitor names are detected.
        """
        names = [c.name.lower() for c in self.competitors]
        if len(names) != len(set(names)):
            raise ValueError(
                "competitors list contains duplicate names — each competitor "
                "must be listed only once."
            )
        return self
