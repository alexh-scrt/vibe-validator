"""Unit tests for vibe_validator.models.

Covers Pydantic model validation, field constraints, serialisation, and
the custom validators defined in each model class.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibe_validator.models import (
    Competitor,
    IdeaRequest,
    MarketSizeEstimate,
    MarketSizeTier,
    MonetizationModel,
    ViabilityReport,
    ViabilityScore,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _valid_market_size() -> dict:
    """Return a valid MarketSizeEstimate payload."""
    return {
        "tier": "medium",
        "tam": "$5 billion globally by 2027",
        "sam": "$800 million for English-speaking markets",
        "growth_rate": "~15% CAGR through 2028",
        "notes": "Based on analyst estimates for the productivity SaaS sector.",
    }


def _valid_competitor() -> dict:
    """Return a valid Competitor payload."""
    return {
        "name": "Notion",
        "description": "All-in-one workspace for notes, wikis, and project management.",
        "url": "https://notion.so",
        "differentiator": "Focus exclusively on solo founders with opinionated templates.",
    }


def _valid_monetization_model() -> dict:
    """Return a valid MonetizationModel payload."""
    return {
        "model_name": "Freemium SaaS",
        "description": "Free tier with 3 validations/month; $19/month for unlimited.",
        "estimated_arpu": "$15 \u2013 $30 / user / month",
        "pros": ["Low barrier to entry", "Viral growth potential"],
        "cons": ["High churn risk at free tier"],
    }


def _valid_report_dict() -> dict:
    """Return a fully valid ViabilityReport payload."""
    return {
        "idea_summary": "An AI tool that validates side-hustle app ideas for non-technical founders.",
        "viability_score": "high",
        "viability_rationale": (
            "The market for founder tooling is large and growing rapidly, and this "
            "idea fills a clear gap for non-technical builders who need instant signal."
        ),
        "market_size": _valid_market_size(),
        "competitors": [_valid_competitor()],
        "monetization_models": [_valid_monetization_model()],
        "key_risks": ["Dependence on OpenAI API uptime and pricing."],
        "key_opportunities": ["Massive wave of non-technical founders shipping with AI tools."],
        "starter_prompt": (
            "Build a FastAPI web app called Vibe Validator. It should accept a text "
            "description of a side-hustle app idea and return a structured viability "
            "report including market size, competitors, monetisation models, and risks. "
            "Use HTMX for the frontend, TailwindCSS for styling, and OpenAI GPT-4o for "
            "the analysis. Store no user data."
        ),
    }


# ---------------------------------------------------------------------------
# ViabilityScore enum tests
# ---------------------------------------------------------------------------


class TestViabilityScore:
    """Tests for the ViabilityScore enumeration."""

    def test_valid_values(self) -> None:
        """ViabilityScore members must have the correct string values."""
        assert ViabilityScore.LOW == "low"
        assert ViabilityScore.MEDIUM == "medium"
        assert ViabilityScore.HIGH == "high"

    def test_from_string_high(self) -> None:
        """ViabilityScore can be constructed from a valid string."""
        assert ViabilityScore("high") is ViabilityScore.HIGH

    def test_from_string_low(self) -> None:
        """ViabilityScore can be constructed from 'low'."""
        assert ViabilityScore("low") is ViabilityScore.LOW

    def test_from_string_medium(self) -> None:
        """ViabilityScore can be constructed from 'medium'."""
        assert ViabilityScore("medium") is ViabilityScore.MEDIUM

    def test_invalid_value_raises(self) -> None:
        """An unknown viability score string must raise ValueError."""
        with pytest.raises(ValueError):
            ViabilityScore("extreme")

    def test_case_sensitive(self) -> None:
        """ViabilityScore must be case-sensitive — 'HIGH' is invalid."""
        with pytest.raises(ValueError):
            ViabilityScore("HIGH")


# ---------------------------------------------------------------------------
# MarketSizeTier enum tests
# ---------------------------------------------------------------------------


class TestMarketSizeTier:
    """Tests for the MarketSizeTier enumeration."""

    def test_all_tiers_present(self) -> None:
        """All five market size tiers must be defined."""
        tiers = {t.value for t in MarketSizeTier}
        assert tiers == {"niche", "small", "medium", "large", "massive"}

    def test_niche_value(self) -> None:
        assert MarketSizeTier.NICHE == "niche"

    def test_massive_value(self) -> None:
        assert MarketSizeTier.MASSIVE == "massive"

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(ValueError):
            MarketSizeTier("galactic")


# ---------------------------------------------------------------------------
# MarketSizeEstimate tests
# ---------------------------------------------------------------------------


class TestMarketSizeEstimate:
    """Tests for the MarketSizeEstimate model."""

    def test_valid_construction(self) -> None:
        """A fully valid payload must produce a correct MarketSizeEstimate."""
        data = _valid_market_size()
        m = MarketSizeEstimate(**data)
        assert m.tier == MarketSizeTier.MEDIUM
        assert "5 billion" in m.total_addressable_market
        assert "800 million" in m.serviceable_addressable_market
        assert "15%" in m.growth_rate

    def test_alias_population_tam(self) -> None:
        """The 'tam' alias must populate total_addressable_market."""
        data = _valid_market_size()
        m = MarketSizeEstimate.model_validate(data)
        assert m.total_addressable_market == data["tam"]

    def test_alias_population_sam(self) -> None:
        """The 'sam' alias must populate serviceable_addressable_market."""
        data = _valid_market_size()
        m = MarketSizeEstimate.model_validate(data)
        assert m.serviceable_addressable_market == data["sam"]

    def test_notes_defaults_to_empty_string(self) -> None:
        """Omitting 'notes' must result in an empty string default."""
        data = _valid_market_size()
        data.pop("notes")
        m = MarketSizeEstimate(**data)
        assert m.notes == ""

    def test_notes_empty_string_accepted(self) -> None:
        """An explicit empty string for 'notes' must be accepted."""
        data = _valid_market_size()
        data["notes"] = ""
        m = MarketSizeEstimate(**data)
        assert m.notes == ""

    def test_missing_tam_raises(self) -> None:
        """Omitting 'tam' must raise a ValidationError."""
        data = _valid_market_size()
        data.pop("tam")
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_missing_sam_raises(self) -> None:
        """Omitting 'sam' must raise a ValidationError."""
        data = _valid_market_size()
        data.pop("sam")
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_missing_growth_rate_raises(self) -> None:
        """Omitting 'growth_rate' must raise a ValidationError."""
        data = _valid_market_size()
        data.pop("growth_rate")
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_invalid_tier_raises(self) -> None:
        """An unknown tier value must raise a ValidationError."""
        data = _valid_market_size()
        data["tier"] = "galactic"
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_tam_too_short_raises(self) -> None:
        """An empty 'tam' string must raise a ValidationError."""
        data = _valid_market_size()
        data["tam"] = ""
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_notes_max_length_enforced(self) -> None:
        """A 'notes' string exceeding 1000 chars must raise ValidationError."""
        data = _valid_market_size()
        data["notes"] = "x" * 1001
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_tam_max_length_enforced(self) -> None:
        """A 'tam' string exceeding 300 chars must raise ValidationError."""
        data = _valid_market_size()
        data["tam"] = "x" * 301
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_model_dump_uses_aliases(self) -> None:
        """model_dump(by_alias=True) must use 'tam' and 'sam' keys."""
        data = _valid_market_size()
        m = MarketSizeEstimate.model_validate(data)
        dumped = m.model_dump(by_alias=True)
        assert "tam" in dumped
        assert "sam" in dumped

    def test_model_dump_without_aliases(self) -> None:
        """model_dump() without aliases must use full field names."""
        data = _valid_market_size()
        m = MarketSizeEstimate.model_validate(data)
        dumped = m.model_dump()
        assert "total_addressable_market" in dumped
        assert "serviceable_addressable_market" in dumped


# ---------------------------------------------------------------------------
# Competitor tests
# ---------------------------------------------------------------------------


class TestCompetitor:
    """Tests for the Competitor model."""

    def test_valid_construction(self) -> None:
        """A fully valid payload must produce a correct Competitor."""
        c = Competitor(**_valid_competitor())
        assert c.name == "Notion"
        assert c.url == "https://notion.so"

    def test_url_none_is_allowed(self) -> None:
        """Explicit None url must be accepted."""
        data = _valid_competitor()
        data["url"] = None
        c = Competitor(**data)
        assert c.url is None

    def test_empty_string_url_becomes_none(self) -> None:
        """A whitespace-only url must be normalised to None."""
        data = _valid_competitor()
        data["url"] = "   "
        c = Competitor(**data)
        assert c.url is None

    def test_empty_string_url_strict_empty(self) -> None:
        """An empty string url must be normalised to None."""
        data = _valid_competitor()
        data["url"] = ""
        c = Competitor(**data)
        assert c.url is None

    def test_url_missing_defaults_to_none(self) -> None:
        """Omitting the url field must default to None."""
        data = _valid_competitor()
        data.pop("url")
        c = Competitor(**data)
        assert c.url is None

    def test_name_required(self) -> None:
        """Omitting 'name' must raise ValidationError."""
        data = _valid_competitor()
        data.pop("name")
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_empty_name_raises(self) -> None:
        """An empty name string must raise ValidationError."""
        data = _valid_competitor()
        data["name"] = ""
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_description_max_length_enforced(self) -> None:
        """A description exceeding 600 chars must raise ValidationError."""
        data = _valid_competitor()
        data["description"] = "d" * 601
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_differentiator_required(self) -> None:
        """Omitting 'differentiator' must raise ValidationError."""
        data = _valid_competitor()
        data.pop("differentiator")
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_differentiator_empty_raises(self) -> None:
        """An empty differentiator must raise ValidationError."""
        data = _valid_competitor()
        data["differentiator"] = ""
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_name_stripped(self) -> None:
        """Leading/trailing whitespace in name must be stripped."""
        data = _valid_competitor()
        data["name"] = "  Notion  "
        c = Competitor(**data)
        assert c.name == "Notion"

    def test_description_stripped(self) -> None:
        """Leading/trailing whitespace in description must be stripped."""
        data = _valid_competitor()
        data["description"] = "  Some description.  "
        c = Competitor(**data)
        assert c.description == "Some description."

    def test_url_with_valid_string_preserved(self) -> None:
        """A non-empty URL string must be preserved as-is."""
        data = _valid_competitor()
        data["url"] = "https://example.com"
        c = Competitor(**data)
        assert c.url == "https://example.com"


# ---------------------------------------------------------------------------
# MonetizationModel tests
# ---------------------------------------------------------------------------


class TestMonetizationModel:
    """Tests for the MonetizationModel model."""

    def test_valid_construction(self) -> None:
        """A fully valid payload must produce a correct MonetizationModel."""
        m = MonetizationModel(**_valid_monetization_model())
        assert m.model_name == "Freemium SaaS"
        assert len(m.pros) == 2
        assert len(m.cons) == 1

    def test_pros_and_cons_default_to_empty_list(self) -> None:
        """Omitting 'pros' and 'cons' must result in empty lists."""
        data = _valid_monetization_model()
        data.pop("pros")
        data.pop("cons")
        m = MonetizationModel(**data)
        assert m.pros == []
        assert m.cons == []

    def test_pros_only(self) -> None:
        """A model with only pros (no cons) must be accepted."""
        data = _valid_monetization_model()
        data.pop("cons")
        m = MonetizationModel(**data)
        assert m.cons == []
        assert len(m.pros) == 2

    def test_model_name_required(self) -> None:
        """Omitting 'model_name' must raise ValidationError."""
        data = _valid_monetization_model()
        data.pop("model_name")
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_estimated_arpu_required(self) -> None:
        """Omitting 'estimated_arpu' must raise ValidationError."""
        data = _valid_monetization_model()
        data.pop("estimated_arpu")
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_description_required(self) -> None:
        """Omitting 'description' must raise ValidationError."""
        data = _valid_monetization_model()
        data.pop("description")
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_description_max_length_enforced(self) -> None:
        """A description exceeding 800 chars must raise ValidationError."""
        data = _valid_monetization_model()
        data["description"] = "d" * 801
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_model_name_max_length_enforced(self) -> None:
        """A model_name exceeding 150 chars must raise ValidationError."""
        data = _valid_monetization_model()
        data["model_name"] = "m" * 151
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_model_name_stripped(self) -> None:
        """Whitespace in model_name must be stripped."""
        data = _valid_monetization_model()
        data["model_name"] = "  Freemium SaaS  "
        m = MonetizationModel(**data)
        assert m.model_name == "Freemium SaaS"

    def test_empty_model_name_raises(self) -> None:
        """An empty model_name must raise ValidationError."""
        data = _valid_monetization_model()
        data["model_name"] = ""
        with pytest.raises(ValidationError):
            MonetizationModel(**data)


# ---------------------------------------------------------------------------
# IdeaRequest tests
# ---------------------------------------------------------------------------


class TestIdeaRequest:
    """Tests for the IdeaRequest model."""

    def test_valid_idea(self) -> None:
        """A valid idea string must be accepted."""
        req = IdeaRequest(idea="A dog-walking app with live GPS tracking for anxious owners.")
        assert req.idea.startswith("A dog")

    def test_idea_is_stripped(self) -> None:
        """Leading/trailing whitespace must be stripped from the idea."""
        req = IdeaRequest(idea="  My brilliant app idea that is long enough to pass.  ")
        assert not req.idea.startswith(" ")
        assert not req.idea.endswith(" ")

    def test_idea_too_short_raises(self) -> None:
        """An idea shorter than 20 characters must raise ValidationError."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea="Too short")

    def test_idea_at_min_boundary_passes(self) -> None:
        """Exactly 20 characters must be accepted."""
        idea = "a" * 20
        req = IdeaRequest(idea=idea)
        assert len(req.idea) == 20

    def test_idea_just_below_min_raises(self) -> None:
        """19 characters must be rejected."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea="a" * 19)

    def test_idea_too_long_raises(self) -> None:
        """An idea exceeding 2000 characters must raise ValidationError."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea="x" * 2001)

    def test_idea_at_max_boundary_passes(self) -> None:
        """Exactly 2000 characters must be accepted."""
        idea = "a" * 2000
        req = IdeaRequest(idea=idea)
        assert len(req.idea) == 2000

    def test_idea_just_above_max_raises(self) -> None:
        """2001 characters must be rejected."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea="a" * 2001)

    def test_whitespace_only_idea_raises(self) -> None:
        """After stripping, blank idea must fail the custom validator."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea=" " * 30)

    def test_idea_missing_raises(self) -> None:
        """Omitting the idea field entirely must raise ValidationError."""
        with pytest.raises(ValidationError):
            IdeaRequest()  # type: ignore[call-arg]

    def test_json_serialisation_round_trip(self) -> None:
        """Serialising and deserialising an IdeaRequest must preserve the idea."""
        req = IdeaRequest(idea="An app that does something really cool and innovative.")
        json_str = req.model_dump_json()
        restored = IdeaRequest.model_validate_json(json_str)
        assert restored.idea == req.idea

    def test_model_dump_contains_idea(self) -> None:
        """model_dump() must contain the 'idea' key."""
        req = IdeaRequest(idea="An app that does something really cool and innovative.")
        data = req.model_dump()
        assert "idea" in data
        assert data["idea"] == req.idea

    def test_idea_with_special_characters(self) -> None:
        """Ideas with special characters must be accepted."""
        idea = "An app for \u00e9clair lovers who want to find the best pâtisseries nearby!"
        req = IdeaRequest(idea=idea)
        assert req.idea == idea

    def test_idea_with_newlines(self) -> None:
        """Ideas with embedded newlines must be accepted as long as they're long enough."""
        idea = "Line one of my idea.\nLine two adds more detail to meet the minimum length."
        req = IdeaRequest(idea=idea)
        assert "\n" in req.idea


# ---------------------------------------------------------------------------
# ViabilityReport tests
# ---------------------------------------------------------------------------


class TestViabilityReport:
    """Tests for the ViabilityReport model."""

    def test_valid_construction(self) -> None:
        """A fully valid payload must produce a correct ViabilityReport."""
        report = ViabilityReport.model_validate(_valid_report_dict())
        assert report.viability_score == ViabilityScore.HIGH
        assert len(report.competitors) == 1
        assert len(report.monetization_models) == 1

    def test_all_required_fields_present(self) -> None:
        """All expected fields must be non-empty on a valid report."""
        report = ViabilityReport.model_validate(_valid_report_dict())
        assert report.idea_summary
        assert report.viability_rationale
        assert report.market_size
        assert report.key_risks
        assert report.key_opportunities
        assert report.starter_prompt

    def test_missing_idea_summary_raises(self) -> None:
        """Omitting 'idea_summary' must raise ValidationError."""
        data = _valid_report_dict()
        data.pop("idea_summary")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_viability_score_raises(self) -> None:
        """Omitting 'viability_score' must raise ValidationError."""
        data = _valid_report_dict()
        data.pop("viability_score")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_invalid_viability_score_raises(self) -> None:
        """An unknown viability_score value must raise ValidationError."""
        data = _valid_report_dict()
        data["viability_score"] = "unknown"
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_viability_rationale_raises(self) -> None:
        """Omitting 'viability_rationale' must raise ValidationError."""
        data = _valid_report_dict()
        data.pop("viability_rationale")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_market_size_raises(self) -> None:
        """Omitting 'market_size' must raise ValidationError."""
        data = _valid_report_dict()
        data.pop("market_size")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_competitors_raises(self) -> None:
        """Omitting 'competitors' must raise ValidationError."""
        data = _valid_report_dict()
        data.pop("competitors")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_competitors_list_raises(self) -> None:
        """An empty competitors list must raise ValidationError."""
        data = _valid_report_dict()
        data["competitors"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_too_many_competitors_raises(self) -> None:
        """More than 5 competitors must raise ValidationError."""
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [
            {**competitor, "name": f"Competitor {i}"} for i in range(6)
        ]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_five_competitors_accepted(self) -> None:
        """Exactly 5 competitors must be accepted."""
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [
            {**competitor, "name": f"Competitor {i}"} for i in range(5)
        ]
        report = ViabilityReport.model_validate(data)
        assert len(report.competitors) == 5

    def test_duplicate_competitor_names_raises(self) -> None:
        """Duplicate competitor names must fail the model_validator."""
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [competitor, competitor.copy()]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_duplicate_competitor_names_case_insensitive(self) -> None:
        """Duplicate names differing only in case must also be rejected."""
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [
            competitor,
            {**competitor, "name": "NOTION"},
        ]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_key_risks_raises(self) -> None:
        """An empty key_risks list must raise ValidationError."""
        data = _valid_report_dict()
        data["key_risks"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_key_opportunities_raises(self) -> None:
        """An empty key_opportunities list must raise ValidationError."""
        data = _valid_report_dict()
        data["key_opportunities"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_starter_prompt_too_short_raises(self) -> None:
        """A starter_prompt below 50 chars must raise ValidationError."""
        data = _valid_report_dict()
        data["starter_prompt"] = "Too short."
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_starter_prompt_at_min_boundary(self) -> None:
        """A starter_prompt of exactly 50 chars must be accepted."""
        data = _valid_report_dict()
        data["starter_prompt"] = "x" * 50
        report = ViabilityReport.model_validate(data)
        assert len(report.starter_prompt) == 50

    def test_starter_prompt_too_long_raises(self) -> None:
        """A starter_prompt exceeding 3000 chars must raise ValidationError."""
        data = _valid_report_dict()
        data["starter_prompt"] = "x" * 3001
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_multiple_monetization_models(self) -> None:
        """Two monetization models must be accepted."""
        data = _valid_report_dict()
        data["monetization_models"] = [
            _valid_monetization_model(),
            {
                "model_name": "One-time purchase",
                "description": "Charge $49 once for lifetime access to the tool.",
                "estimated_arpu": "$49 one-time",
                "pros": ["Simple pricing"],
                "cons": ["No recurring revenue"],
            },
        ]
        report = ViabilityReport.model_validate(data)
        assert len(report.monetization_models) == 2

    def test_too_many_monetization_models_raises(self) -> None:
        """More than 3 monetization models must raise ValidationError."""
        data = _valid_report_dict()
        model = _valid_monetization_model()
        data["monetization_models"] = [
            {**model, "model_name": f"Model {i}"} for i in range(4)
        ]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_monetization_models_raises(self) -> None:
        """An empty monetization_models list must raise ValidationError."""
        data = _valid_report_dict()
        data["monetization_models"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_json_serialisation_round_trip(self) -> None:
        """Serialising and deserialising a ViabilityReport must be lossless."""
        report = ViabilityReport.model_validate(_valid_report_dict())
        json_str = report.model_dump_json(by_alias=True)
        restored = ViabilityReport.model_validate_json(json_str)
        assert restored.viability_score == report.viability_score
        assert restored.idea_summary == report.idea_summary
        assert len(restored.competitors) == len(report.competitors)

    def test_dict_serialisation(self) -> None:
        """model_dump() must produce a dict with all expected top-level keys."""
        report = ViabilityReport.model_validate(_valid_report_dict())
        data = report.model_dump()
        assert "viability_score" in data
        assert "market_size" in data
        assert isinstance(data["competitors"], list)
        assert isinstance(data["monetization_models"], list)
        assert isinstance(data["key_risks"], list)
        assert isinstance(data["key_opportunities"], list)

    def test_low_viability_score_accepted(self) -> None:
        """A 'low' viability_score must be accepted and parsed correctly."""
        data = _valid_report_dict()
        data["viability_score"] = "low"
        report = ViabilityReport.model_validate(data)
        assert report.viability_score == ViabilityScore.LOW

    def test_medium_viability_score_accepted(self) -> None:
        """A 'medium' viability_score must be accepted and parsed correctly."""
        data = _valid_report_dict()
        data["viability_score"] = "medium"
        report = ViabilityReport.model_validate(data)
        assert report.viability_score == ViabilityScore.MEDIUM

    def test_idea_summary_max_length_enforced(self) -> None:
        """An idea_summary exceeding 500 chars must raise ValidationError."""
        data = _valid_report_dict()
        data["idea_summary"] = "x" * 501
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_market_size_nested_validation(self) -> None:
        """An invalid market_size tier must cause the report to fail validation."""
        data = _valid_report_dict()
        data["market_size"] = {**_valid_market_size(), "tier": "invalid"}
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_multiple_risks_and_opportunities(self) -> None:
        """Multiple risks and opportunities must all be stored correctly."""
        data = _valid_report_dict()
        data["key_risks"] = ["Risk one", "Risk two", "Risk three"]
        data["key_opportunities"] = ["Opportunity A", "Opportunity B"]
        report = ViabilityReport.model_validate(data)
        assert len(report.key_risks) == 3
        assert len(report.key_opportunities) == 2

    def test_too_many_key_risks_raises(self) -> None:
        """More than 8 key risks must raise ValidationError."""
        data = _valid_report_dict()
        data["key_risks"] = [f"Risk {i}" for i in range(9)]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_too_many_key_opportunities_raises(self) -> None:
        """More than 8 key opportunities must raise ValidationError."""
        data = _valid_report_dict()
        data["key_opportunities"] = [f"Opportunity {i}" for i in range(9)]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)
