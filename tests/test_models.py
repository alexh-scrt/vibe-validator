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
    return {
        "tier": "medium",
        "tam": "$5 billion globally by 2027",
        "sam": "$800 million for English-speaking markets",
        "growth_rate": "~15% CAGR through 2028",
        "notes": "Based on analyst estimates for the productivity SaaS sector.",
    }


def _valid_competitor() -> dict:
    return {
        "name": "Notion",
        "description": "All-in-one workspace for notes, wikis, and project management.",
        "url": "https://notion.so",
        "differentiator": "Focus exclusively on solo founders with opinionated templates.",
    }


def _valid_monetization_model() -> dict:
    return {
        "model_name": "Freemium SaaS",
        "description": "Free tier with 3 validations/month; $19/month for unlimited.",
        "estimated_arpu": "$15 – $30 / user / month",
        "pros": ["Low barrier to entry", "Viral growth potential"],
        "cons": ["High churn risk at free tier"],
    }


def _valid_report_dict() -> dict:
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
    def test_valid_values(self) -> None:
        assert ViabilityScore.LOW == "low"
        assert ViabilityScore.MEDIUM == "medium"
        assert ViabilityScore.HIGH == "high"

    def test_from_string(self) -> None:
        assert ViabilityScore("high") is ViabilityScore.HIGH

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ViabilityScore("extreme")


# ---------------------------------------------------------------------------
# MarketSizeTier enum tests
# ---------------------------------------------------------------------------


class TestMarketSizeTier:
    def test_all_tiers_present(self) -> None:
        tiers = {t.value for t in MarketSizeTier}
        assert tiers == {"niche", "small", "medium", "large", "massive"}


# ---------------------------------------------------------------------------
# MarketSizeEstimate tests
# ---------------------------------------------------------------------------


class TestMarketSizeEstimate:
    def test_valid_construction(self) -> None:
        data = _valid_market_size()
        m = MarketSizeEstimate(**data)
        assert m.tier == MarketSizeTier.MEDIUM
        assert "5 billion" in m.total_addressable_market
        assert "800 million" in m.serviceable_addressable_market
        assert "15%" in m.growth_rate

    def test_alias_population(self) -> None:
        """Alias fields 'tam' and 'sam' must populate the full field names."""
        data = _valid_market_size()
        m = MarketSizeEstimate.model_validate(data)
        assert m.total_addressable_market
        assert m.serviceable_addressable_market

    def test_notes_defaults_to_empty_string(self) -> None:
        data = _valid_market_size()
        data.pop("notes")
        m = MarketSizeEstimate(**data)
        assert m.notes == ""

    def test_missing_required_field_raises(self) -> None:
        data = _valid_market_size()
        data.pop("tam")
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_invalid_tier_raises(self) -> None:
        data = _valid_market_size()
        data["tier"] = "galactic"
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_tam_too_short_raises(self) -> None:
        data = _valid_market_size()
        data["tam"] = ""
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)

    def test_notes_max_length_enforced(self) -> None:
        data = _valid_market_size()
        data["notes"] = "x" * 1001
        with pytest.raises(ValidationError):
            MarketSizeEstimate(**data)


# ---------------------------------------------------------------------------
# Competitor tests
# ---------------------------------------------------------------------------


class TestCompetitor:
    def test_valid_construction(self) -> None:
        c = Competitor(**_valid_competitor())
        assert c.name == "Notion"
        assert c.url == "https://notion.so"

    def test_url_none_is_allowed(self) -> None:
        data = _valid_competitor()
        data["url"] = None
        c = Competitor(**data)
        assert c.url is None

    def test_empty_string_url_becomes_none(self) -> None:
        data = _valid_competitor()
        data["url"] = "   "
        c = Competitor(**data)
        assert c.url is None

    def test_url_missing_defaults_to_none(self) -> None:
        data = _valid_competitor()
        data.pop("url")
        c = Competitor(**data)
        assert c.url is None

    def test_name_required(self) -> None:
        data = _valid_competitor()
        data.pop("name")
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_empty_name_raises(self) -> None:
        data = _valid_competitor()
        data["name"] = ""
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_description_max_length_enforced(self) -> None:
        data = _valid_competitor()
        data["description"] = "d" * 601
        with pytest.raises(ValidationError):
            Competitor(**data)

    def test_differentiator_required(self) -> None:
        data = _valid_competitor()
        data.pop("differentiator")
        with pytest.raises(ValidationError):
            Competitor(**data)


# ---------------------------------------------------------------------------
# MonetizationModel tests
# ---------------------------------------------------------------------------


class TestMonetizationModel:
    def test_valid_construction(self) -> None:
        m = MonetizationModel(**_valid_monetization_model())
        assert m.model_name == "Freemium SaaS"
        assert len(m.pros) == 2
        assert len(m.cons) == 1

    def test_pros_and_cons_default_to_empty_list(self) -> None:
        data = _valid_monetization_model()
        data.pop("pros")
        data.pop("cons")
        m = MonetizationModel(**data)
        assert m.pros == []
        assert m.cons == []

    def test_model_name_required(self) -> None:
        data = _valid_monetization_model()
        data.pop("model_name")
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_estimated_arpu_required(self) -> None:
        data = _valid_monetization_model()
        data.pop("estimated_arpu")
        with pytest.raises(ValidationError):
            MonetizationModel(**data)

    def test_description_max_length_enforced(self) -> None:
        data = _valid_monetization_model()
        data["description"] = "d" * 801
        with pytest.raises(ValidationError):
            MonetizationModel(**data)


# ---------------------------------------------------------------------------
# IdeaRequest tests
# ---------------------------------------------------------------------------


class TestIdeaRequest:
    def test_valid_idea(self) -> None:
        req = IdeaRequest(idea="A dog-walking app with live GPS tracking for anxious owners.")
        assert req.idea.startswith("A dog")

    def test_idea_is_stripped(self) -> None:
        req = IdeaRequest(idea="  My brilliant app idea that is long enough to pass.  ")
        assert not req.idea.startswith(" ")
        assert not req.idea.endswith(" ")

    def test_idea_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            IdeaRequest(idea="Too short")

    def test_idea_at_min_boundary_passes(self) -> None:
        idea = "a" * 20
        req = IdeaRequest(idea=idea)
        assert len(req.idea) == 20

    def test_idea_too_long_raises(self) -> None:
        with pytest.raises(ValidationError):
            IdeaRequest(idea="x" * 2001)

    def test_idea_at_max_boundary_passes(self) -> None:
        idea = "a" * 2000
        req = IdeaRequest(idea=idea)
        assert len(req.idea) == 2000

    def test_whitespace_only_idea_raises(self) -> None:
        """After stripping, blank idea must fail the custom validator."""
        with pytest.raises(ValidationError):
            IdeaRequest(idea=" " * 30)

    def test_idea_missing_raises(self) -> None:
        with pytest.raises(ValidationError):
            IdeaRequest()

    def test_json_serialisation_round_trip(self) -> None:
        req = IdeaRequest(idea="An app that does something really cool and innovative.")
        json_str = req.model_dump_json()
        restored = IdeaRequest.model_validate_json(json_str)
        assert restored.idea == req.idea


# ---------------------------------------------------------------------------
# ViabilityReport tests
# ---------------------------------------------------------------------------


class TestViabilityReport:
    def test_valid_construction(self) -> None:
        report = ViabilityReport.model_validate(_valid_report_dict())
        assert report.viability_score == ViabilityScore.HIGH
        assert len(report.competitors) == 1
        assert len(report.monetization_models) == 1

    def test_all_required_fields_present(self) -> None:
        report = ViabilityReport.model_validate(_valid_report_dict())
        assert report.idea_summary
        assert report.viability_rationale
        assert report.market_size
        assert report.key_risks
        assert report.key_opportunities
        assert report.starter_prompt

    def test_missing_idea_summary_raises(self) -> None:
        data = _valid_report_dict()
        data.pop("idea_summary")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_viability_score_raises(self) -> None:
        data = _valid_report_dict()
        data.pop("viability_score")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_invalid_viability_score_raises(self) -> None:
        data = _valid_report_dict()
        data["viability_score"] = "unknown"
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_missing_competitors_raises(self) -> None:
        data = _valid_report_dict()
        data.pop("competitors")
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_competitors_list_raises(self) -> None:
        data = _valid_report_dict()
        data["competitors"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_too_many_competitors_raises(self) -> None:
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [
            {**competitor, "name": f"Competitor {i}"} for i in range(6)
        ]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_duplicate_competitor_names_raises(self) -> None:
        data = _valid_report_dict()
        competitor = _valid_competitor()
        data["competitors"] = [competitor, competitor]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_key_risks_raises(self) -> None:
        data = _valid_report_dict()
        data["key_risks"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_empty_key_opportunities_raises(self) -> None:
        data = _valid_report_dict()
        data["key_opportunities"] = []
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_starter_prompt_too_short_raises(self) -> None:
        data = _valid_report_dict()
        data["starter_prompt"] = "Too short."
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_multiple_monetization_models(self) -> None:
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
        data = _valid_report_dict()
        model = _valid_monetization_model()
        data["monetization_models"] = [
            {**model, "model_name": f"Model {i}"} for i in range(4)
        ]
        with pytest.raises(ValidationError):
            ViabilityReport.model_validate(data)

    def test_json_serialisation_round_trip(self) -> None:
        report = ViabilityReport.model_validate(_valid_report_dict())
        json_str = report.model_dump_json(by_alias=True)
        restored = ViabilityReport.model_validate_json(json_str)
        assert restored.viability_score == report.viability_score
        assert restored.idea_summary == report.idea_summary

    def test_dict_serialisation(self) -> None:
        report = ViabilityReport.model_validate(_valid_report_dict())
        data = report.model_dump()
        assert "viability_score" in data
        assert "market_size" in data
        assert isinstance(data["competitors"], list)
