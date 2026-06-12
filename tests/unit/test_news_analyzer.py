"""Unit tests for the news analyzer module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalyzer
from cop_fx.contracts import ArticleAnalysis, BatchAnalysis
from cop_fx.data.news_fetcher import Article


@pytest.fixture()
def sample_articles() -> list[Article]:
    now = datetime.now(tz=UTC)
    return [
        Article(
            title="BanRep sube tasa de interés 50 pb",
            summary=(
                "El Banco de la República aumentó su tasa de referencia "
                "ante presiones inflacionarias."
            ),
            url="https://example.com/1",
            published_at=now,
            source="Portafolio",
        ),
        Article(
            title="Exportaciones de café aumentan 12% en mayo",
            summary=(
                "Colombia exportó 1.2 millones de sacos en mayo, "
                "impulsado por precios internacionales."
            ),
            url="https://example.com/2",
            published_at=now,
            source="El Tiempo",
        ),
        Article(
            title="Protesta social paraliza puertos del Pacífico",
            summary="Bloqueos en Buenaventura afectan importaciones y exportaciones.",
            url="https://example.com/3",
            published_at=now,
            source="Semana",
        ),
    ]


def _analyzer_with(structured_llm: MagicMock) -> NewsAnalyzer:
    """Build a NewsAnalyzer whose structured LLM is mocked."""
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm
    with patch("cop_fx.analysis.news_analyzer.get_chat_model", return_value=base_llm):
        return NewsAnalyzer()


def _batch_for_three() -> BatchAnalysis:
    return BatchAnalysis(
        items=[
            ArticleAnalysis(
                index=0,
                topic="monetary_policy",
                keywords=["BanRep", "tasas", "inflación"],
                severity="high",
                bullish_cop=True,
                reasoning="Rate hike supports peso",
            ),
            ArticleAnalysis(
                index=1,
                topic="commodities",
                keywords=["café", "exportaciones"],
                severity="medium",
                bullish_cop=True,
                reasoning="Coffee export boost",
            ),
            ArticleAnalysis(
                index=2,
                topic="political_risk",
                keywords=["protesta", "puertos"],
                severity="high",
                bullish_cop=False,
                reasoning="Port blockade",
            ),
        ],
        market_narrative="El peso se fortalece pese al riesgo político en los puertos.",
    )


@pytest.mark.unit()
def test_fallback_classifies_all_articles(sample_articles: list[Article]) -> None:
    analyzer = NewsAnalyzer.__new__(NewsAnalyzer)  # skip __init__ to avoid LLM init
    result = analyzer._fallback_classify(sample_articles)
    assert len(result) == len(sample_articles)
    for item in result:
        assert isinstance(item, AnalyzedArticle)
        assert item.topic == "other"
        assert item.severity in ("high", "low")


@pytest.mark.unit()
def test_fallback_detects_high_severity_keywords(sample_articles: list[Article]) -> None:
    analyzer = NewsAnalyzer.__new__(NewsAnalyzer)
    result = analyzer._fallback_classify(sample_articles)
    # First article mentions "tasas" — should be high severity
    assert result[0].severity == "high"


@pytest.mark.unit()
def test_analyze_returns_structured_verdicts(sample_articles: list[Article]) -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _batch_for_three()

    analyzer = _analyzer_with(structured_llm)
    analysis = analyzer.analyze(sample_articles)

    assert len(analysis.items) == 3
    assert analysis.items[0].topic == "monetary_policy"
    assert analysis.items[0].severity == "high"
    assert analysis.items[0].bullish_cop is True
    assert analysis.items[0].keywords == ["BanRep", "tasas", "inflación"]
    assert analysis.items[2].bullish_cop is False
    assert "peso" in analysis.narrative


@pytest.mark.unit()
def test_analyze_batch_keeps_legacy_signature(sample_articles: list[Article]) -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _batch_for_three()

    analyzer = _analyzer_with(structured_llm)
    result = analyzer.analyze_batch(sample_articles)

    assert isinstance(result, list)
    assert all(isinstance(item, AnalyzedArticle) for item in result)


@pytest.mark.unit()
def test_analyze_empty_input() -> None:
    analyzer = _analyzer_with(MagicMock())
    analysis = analyzer.analyze([])
    assert analysis.items == []
    assert "No news" in analysis.narrative


@pytest.mark.unit()
def test_analyze_falls_back_on_llm_error(sample_articles: list[Article]) -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("api down")

    analyzer = _analyzer_with(structured_llm)
    analysis = analyzer.analyze(sample_articles)

    # Should return fallback results (one per article)
    assert len(analysis.items) == len(sample_articles)
    assert all(item.reasoning.startswith("fallback") for item in analysis.items)


@pytest.mark.unit()
def test_analyze_fills_missing_indices_with_fallback(sample_articles: list[Article]) -> None:
    incomplete = BatchAnalysis(
        items=_batch_for_three().items[:2],  # LLM omitted article 2
        market_narrative="Narrativa parcial.",
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = incomplete

    analyzer = _analyzer_with(structured_llm)
    analysis = analyzer.analyze(sample_articles)

    assert len(analysis.items) == 3
    assert analysis.items[2].reasoning.startswith("fallback")
