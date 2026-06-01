"""Unit tests for the news analyzer module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalyzer
from cop_fx.data.news_fetcher import Article


@pytest.fixture()
def sample_articles() -> list[Article]:
    now = datetime.now(tz=timezone.utc)
    return [
        Article(
            title="BanRep sube tasa de interés 50 pb",
            summary="El Banco de la República aumentó su tasa de referencia ante presiones inflacionarias.",
            url="https://example.com/1",
            published_at=now,
            source="Portafolio",
        ),
        Article(
            title="Exportaciones de café aumentan 12% en mayo",
            summary="Colombia exportó 1.2 millones de sacos en mayo, impulsado por precios internacionales.",
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
def test_analyze_batch_returns_analyzed_articles(sample_articles: list[Article]) -> None:
    llm_response_json = """[
        {"index": 0, "topic": "monetary_policy", "severity": "high", "bullish_cop": true, "reasoning": "Rate hike supports peso"},
        {"index": 1, "topic": "commodities", "severity": "medium", "bullish_cop": true, "reasoning": "Coffee export boost"},
        {"index": 2, "topic": "political_risk", "severity": "high", "bullish_cop": false, "reasoning": "Port blockade"}
    ]"""

    mock_response = MagicMock()
    mock_response.content = llm_response_json

    with patch("cop_fx.analysis.news_analyzer.get_settings") as mock_settings:
        settings = MagicMock()
        settings.llm_model = "claude-sonnet-4-6"
        settings.llm_temperature = 0.0
        settings.anthropic_api_key.get_secret_value.return_value = "test-key"
        mock_settings.return_value = settings

        with patch("cop_fx.analysis.news_analyzer.ChatAnthropic") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            analyzer = NewsAnalyzer()
            result = analyzer.analyze_batch(sample_articles)

    assert len(result) == 3
    assert result[0].topic == "monetary_policy"
    assert result[0].severity == "high"
    assert result[0].bullish_cop is True
    assert result[2].bullish_cop is False


@pytest.mark.unit()
def test_analyze_batch_empty_input() -> None:
    with patch("cop_fx.analysis.news_analyzer.get_settings"):
        with patch("cop_fx.analysis.news_analyzer.ChatAnthropic"):
            analyzer = NewsAnalyzer()
            result = analyzer.analyze_batch([])
    assert result == []


@pytest.mark.unit()
def test_analyze_batch_falls_back_on_invalid_json(sample_articles: list[Article]) -> None:
    mock_response = MagicMock()
    mock_response.content = "not valid json at all {{{"

    with patch("cop_fx.analysis.news_analyzer.get_settings") as mock_settings:
        settings = MagicMock()
        settings.llm_model = "claude-sonnet-4-6"
        settings.llm_temperature = 0.0
        settings.anthropic_api_key.get_secret_value.return_value = "test-key"
        mock_settings.return_value = settings

        with patch("cop_fx.analysis.news_analyzer.ChatAnthropic") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            analyzer = NewsAnalyzer()
            result = analyzer.analyze_batch(sample_articles)

    # Should return fallback results (one per article)
    assert len(result) == len(sample_articles)
