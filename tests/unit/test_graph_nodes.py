"""Unit tests for LangGraph node functions."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cop_fx.agents.nodes import analyze_news, fetch_fx, fetch_news, generate_report
from cop_fx.agents.state import PipelineState
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalysis
from cop_fx.data.news_fetcher import Article


def _base_state(**overrides) -> PipelineState:  # type: ignore[return]
    state: PipelineState = {
        "run_date": "2024-06-01",
        "horizon_days": 3,
        "publish_enabled": False,
        "errors": [],
        "raw_articles": [],
        "analyzed_articles": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


@pytest.fixture()
def sample_fx_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    return pd.DataFrame({"ds": dates, "y": [4000 + i * 0.5 for i in range(100)]})


# ── fetch_fx ──────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_fetch_fx_populates_state(sample_fx_df: pd.DataFrame) -> None:
    with patch("cop_fx.agents.nodes.FXFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.return_value = sample_fx_df
        result = fetch_fx(_base_state())

    assert "fx_df" in result
    assert "latest_rate" in result
    assert result["latest_rate"] > 0
    assert "rate_change_pct" in result


@pytest.mark.unit()
def test_fetch_fx_captures_errors_on_failure() -> None:
    with patch("cop_fx.agents.nodes.FXFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.side_effect = RuntimeError("timeout")
        result = fetch_fx(_base_state())

    assert any("fetch_fx" in e for e in result.get("errors", []))
    assert "fx_df" not in result


# ── fetch_news ────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_fetch_news_populates_state() -> None:
    fake_articles = [
        Article(
            title="Dólar sube",
            summary="El dólar subió frente al peso.",
            url="https://example.com",
            published_at=datetime.now(tz=timezone.utc),
            source="Test",
        )
    ]
    with patch("cop_fx.agents.nodes.NewsFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.return_value = fake_articles
        result = fetch_news(_base_state())

    assert result["raw_articles"] == fake_articles


# ── analyze_news ──────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_analyze_news_empty_articles_returns_defaults() -> None:
    result = analyze_news(_base_state(raw_articles=[]))
    assert result["analyzed_articles"] == []
    assert "No news" in result.get("news_summary", "")


@pytest.mark.unit()
def test_analyze_news_uses_analyzer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    articles = [
        Article(
            title="BanRep mantiene tasas",
            summary="Junta directiva decide no mover tasas.",
            url="https://x.com",
            published_at=datetime.now(tz=timezone.utc),
            source="Portafolio",
        )
    ]

    verdict = AnalyzedArticle(
        article=articles[0],
        topic="monetary_policy",
        severity="medium",
        bullish_cop=False,
        reasoning="Rates on hold",
        keywords=["BanRep", "tasas"],
    )
    mock_analysis = NewsAnalysis(items=[verdict], narrative="Market is neutral.")

    with patch("cop_fx.agents.nodes.NewsAnalyzer") as MockAnalyzer:
        MockAnalyzer.return_value.analyze.return_value = mock_analysis
        result = analyze_news(_base_state(raw_articles=articles))

    analyzed = result.get("analyzed_articles", [])
    assert len(analyzed) == 1
    assert analyzed[0]["topic"] == "monetary_policy"
    assert analyzed[0]["keywords"] == ["BanRep", "tasas"]
    assert result.get("news_summary") == "Market is neutral."


# ── generate_report ───────────────────────────────────────────────────────

@pytest.mark.unit()
def test_generate_report_creates_markdown(sample_fx_df: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    ensemble = pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-02", periods=3, freq="B"),
            "yhat": [4100.0, 4105.0, 4110.0],
            "yhat_lower": [4080.0, 4085.0, 4090.0],
            "yhat_upper": [4120.0, 4125.0, 4130.0],
        }
    )

    state = _base_state(
        fx_df=sample_fx_df,
        latest_rate=4050.0,
        rate_change_pct=1.25,
        news_summary="Peso weakened due to global risk-off.",
        ensemble_df=ensemble,
        eval_metrics=[],
    )

    with patch("cop_fx.agents.nodes.get_settings") as mock_settings:
        settings = MagicMock()
        settings.forecast_horizon_days = 3
        settings.report_output_dir = str(tmp_path)
        mock_settings.return_value = settings

        result = generate_report(state)

    assert "report_markdown" in result
    assert "1 USD = 4,050.00 COP" in result["report_markdown"]
    assert "+1.25%" in result["report_markdown"]
    assert result.get("tweet_text", "").startswith("COP/USD")
    assert len(result.get("tweet_text", "")) <= 280
