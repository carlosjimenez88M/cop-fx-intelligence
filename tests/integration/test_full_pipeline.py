"""Integration test: run the full pipeline with mocked external calls."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cop_fx.agents.graph import run_pipeline
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalysis
from cop_fx.contracts import HeadlineTag, MaterialityGate
from cop_fx.data.news_fetcher import Article


def _material_gate_llm(material: bool = True) -> MagicMock:
    """Mock de get_chat_model para el nodo check_materiality."""
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = MaterialityGate(
        has_material_news=material,
        reason="mocked gate",
        tags=[
            HeadlineTag(index=0, topic="trade", material=True),
            HeadlineTag(index=1, topic="energy_commodities", material=True),
        ],
    )
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm
    return base_llm


@pytest.fixture()
def synthetic_fx_df() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=250, freq="B")
    values = 4000.0 + rng.normal(0, 30, 250).cumsum()
    return pd.DataFrame({"ds": dates, "y": values})


@pytest.mark.integration()
def test_full_pipeline_runs_end_to_end(synthetic_fx_df: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    fake_articles = [
        Article(
            title="Colombia registra déficit comercial",
            summary="El déficit comercial se amplió en abril según el DANE.",
            url="https://example.com/news/1",
            published_at=datetime.now(tz=UTC),
            source="El Tiempo",
        ),
        Article(
            title="Petróleo cae 3% por temores de recesión",
            summary="Los precios del petróleo cayeron presionados por datos económicos de EEUU.",
            url="https://example.com/news/2",
            published_at=datetime.now(tz=UTC),
            source="Portafolio",
        ),
    ]

    mock_analysis = NewsAnalysis(
        items=[
            AnalyzedArticle(
                article=fake_articles[0],
                topic="trade",
                severity="medium",
                bullish_cop=False,
                reasoning="deficit",
                keywords=["déficit", "comercio"],
                entities=["DANE"],
                fx_relevance="direct",
                fx_channel="terms_of_trade",
            ),
            AnalyzedArticle(
                article=fake_articles[1],
                topic="energy_commodities",
                severity="high",
                bullish_cop=False,
                reasoning="oil drop",
                keywords=["petróleo", "recesión"],
                entities=["Brent"],
                fx_relevance="direct",
                fx_channel="terms_of_trade",
            ),
        ],
        narrative="Peso under pressure from weak commodities and trade deficit.",
    )

    with (
        patch("cop_fx.agents.nodes.FXFetcher") as MockFX,
        patch("cop_fx.agents.nodes.NewsFetcher") as MockNews,
        patch("cop_fx.agents.nodes.NewsAnalyzer") as MockAnalyzer,
        patch("cop_fx.agents.nodes.get_chat_model", return_value=_material_gate_llm()),
        patch("cop_fx.agents.nodes.get_settings") as MockSettings,
    ):
        MockAnalyzer.return_value.analyze.return_value = mock_analysis
        settings = MagicMock()
        settings.forecast_horizon_days = 5
        settings.report_output_dir = str(tmp_path)
        settings.twitter_enabled = False
        MockSettings.return_value = settings
        MockFX.return_value.fetch.return_value = synthetic_fx_df
        MockNews.return_value.fetch.return_value = fake_articles

        final_state = run_pipeline(run_date="2024-06-01", publish_enabled=False)

    # Validate output keys
    assert "report_markdown" in final_state
    assert "ensemble_df" in final_state
    assert "tweet_text" in final_state

    # No catastrophic errors
    errors = final_state.get("errors", [])
    assert errors == [], f"Pipeline errors: {errors}"

    # Report written to disk
    import os

    report_path = final_state.get("report_path", "")
    assert os.path.isfile(report_path)


@pytest.mark.integration()
def test_pipeline_handles_news_fetch_failure(synthetic_fx_df: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with (
        patch("cop_fx.agents.nodes.FXFetcher") as MockFX,
        patch("cop_fx.agents.nodes.NewsFetcher") as MockNews,
        patch("cop_fx.agents.nodes.NewsAnalyzer") as MockAnalyzer,
        patch("cop_fx.agents.nodes.get_chat_model", return_value=_material_gate_llm()),
        patch("cop_fx.agents.nodes.get_settings") as MockSettings,
    ):
        MockAnalyzer.return_value.analyze.return_value = NewsAnalysis(
            items=[], narrative="No news to analyze."
        )
        settings = MagicMock()
        settings.forecast_horizon_days = 3
        settings.report_output_dir = str(tmp_path)
        settings.twitter_enabled = False
        MockSettings.return_value = settings
        MockFX.return_value.fetch.return_value = synthetic_fx_df
        MockNews.return_value.fetch.side_effect = ConnectionError("RSS unreachable")

        final_state = run_pipeline(run_date="2024-06-01")

    # fx and forecast should still work despite news failure
    assert "ensemble_df" in final_state
    assert any("fetch_news" in e for e in final_state.get("errors", []))
