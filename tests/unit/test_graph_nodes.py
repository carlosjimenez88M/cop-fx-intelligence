"""Unit tests for LangGraph node functions."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langgraph.types import Send

from cop_fx.agents.nodes import (
    MAX_TOPIC_WORKERS,
    adjudicate,
    aggregate_signals,
    analyze_news,
    check_materiality,
    compute_ts_signal,
    fan_out_clusters,
    fetch_fx,
    fetch_market,
    fetch_news,
    generate_report,
    orchestrate,
    route_materiality,
    skip_news,
    topic_worker,
)
from cop_fx.agents.state import PipelineState
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalysis
from cop_fx.contracts import MaterialityGate
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
            published_at=datetime.now(tz=UTC),
            source="Test",
        )
    ]
    with patch("cop_fx.agents.nodes.NewsFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.return_value = fake_articles
        result = fetch_news(_base_state())

    assert result["raw_articles"] == fake_articles


# ── check_materiality / router / skip_news ───────────────────────────────

def _fake_article(title: str = "BanRep sube tasas") -> Article:
    return Article(
        title=title,
        summary="El Banco de la República ajustó su tasa de referencia ante presiones inflacionarias persistentes en alimentos.",
        url="https://example.com/a",
        published_at=datetime.now(tz=UTC),
        source="Test",
    )


@pytest.mark.unit()
def test_check_materiality_no_articles_is_not_material() -> None:
    result = check_materiality(_base_state(raw_articles=[]))
    assert result["has_material_news"] is False


@pytest.mark.unit()
def test_check_materiality_uses_gate_verdict() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = MaterialityGate(
        has_material_news=True, reason="BanRep decision moves rates"
    )
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = check_materiality(_base_state(raw_articles=[_fake_article()]))

    assert result["has_material_news"] is True
    assert "BanRep" in result["materiality_reason"]


@pytest.mark.unit()
def test_check_materiality_fails_open_on_llm_error() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("api down")
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = check_materiality(_base_state(raw_articles=[_fake_article()]))

    assert result["has_material_news"] is True  # fail-open: no perder señal real


@pytest.mark.unit()
def test_route_materiality_branches() -> None:
    assert route_materiality(_base_state(has_material_news=True)) == "orchestrate"
    assert route_materiality(_base_state(has_material_news=False)) == "skip_news"
    assert route_materiality(_base_state()) == "skip_news"  # ausente = no material


@pytest.mark.unit()
def test_skip_news_sets_empty_analysis_with_reason() -> None:
    result = skip_news(_base_state(materiality_reason="Only sports today."))
    assert result["analyzed_articles"] == []
    assert "Only sports today." in result["news_summary"]


# ── Etapa 3: orchestrate / fan_out / topic_worker / aggregate_signals ────

@pytest.mark.unit()
def test_orchestrate_clusters_by_gate_tags() -> None:
    articles = [_fake_article(f"t{i}") for i in range(4)]
    tags = [
        {"index": 0, "topic": "monetary_policy", "material": True},
        {"index": 1, "topic": "sports", "material": False},   # descartado
        {"index": 2, "topic": "monetary_policy", "material": True},
        # índice 3 sin tag → cae en "other"
    ]
    result = orchestrate(_base_state(raw_articles=articles, headline_tags=tags))

    clusters = result["clusters"]
    assert clusters["monetary_policy"] == [0, 2]
    assert clusters["other"] == [3]
    assert all(1 not in idx for idx in clusters.values())


@pytest.mark.unit()
def test_orchestrate_caps_worker_count() -> None:
    articles = [_fake_article(f"t{i}") for i in range(10)]
    tags = [
        {"index": i, "topic": topic, "material": True}
        for i, topic in enumerate(
            ["monetary_policy", "fiscal_policy", "trade", "political_risk",
             "security_conflict", "energy_commodities", "agro_commodities",
             "public_health", "environment_climate", "labor_social"]
        )
    ]
    result = orchestrate(_base_state(raw_articles=articles, headline_tags=tags))

    clusters = result["clusters"]
    assert len(clusters) <= MAX_TOPIC_WORKERS
    assert sum(len(v) for v in clusters.values()) == 10  # nada se pierde


@pytest.mark.unit()
def test_fan_out_emits_one_send_per_cluster() -> None:
    articles = [_fake_article(f"t{i}") for i in range(3)]
    state = _base_state(
        raw_articles=articles,
        clusters={"monetary_policy": [0, 2], "trade": [1]},
    )
    sends = fan_out_clusters(state)

    assert all(isinstance(s, Send) for s in sends)
    assert {s.node for s in sends} == {"topic_worker"}
    by_topic = {s.arg["cluster_topic"]: s.arg["cluster_articles"] for s in sends}
    assert [a.title for a in by_topic["monetary_policy"]] == ["t0", "t2"]
    assert [a.title for a in by_topic["trade"]] == ["t1"]


@pytest.mark.unit()
def test_topic_worker_returns_reducer_updates() -> None:
    article = _fake_article()
    verdict = AnalyzedArticle(
        article=article,
        topic="monetary_policy",
        severity="high",
        bullish_cop=True,
        reasoning="Rate hike",
        keywords=["tasas"],
        entities=["BanRep"],
        fx_relevance="direct",
        fx_channel="interest_rates",
    )
    with (
        patch("cop_fx.agents.nodes.NewsAnalyzer") as MockAnalyzer,
        patch("cop_fx.agents.nodes.attach_bodies"),  # sin red en tests
    ):
        MockAnalyzer.return_value.analyze.return_value = NewsAnalysis(
            items=[verdict], narrative="Peso firme."
        )
        result = topic_worker(
            {"cluster_topic": "monetary_policy", "cluster_articles": [article]}
        )

    assert result["worker_analyses"][0]["fx_channel"] == "interest_rates"
    assert result["cluster_narratives"] == ["[monetary_policy] Peso firme."]


@pytest.mark.unit()
def test_aggregate_signals_is_deterministic() -> None:
    worker_analyses = [
        {
            "title": "BanRep sube tasas",
            "url": "https://x.com/1",
            "topic": "monetary_policy",
            "keywords": ["tasas"],
            "entities": ["BanRep"],
            "fx_relevance": "direct",
            "fx_channel": "interest_rates",
            "severity": "high",
            "bullish_cop": True,
            "reasoning": "r",
        },
        {
            "title": "Colombia gana el partido",
            "url": "https://x.com/2",
            "topic": "sports",
            "keywords": ["fútbol"],
            "entities": [],
            "fx_relevance": "none",
            "fx_channel": "none",
            "severity": "low",
            "bullish_cop": True,   # irrelevante: pesa 0
            "reasoning": "r",
        },
    ]
    result = aggregate_signals(
        _base_state(
            worker_analyses=worker_analyses,
            cluster_narratives=["[monetary_policy] Peso firme."],
        )
    )

    signal = result["news_signal"]
    assert signal["direction"] == "down"      # solo pesa la noticia de tasas
    assert signal["score"] == 1.0
    assert signal["drivers"] == ["BanRep sube tasas"]
    assert "[monetary_policy]" in result["news_summary"]
    assert len(result["analyzed_articles"]) == 2


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
            summary="La junta directiva del Banco de la República decidió mantener inalterada su tasa de referencia ante la persistencia inflacionaria.",
            url="https://x.com",
            published_at=datetime.now(tz=UTC),
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
        entities=["BanRep"],
        fx_relevance="direct",
        fx_channel="interest_rates",
    )
    mock_analysis = NewsAnalysis(items=[verdict], narrative="Market is neutral.")

    with (
        patch("cop_fx.agents.nodes.NewsAnalyzer") as MockAnalyzer,
        patch("cop_fx.agents.nodes.attach_bodies"),  # sin red en tests
    ):
        MockAnalyzer.return_value.analyze.return_value = mock_analysis
        result = analyze_news(_base_state(raw_articles=articles))

    analyzed = result.get("analyzed_articles", [])
    assert len(analyzed) == 1
    assert analyzed[0]["topic"] == "monetary_policy"
    assert analyzed[0]["keywords"] == ["BanRep", "tasas"]
    assert result.get("news_summary") == "Market is neutral."


# ── fetch_market (integración del estudio macro) ─────────────────────────

def _market_series(last_two: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ds": pd.date_range("2026-06-01", periods=2, freq="D"), "y": list(last_two)}
    )


@pytest.mark.unit()
def test_fetch_market_equity_rule() -> None:
    # bolsa +1% ayer ⇒ COP se fortalece ⇒ USD/COP down
    series = {
        "GXG": _market_series((30.0, 30.3)),       # +1.0%
        "DX-Y.NYB": _market_series((100.0, 100.1)),
        "BZ=F": _market_series((70.0, 69.0)),
    }
    with patch(
        "cop_fx.agents.nodes.fetch_yahoo_series",
        side_effect=lambda symbol, lookback_days=30: series[symbol],
    ):
        result = fetch_market(_base_state())

    signal = result["market_signal"]
    assert signal["direction"] == "down"
    assert signal["equity_ret_1d_pct"] == 1.0


@pytest.mark.unit()
def test_fetch_market_dead_band_is_neutral() -> None:
    series = {
        "GXG": _market_series((30.0, 30.03)),      # +0.1% < banda 0.3%
        "DX-Y.NYB": _market_series((100.0, 100.0)),
        "BZ=F": _market_series((70.0, 70.0)),
    }
    with patch(
        "cop_fx.agents.nodes.fetch_yahoo_series",
        side_effect=lambda symbol, lookback_days=30: series[symbol],
    ):
        result = fetch_market(_base_state())

    assert result["market_signal"]["direction"] == "neutral"


@pytest.mark.unit()
def test_fetch_market_fails_soft() -> None:
    with patch(
        "cop_fx.agents.nodes.fetch_yahoo_series",
        side_effect=ConnectionError("yahoo down"),
    ):
        result = fetch_market(_base_state())

    assert result["market_signal"] == {}  # contexto opcional: no rompe el pipeline


# ── Etapa 4: compute_ts_signal / adjudicate ──────────────────────────────

def _ensemble_df(latest: float, yhat_final: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-06-03", periods=3, freq="B"),
            "yhat": [latest, (latest + yhat_final) / 2, yhat_final],
            "yhat_lower": [latest - 20] * 3,
            "yhat_upper": [latest + 20] * 3,
        }
    )


@pytest.mark.unit()
def test_compute_ts_signal_directions() -> None:
    up = compute_ts_signal(
        latest=4000.0,
        ensemble_df=_ensemble_df(4000.0, 4080.0),  # +2%
        prophet_result=None,
        arima_result=None,
    )
    assert up.direction == "up"
    assert up.models_agree is False  # un solo modelo (o ninguno) nunca "concuerda"

    flat = compute_ts_signal(
        latest=4000.0,
        ensemble_df=_ensemble_df(4000.0, 4001.0),  # +0.025% < banda muerta
        prophet_result=None,
        arima_result=None,
    )
    assert flat.direction == "neutral"


def _news(direction: str, score: float) -> dict:
    return {"direction": direction, "score": score, "drivers": ["BanRep sube tasas"]}


def _ts(direction: str, delta: float) -> dict:
    return {"direction": direction, "yhat_delta_pct": delta, "models_agree": True}


@pytest.mark.unit()
def test_adjudicate_uses_judge_verdict() -> None:
    from cop_fx.contracts import AdjudicatorVerdict

    verdict = AdjudicatorVerdict(
        direction="down",
        confidence=0.72,
        reconciliation="agree",
        rationale="Noticias y serie apuntan a COP fuerte por tasas.",
        devils_advocate="El DXY podría repuntar tras el dato de empleo en EE.UU.",
        caveats=["Fuente única (CNN)"],
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = verdict
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = adjudicate(
            _base_state(news_signal=_news("down", 2.0), ts_signal=_ts("down", -0.5))
        )

    call = result["directional_call"]
    assert call["direction"] == "down"
    assert call["confidence"] == 0.72
    assert call["news_signal"]["score"] == 2.0      # inyectado por el sistema
    assert call["ts_signal"]["yhat_delta_pct"] == -0.5


@pytest.mark.unit()
def test_adjudicate_divergence_cap_applies_to_llm_output() -> None:
    from cop_fx.contracts import AdjudicatorVerdict

    overconfident = AdjudicatorVerdict(
        direction="up",
        confidence=0.95,                 # el LLM exagera...
        reconciliation="diverge",        # ...en plena divergencia
        rationale="r",
        devils_advocate="Las noticias apuntan exactamente en la dirección contraria.",
        caveats=[],
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = overconfident
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = adjudicate(
            _base_state(news_signal=_news("down", 2.0), ts_signal=_ts("up", 0.8))
        )

    assert result["directional_call"]["confidence"] == 0.5  # acotado por contrato


@pytest.mark.unit()
def test_adjudicate_falls_back_deterministically() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("api down")
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        agree = adjudicate(
            _base_state(news_signal=_news("down", 2.0), ts_signal=_ts("down", -0.5))
        )["directional_call"]
        diverge = adjudicate(
            _base_state(news_signal=_news("down", 2.0), ts_signal=_ts("up", 0.8))
        )["directional_call"]
        partial = adjudicate(
            _base_state(news_signal=_news("down", 2.0), ts_signal=_ts("neutral", 0.0))
        )["directional_call"]

    assert (agree["direction"], agree["confidence"]) == ("down", 0.6)
    assert (diverge["direction"], diverge["confidence"]) == ("neutral", 0.3)
    assert (partial["direction"], partial["reconciliation"]) == ("down", "partial")


@pytest.mark.unit()
def test_adjudicate_without_signals_abstains() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("api down")
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        call = adjudicate(_base_state())["directional_call"]

    assert call["direction"] == "neutral"


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


# ── order_articles: mismo día = mismo pie ────────────────────────────────

@pytest.mark.unit()
def test_order_articles_same_day_interleaves_sources() -> None:
    from datetime import datetime, timedelta

    from cop_fx.data.news_fetcher import order_articles

    def art(source: str, title: str, dt: datetime) -> Article:
        return Article(title=title, summary="x" * 100, url=f"https://x.com/{title}",
                       published_at=dt, source=source)

    today = datetime(2026, 6, 12, tzinfo=UTC)
    arts = [
        # RSS con hora fina (tarde) — antes monopolizaba el tope
        art("Portafolio", "p1", today.replace(hour=15)),
        art("Portafolio", "p2", today.replace(hour=14)),
        art("Portafolio", "p3", today.replace(hour=13)),
        # CNN con fecha sin hora (medianoche) — antes quedaba al fondo
        art("CNN", "c1", today),
        art("CNN", "c2", today),
        # ayer
        art("Portafolio", "y1", today - timedelta(days=1)),
    ]
    ordered = order_articles(arts)
    top4_sources = [a.source for a in ordered[:4]]

    # Mismo día: las fuentes se intercalan — CNN aparece en el top aunque
    # su timestamp sea medianoche
    assert "CNN" in top4_sources[:2]
    assert "Portafolio" in top4_sources[:2]
    # La fecha sigue mandando: lo de ayer va al final
    assert ordered[-1].title == "y1"


# ── pick_top_story: el agente editor ─────────────────────────────────────

def _wa(title: str, severity: str = "high", relevance: str = "direct") -> dict:
    return {
        "title": title, "source": "La República", "url": f"https://x.com/{title}",
        "topic": "fiscal_policy", "fx_channel": "country_risk",
        "severity": severity, "fx_relevance": relevance, "bullish_cop": False,
        "reasoning": "presión fiscal",
    }


@pytest.mark.unit()
def test_pick_top_story_uses_editor_choice() -> None:
    from cop_fx.agents.nodes import pick_top_story
    from cop_fx.contracts import TopStory

    structured_llm = MagicMock()
    structured_llm.invoke.return_value = TopStory(
        chosen_index=1,
        why_it_matters="La reforma tributaria redefine la senda fiscal y la prima de riesgo país.",
        watch_next="El texto del proyecto de ley y la reacción de las calificadoras.",
    )
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    analyses = [_wa("Dato de empleo"), _wa("Reforma tributaria"), _wa("Peaje sube", "low")]
    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = pick_top_story(_base_state(worker_analyses=analyses))

    top = result["top_story"]
    assert top["title"] == "Reforma tributaria"          # eligió el índice 1
    assert top["source"] == "La República"               # hechos del sistema, no del LLM
    assert "fiscal" in top["why_it_matters"]


@pytest.mark.unit()
def test_pick_top_story_fallback_takes_heaviest() -> None:
    from cop_fx.agents.nodes import pick_top_story

    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("api down")
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm

    analyses = [_wa("Ruido", "low", "indirect"), _wa("Shock fiscal", "high", "direct")]
    with patch("cop_fx.agents.nodes.get_chat_model", return_value=base_llm):
        result = pick_top_story(_base_state(worker_analyses=analyses))

    assert result["top_story"]["title"] == "Shock fiscal"  # mayor severidad×relevancia


@pytest.mark.unit()
def test_pick_top_story_empty_when_no_relevant_news() -> None:
    from cop_fx.agents.nodes import pick_top_story

    analyses = [_wa("Partido de fútbol", "low", "none")]
    result = pick_top_story(_base_state(worker_analyses=analyses))
    assert result["top_story"] == {}  # sin candidatos: ni siquiera llama al LLM
