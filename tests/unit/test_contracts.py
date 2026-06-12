"""Unit tests for the Pydantic contracts (Etapa 0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cop_fx.contracts import (
    ArticleAnalysis,
    BatchAnalysis,
    DirectionalCall,
    NewsSignal,
    Severity,
    TimeSeriesSignal,
    aggregate_news_signal,
)


def _signals() -> dict:
    return {
        "news_signal": NewsSignal(direction="down", score=1.5, drivers=["BanRep sube tasas"]),
        "ts_signal": TimeSeriesSignal(direction="down", yhat_delta_pct=-0.4, models_agree=True),
    }


def _call(**overrides: object) -> DirectionalCall:
    base = {
        "direction": "down",
        "confidence": 0.7,
        "horizon_days": 5,
        "reconciliation": "agree",
        "dominant_signal": "news",
        "consistency_notes": [],
        "rationale": "Noticias y serie concuerdan en fortalecimiento del COP.",
        "devils_advocate": "El DXY podría repuntar tras el dato de empleo de EE.UU.",
        "caveats": [],
        **_signals(),
    }
    base.update(overrides)
    return DirectionalCall(**base)


@pytest.mark.unit()
def test_directional_call_valid() -> None:
    call = _call()
    assert call.direction == "down"
    assert call.confidence == 0.7


@pytest.mark.unit()
def test_divergence_caps_confidence_at_half() -> None:
    call = _call(reconciliation="diverge", confidence=0.9)
    assert call.confidence == 0.5


@pytest.mark.unit()
def test_low_confidence_forces_abstention() -> None:
    call = _call(confidence=0.2)
    assert call.direction == "neutral"


@pytest.mark.unit()
def test_no_dominant_signal_forces_abstention() -> None:
    call = _call(dominant_signal="none", confidence=0.8)
    assert call.direction == "neutral"
    assert call.confidence == 0.45


@pytest.mark.unit()
def test_devils_advocate_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        _call(devils_advocate="")


@pytest.mark.unit()
def test_article_analysis_keyword_bounds() -> None:
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            index=0,
            topic="us_global_macro",
            keywords=[],  # min_length=1
            fx_relevance="direct",
            fx_channel="growth",
            severity="low",
            bullish_cop=False,
            reasoning="x",
        )


@pytest.mark.unit()
def test_no_relevance_forces_no_channel_and_low_severity() -> None:
    # Coherencia por contrato: un partido de fútbol no puede tener canal FX.
    sports = ArticleAnalysis(
        index=0,
        topic="sports",
        keywords=["fútbol"],
        fx_relevance="none",
        fx_channel="inflation",  # incoherente a propósito
        severity="high",
        bullish_cop=True,
        reasoning="r",
    )
    assert sports.fx_channel == "none"
    assert sports.severity == "low"


@pytest.mark.unit()
def test_batch_analysis_parses_llm_dict() -> None:
    payload = {
        "items": [
            {
                "index": 0,
                "topic": "monetary_policy",
                "keywords": ["BanRep", "tasas"],
                "entities": ["BanRep"],
                "fx_relevance": "direct",
                "fx_channel": "interest_rates",
                "severity": "high",
                "bullish_cop": True,
                "reasoning": "Subida de tasas atrae flujos",
            }
        ],
        "market_narrative": "El peso se fortalece tras la decisión del BanRep.",
    }
    batch = BatchAnalysis.model_validate(payload)
    assert batch.items[0].topic == "monetary_policy"
    assert batch.items[0].fx_channel == "interest_rates"


def _analysis(
    i: int,
    bullish: bool,
    severity: Severity,
    relevance: str = "direct",
) -> ArticleAnalysis:
    return ArticleAnalysis(
        index=i,
        topic="us_global_macro",
        keywords=["kw"],
        fx_relevance=relevance,  # type: ignore[arg-type]
        fx_channel="growth" if relevance != "none" else "none",
        severity=severity,
        bullish_cop=bullish,
        reasoning="r",
    )


@pytest.mark.unit()
def test_aggregate_news_signal_directions() -> None:
    bullish = [_analysis(0, True, "high"), _analysis(1, True, "medium")]
    signal = aggregate_news_signal(bullish, {0: "Brent sube"})
    assert signal.direction == "down"  # COP fuerte => USD/COP cae
    assert signal.drivers == ["Brent sube"]

    bearish = [_analysis(0, False, "high"), _analysis(1, False, "high")]
    assert aggregate_news_signal(bearish).direction == "up"

    weak = [_analysis(0, True, "low")]
    assert aggregate_news_signal(weak).direction == "neutral"


@pytest.mark.unit()
def test_aggregate_ignores_irrelevant_and_halves_indirect() -> None:
    # 10 noticias de deportes (relevance none) no mueven la señal...
    noise = [_analysis(i, True, "high", relevance="none") for i in range(10)]
    assert aggregate_news_signal(noise).score == 0.0
    assert aggregate_news_signal(noise).direction == "neutral"

    # ...y una indirecta pesa la mitad que una directa de igual severidad.
    direct = aggregate_news_signal([_analysis(0, True, "high", relevance="direct")])
    indirect = aggregate_news_signal([_analysis(0, True, "high", relevance="indirect")])
    assert indirect.score == direct.score / 2
