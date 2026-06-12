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
def test_devils_advocate_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        _call(devils_advocate="")


@pytest.mark.unit()
def test_article_analysis_keyword_bounds() -> None:
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            index=0,
            topic="macro",
            keywords=[],  # min_length=1
            severity="low",
            bullish_cop=False,
            reasoning="x",
        )


@pytest.mark.unit()
def test_batch_analysis_parses_llm_dict() -> None:
    payload = {
        "items": [
            {
                "index": 0,
                "topic": "monetary_policy",
                "keywords": ["BanRep", "tasas"],
                "severity": "high",
                "bullish_cop": True,
                "reasoning": "Subida de tasas atrae flujos",
            }
        ],
        "market_narrative": "El peso se fortalece tras la decisión del BanRep.",
    }
    batch = BatchAnalysis.model_validate(payload)
    assert batch.items[0].topic == "monetary_policy"


@pytest.mark.unit()
def test_aggregate_news_signal_directions() -> None:
    def analysis(i: int, bullish: bool, severity: Severity) -> ArticleAnalysis:
        return ArticleAnalysis(
            index=i,
            topic="macro",
            keywords=["kw"],
            severity=severity,
            bullish_cop=bullish,
            reasoning="r",
        )

    bullish = [analysis(0, True, "high"), analysis(1, True, "medium")]
    signal = aggregate_news_signal(bullish, {0: "Brent sube"})
    assert signal.direction == "down"  # COP fuerte ⇒ USD/COP cae
    assert signal.drivers == ["Brent sube"]

    bearish = [analysis(0, False, "high"), analysis(1, False, "high")]
    assert aggregate_news_signal(bearish).direction == "up"

    weak = [analysis(0, True, "low")]
    assert aggregate_news_signal(weak).direction == "neutral"
