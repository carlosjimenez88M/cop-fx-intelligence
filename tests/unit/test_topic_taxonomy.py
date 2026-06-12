from __future__ import annotations

import pytest

from cop_fx.analysis.topic_taxonomy import (
    article_importance,
    channel_label,
    directional_bias,
    normalize_topic,
    research_gap,
)


@pytest.mark.unit()
def test_normalize_topic_groups_risk_family() -> None:
    assert normalize_topic("political_risk") == "Riesgo pais"
    assert normalize_topic("security_conflict") == "Riesgo pais"
    assert normalize_topic("unknown") == "Sin clasificar"


@pytest.mark.unit()
def test_article_importance_rewards_material_high_signal_news() -> None:
    row = {
        "topic": "fiscal_policy",
        "fx_relevance": "direct",
        "fx_channel": "country_risk",
        "severity": "high",
    }
    assert article_importance(row) == 1.0


@pytest.mark.unit()
def test_article_importance_zeroes_non_fx_news() -> None:
    row = {
        "topic": "sports",
        "fx_relevance": "none",
        "fx_channel": "none",
        "severity": "high",
    }
    assert article_importance(row) == 0.0
    assert directional_bias(row) == "Sin voto"


@pytest.mark.unit()
def test_directional_bias_uses_usdcop_language() -> None:
    assert (
        directional_bias(
            {"fx_relevance": "direct", "fx_channel": "country_risk", "bullish_cop": False}
        )
        == "USD/COP sube"
    )
    assert (
        directional_bias(
            {"fx_relevance": "direct", "fx_channel": "capital_flows", "bullish_cop": True}
        )
        == "USD/COP baja"
    )


@pytest.mark.unit()
def test_research_gap_flags_material_other_topic() -> None:
    row = {"topic": "other", "fx_channel": "growth", "fx_relevance": "indirect"}
    assert "taxonomia" in research_gap(row)


@pytest.mark.unit()
def test_channel_label_falls_back_to_original_value() -> None:
    assert channel_label("country_risk") == "Riesgo pais"
    assert channel_label("new_channel") == "new_channel"
