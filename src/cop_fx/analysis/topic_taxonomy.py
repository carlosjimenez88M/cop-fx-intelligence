"""Topic normalization and news-importance scoring.

This module keeps the product taxonomy outside the dashboard so notebooks,
reports and future pipeline steps can reuse the same decision logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cop_fx.contracts import RELEVANCE_WEIGHT, SEVERITY_WEIGHT

if TYPE_CHECKING:
    from collections.abc import Mapping

TOPIC_FAMILY: dict[str, str] = {
    "monetary_policy": "Politica monetaria",
    "fiscal_policy": "Riesgo fiscal",
    "financial_markets": "Mercados y flujos",
    "us_global_macro": "Dolar global",
    "political_risk": "Riesgo pais",
    "security_conflict": "Riesgo pais",
    "labor_social": "Riesgo pais",
    "trade": "Sector externo",
    "energy_commodities": "Commodities",
    "agro_commodities": "Commodities",
    "environment_climate": "Oferta e inflacion",
    "public_health": "Actividad interna",
    "culture_society": "Baja senal FX",
    "sports": "Baja senal FX",
    "other": "Sin clasificar",
}

CHANNEL_LABELS: dict[str, str] = {
    "interest_rates": "Tasas",
    "inflation": "Inflacion",
    "terms_of_trade": "Terminos de intercambio",
    "country_risk": "Riesgo pais",
    "capital_flows": "Flujos de capital",
    "growth": "Crecimiento",
    "none": "Sin canal FX",
}

HIGH_SIGNAL_TOPICS = {"monetary_policy", "fiscal_policy"}


def normalize_topic(topic: str | None) -> str:
    """Return a decision-family label for the raw classifier topic."""
    return TOPIC_FAMILY.get(str(topic or "other"), "Sin clasificar")


def channel_label(channel: str | None) -> str:
    """Return a readable transmission-channel label."""
    channel_value = str(channel or "none")
    return CHANNEL_LABELS.get(channel_value, channel_value)


def article_importance(row: Mapping[str, Any]) -> float:
    """Score article materiality from stable fields, not from prose.

    The score is intentionally conservative: relevance and severity dominate;
    channel and high-signal topics are only small bonuses.
    """
    severity = SEVERITY_WEIGHT.get(str(row.get("severity")), 0.0)
    relevance = RELEVANCE_WEIGHT.get(str(row.get("fx_relevance")), 0.0)
    channel_bonus = 0.0 if row.get("fx_channel") == "none" else 0.15
    topic_bonus = 0.1 if row.get("topic") in HIGH_SIGNAL_TOPICS else 0.0
    return round(min(1.0, severity * relevance + channel_bonus + topic_bonus), 3)


def directional_bias(row: Mapping[str, Any]) -> str:
    """Human-readable USD/COP directional bias from an article row."""
    if row.get("fx_relevance") == "none" or row.get("fx_channel") == "none":
        return "Sin voto"
    if bool(row.get("bullish_cop")):
        return "USD/COP baja"
    return "USD/COP sube"


def research_gap(row: Mapping[str, Any]) -> str:
    """Flag classification cases that deserve analyst review."""
    topic = str(row.get("topic", "other"))
    channel = str(row.get("fx_channel", "none"))
    relevance = str(row.get("fx_relevance", "none"))
    if topic == "other" and relevance != "none":
        return "Revisar taxonomia: material sin topic especifico"
    if relevance != "none" and channel == "none":
        return "Inconsistencia: material sin canal FX"
    if topic in {"sports", "culture_society"} and relevance != "none":
        return "Revisar falso positivo de baja senal FX"
    if topic == "security_conflict" and channel == "none":
        return "Evaluar canal riesgo pais"
    return ""
