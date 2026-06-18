"""LangGraph state definition for the COP/USD intelligence pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class TopicWorkerState(TypedDict):
    """Payload que viaja en cada `Send` hacia un topic_worker (Etapa 3)."""

    cluster_topic: str
    cluster_articles: list[Any]


class PipelineState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────
    run_date: str  # ISO date string  "2024-06-01"
    horizon_days: int

    # ── FX data ───────────────────────────────────────────────────────
    fx_df: Any  # pandas.DataFrame columns: ds, y
    latest_rate: float
    rate_change_pct: float  # vs 30 days ago

    # ── News ──────────────────────────────────────────────────────────
    raw_articles: list[Any]
    has_material_news: bool  # veredicto del router (Etapa 2)
    materiality_reason: str
    headline_tags: list[dict[str, Any]]  # tags gruesos del gate (siembran clusters)
    clusters: dict[str, list[int]]  # topic → índices en raw_articles
    # Reducers: N topic_workers (Send) escriben en el mismo paso — operator.add
    # concatena sus aportes en vez de chocar.
    worker_analyses: Annotated[list[dict[str, Any]], operator.add]
    cluster_narratives: Annotated[list[str], operator.add]
    analyzed_articles: list[dict[str, Any]]  # consolidado por aggregate_signals
    news_signal: dict[str, Any]  # NewsSignal serializado (para el adjudicador)
    top_story: dict[str, Any]  # la noticia del día (agente editor)
    news_summary: str  # LLM-generated summary

    # ── Adjudicación (Etapa 4) ────────────────────────────────────────
    ts_signal: dict[str, Any]  # TimeSeriesSignal serializado
    market_signal: dict[str, Any]  # MarketSignal serializado (contexto)
    directional_call: dict[str, Any]  # DirectionalCall serializado

    # ── Memoria entre corridas (Etapa 6) ──────────────────────────────
    prior_performance: dict[str, Any]  # track-record destilado para el adjudicador

    # ── Forecast ──────────────────────────────────────────────────────
    prophet_result: Any
    arima_result: Any
    ensemble_df: Any  # pandas.DataFrame columns: ds, yhat, yhat_lower, yhat_upper
    eval_metrics: list[Any]

    # ── Report ────────────────────────────────────────────────────────
    report_markdown: str
    report_path: str  # local file path
    tweet_text: str
    tweet_id: str | None  # set after publishing

    # ── Control ───────────────────────────────────────────────────────
    # Reducer: ramas paralelas (fetch_fx / fetch_news) pueden aportar
    # errores en el mismo paso — operator.add los concatena en vez de chocar.
    errors: Annotated[list[str], operator.add]
    publish_enabled: bool
    persist_gold: bool
    # HITL (Etapa 6): cuando hitl_enabled, el nodo human_review hace `interrupt`
    # antes de publicar y espera un Command(resume=...) con la decisión humana.
    hitl_enabled: bool
    publish_approved: bool
