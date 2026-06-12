"""LangGraph state definition for the COP/USD intelligence pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pandas as pd

from cop_fx.data.news_fetcher import Article
from cop_fx.timeseries.evaluator import EvalMetrics
from cop_fx.timeseries.models import ForecastResult


class PipelineState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────
    run_date: str                   # ISO date string  "2024-06-01"
    horizon_days: int

    # ── FX data ───────────────────────────────────────────────────────
    fx_df: pd.DataFrame             # columns: ds, y
    latest_rate: float
    rate_change_pct: float          # vs 30 days ago

    # ── News ──────────────────────────────────────────────────────────
    raw_articles: list[Article]
    analyzed_articles: list[dict[str, Any]]  # enriched with topic/severity
    news_summary: str               # LLM-generated summary

    # ── Forecast ──────────────────────────────────────────────────────
    prophet_result: ForecastResult
    arima_result: ForecastResult
    ensemble_df: pd.DataFrame       # columns: ds, yhat, yhat_lower, yhat_upper
    eval_metrics: list[EvalMetrics]

    # ── Report ────────────────────────────────────────────────────────
    report_markdown: str
    report_path: str                # local file path
    tweet_text: str
    tweet_id: str | None            # set after publishing

    # ── Control ───────────────────────────────────────────────────────
    # Reducer: ramas paralelas (fetch_fx / fetch_news) pueden aportar
    # errores en el mismo paso — operator.add los concatena en vez de chocar.
    errors: Annotated[list[str], operator.add]
    publish_enabled: bool
