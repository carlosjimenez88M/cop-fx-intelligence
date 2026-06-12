"""LangGraph node functions for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from cop_fx.agents.state import PipelineState
from cop_fx.analysis.news_analyzer import NewsAnalyzer
from cop_fx.config.settings import get_settings
from cop_fx.contracts import MaterialityGate
from cop_fx.data.fx_fetcher import FXFetcher
from cop_fx.data.news_fetcher import NewsFetcher
from cop_fx.llm import get_chat_model
from cop_fx.logger import get_logger
from cop_fx.timeseries.evaluator import evaluate
from cop_fx.timeseries.models import (
    ARIMAForecaster,
    ForecastResult,
    ProphetForecaster,
    ensemble_forecast,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node: fetch_fx
# ---------------------------------------------------------------------------

def fetch_fx(state: PipelineState) -> PipelineState:
    """Download historical FX data and compute summary stats."""
    try:
        fetcher = FXFetcher()
        df = fetcher.fetch()
        latest = float(df["y"].iloc[-1])
        rate_30d_ago = float(df["y"].iloc[-30]) if len(df) >= 30 else float(df["y"].iloc[0])
        change_pct = (latest - rate_30d_ago) / rate_30d_ago * 100

        return {
            "fx_df": df,
            "latest_rate": latest,
            "rate_change_pct": round(change_pct, 2),
        }
    except Exception as exc:
        logger.error("fetch_fx failed: %s", exc)
        return {"errors": [f"fetch_fx: {exc}"]}


# ---------------------------------------------------------------------------
# Node: fetch_news
# ---------------------------------------------------------------------------

def fetch_news(state: PipelineState) -> PipelineState:
    """Download latest economic news articles."""
    try:
        fetcher = NewsFetcher()
        articles = fetcher.fetch()
        return {"raw_articles": articles}
    except Exception as exc:
        logger.error("fetch_news failed: %s", exc)
        return {"errors": [f"fetch_news: {exc}"]}


# ---------------------------------------------------------------------------
# Node: check_materiality  (Etapa 2 — router / gate de costo)
# ---------------------------------------------------------------------------

_MATERIALITY_PROMPT = """You are a gatekeeper for a Colombian FX analysis pipeline.

Given today's headlines, decide if ANY of them could materially move the
USD/COP exchange rate (monetary policy, oil/commodities, fiscal or political
risk, trade, US macro). Sports, entertainment and human-interest stories are
NOT material.

HEADLINES:
{digest}
"""


def check_materiality(state: PipelineState) -> PipelineState:
    """Cheap LLM gate: is there any FX-material news today?

    Falls open (material=True) on LLM failure: better to spend one extra
    analysis call than to silently drop a real signal.
    """
    articles = state.get("raw_articles", [])
    if not articles:
        return {
            "has_material_news": False,
            "materiality_reason": "No articles fetched today.",
        }

    digest = "\n".join(f"- {a.title}" for a in articles[:30])
    try:
        llm = get_chat_model("fast", temperature=0.0).with_structured_output(MaterialityGate)
        raw = llm.invoke(_MATERIALITY_PROMPT.format(digest=digest))
        gate = raw if isinstance(raw, MaterialityGate) else MaterialityGate.model_validate(raw)
    except Exception as exc:
        logger.warning("check_materiality LLM failed (%s) — failing open", exc)
        return {
            "has_material_news": True,
            "materiality_reason": "Materiality check unavailable; assuming material.",
        }

    logger.info("Materiality gate: %s — %s", gate.has_material_news, gate.reason)
    return {
        "has_material_news": gate.has_material_news,
        "materiality_reason": gate.reason,
    }


def route_materiality(state: PipelineState) -> str:
    """Conditional edge: full analysis only when the gate says material."""
    return "analyze_news" if state.get("has_material_news") else "skip_news"


# ---------------------------------------------------------------------------
# Node: skip_news  (ruta barata: cero tokens adicionales)
# ---------------------------------------------------------------------------

def skip_news(state: PipelineState) -> PipelineState:
    """No material news: the forecast carries the call, with low confidence."""
    reason = state.get("materiality_reason", "")
    return {
        "analyzed_articles": [],
        "news_summary": f"No FX-material news today. {reason}".strip(),
    }


# ---------------------------------------------------------------------------
# Node: analyze_news
# ---------------------------------------------------------------------------

def analyze_news(state: PipelineState) -> PipelineState:
    """Classify articles via NewsAnalyzer (structured output, tier 'fast')."""
    articles = state.get("raw_articles", [])
    if not articles:
        return {"analyzed_articles": [], "news_summary": "No news available."}

    analysis = NewsAnalyzer().analyze(articles[:20])

    analyzed = [
        {
            "title": item.article.title,
            "url": item.article.url,
            "topic": item.topic,
            "keywords": item.keywords,
            "severity": item.severity,
            "bullish_cop": item.bullish_cop,
            "reasoning": item.reasoning,
        }
        for item in analysis.items
    ]
    return {"analyzed_articles": analyzed, "news_summary": analysis.narrative}


# ---------------------------------------------------------------------------
# Node: run_forecast
# ---------------------------------------------------------------------------

def run_forecast(state: PipelineState) -> PipelineState:
    """Fit Prophet + ARIMA and build ensemble forecast."""
    df = state.get("fx_df")
    if df is None or df.empty:
        return {"errors": ["run_forecast: no FX data"]}

    horizon = state.get("horizon_days", get_settings().forecast_horizon_days)

    try:
        prophet = ProphetForecaster()
        prophet_result: ForecastResult = prophet.fit_predict(df, horizon_days=horizon)
    except Exception as exc:
        logger.error("Prophet failed: %s", exc)
        prophet_result = None  # type: ignore[assignment]

    try:
        arima = ARIMAForecaster()
        arima_result: ForecastResult = arima.fit_predict(df, horizon_days=horizon)
    except Exception as exc:
        logger.error("ARIMA failed: %s", exc)
        arima_result = None  # type: ignore[assignment]

    valid = [r for r in [prophet_result, arima_result] if r is not None]
    if not valid:
        return {"errors": ["run_forecast: both models failed"]}

    ensemble = ensemble_forecast(valid)

    # Quick sanity eval on last 30 rows
    eval_metrics = []
    test_df = df.tail(horizon)
    train_df = df.iloc[: len(df) - horizon]
    for forecaster_cls in [ProphetForecaster, ARIMAForecaster]:
        try:
            r = forecaster_cls().fit_predict(train_df, horizon_days=horizon)  # type: ignore[operator]
            eval_metrics.append(evaluate(r, test_df))
        except Exception:
            pass

    updates: PipelineState = {
        "ensemble_df": ensemble,
        "eval_metrics": eval_metrics,
    }
    if prophet_result:
        updates["prophet_result"] = prophet_result
    if arima_result:
        updates["arima_result"] = arima_result
    return updates


# ---------------------------------------------------------------------------
# Node: generate_report
# ---------------------------------------------------------------------------

def generate_report(state: PipelineState) -> PipelineState:
    """Compose a Markdown intelligence report and persist it to disk."""
    run_date = state.get("run_date", date.today().isoformat())
    latest = state.get("latest_rate", 0.0)
    change = state.get("rate_change_pct", 0.0)
    narrative = state.get("news_summary", "")
    ensemble = state.get("ensemble_df")

    forecast_table = ""
    if ensemble is not None and not ensemble.empty:
        rows = []
        for _, row in ensemble.iterrows():
            rows.append(
                f"| {row['ds'].strftime('%Y-%m-%d')} "
                f"| {row['yhat']:.2f} "
                f"| {row['yhat_lower']:.2f} – {row['yhat_upper']:.2f} |"
            )
        forecast_table = (
            "| Date | Forecast | 95% CI |\n"
            "|------|----------|--------|\n" + "\n".join(rows)
        )

    metrics_section = ""
    for m in state.get("eval_metrics", []):
        metrics_section += f"- **{m.model_name.upper()}**: MAE={m.mae:.2f}, RMSE={m.rmse:.2f}, MAPE={m.mape:.2f}%\n"

    trend_emoji = "📈" if change > 0 else "📉"
    report = f"""# COP/USD Intelligence Report — {run_date}

## Current Rate
**1 USD = {latest:,.2f} COP** {trend_emoji} ({change:+.2f}% vs 30 days ago)

## Market Narrative
{narrative}

## {get_settings().forecast_horizon_days}-Day Forecast
{forecast_table}

## Model Performance (back-test)
{metrics_section or "N/A"}

---
*Generated by cop-fx-intelligence at {datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}*
"""

    # Persist
    output_dir = Path(get_settings().report_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = str(output_dir / f"report_{run_date}.md")
    Path(report_path).write_text(report, encoding="utf-8")

    # Craft tweet (≤280 chars)
    tweet = (
        f"COP/USD — {run_date}\n"
        f"💵 1 USD = {latest:,.0f} COP ({change:+.1f}% 30d)\n"
        f"{narrative[:120]}...\n"
        f"#COP #Dólar #Colombia"
    )[:280]

    return {"report_markdown": report, "report_path": report_path, "tweet_text": tweet}


# ---------------------------------------------------------------------------
# Node: publish
# ---------------------------------------------------------------------------

def publish(state: PipelineState) -> PipelineState:
    """Post the daily tweet (only when twitter_enabled=True)."""
    if not state.get("publish_enabled", False):
        logger.info("publish: skipped (publish_enabled=False)")
        return {}

    settings = get_settings()
    if not settings.twitter_enabled:
        return {}

    tweet_text = state.get("tweet_text", "")
    if not tweet_text:
        return {}

    try:
        import tweepy

        client = tweepy.Client(
            bearer_token=settings.twitter_bearer_token.get_secret_value() if settings.twitter_bearer_token else None,
            consumer_key=settings.twitter_api_key.get_secret_value() if settings.twitter_api_key else None,
            consumer_secret=settings.twitter_api_secret.get_secret_value() if settings.twitter_api_secret else None,
            access_token=settings.twitter_access_token.get_secret_value() if settings.twitter_access_token else None,
            access_token_secret=settings.twitter_access_token_secret.get_secret_value() if settings.twitter_access_token_secret else None,
        )
        resp = client.create_tweet(text=tweet_text)
        tweet_id = str(resp.data["id"])  # type: ignore[index]
        logger.info("Tweet published: %s", tweet_id)
        return {"tweet_id": tweet_id}
    except Exception as exc:
        logger.error("publish tweet failed: %s", exc)
        return {"errors": [f"publish: {exc}"]}
