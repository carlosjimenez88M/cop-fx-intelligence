"""LangGraph node functions for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from langgraph.types import Send

from cop_fx.agents.state import PipelineState, TopicWorkerState
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalyzer
from cop_fx.config.settings import get_settings
from cop_fx.contracts import ArticleAnalysis, MaterialityGate, aggregate_news_signal
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

# Cota superior de workers por corrida: controla el costo en días muy noticiosos.
MAX_TOPIC_WORKERS = 6


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

For EACH headline below, produce a tag with its 0-based index, its topic
(use the full taxonomy — sports and culture have their own categories),
and whether THAT headline could materially move the USD/COP exchange rate.

Think in transmission channels, including second-order ones:
  - monetary/fiscal policy, oil/commodities, political risk, trade, US macro
    are usually material;
  - public health crises transmit via growth and fiscal cost;
  - climate events (drought, El Niño, floods) via food inflation and energy;
  - strikes and social unrest via country risk;
  - sports, entertainment and human-interest stories are NOT material.

Then set `has_material_news` = true if at least one headline is material,
and explain the overall verdict in `reason`.

HEADLINES:
{digest}
"""


def check_materiality(state: PipelineState) -> PipelineState:
    """Cheap LLM gate: is there any FX-material news today?

    The same single call also tags every headline with a coarse topic —
    those tags seed the topic clusters for the Etapa 3 fan-out, so the
    router costs nothing extra.

    Falls open (material=True) on LLM failure: better to spend one extra
    analysis call than to silently drop a real signal.
    """
    articles = state.get("raw_articles", [])
    if not articles:
        return {
            "has_material_news": False,
            "materiality_reason": "No articles fetched today.",
            "headline_tags": [],
        }

    digest = "\n".join(f"[{i}] {a.title}" for i, a in enumerate(articles[:30]))
    try:
        llm = get_chat_model("fast", temperature=0.0).with_structured_output(MaterialityGate)
        raw = llm.invoke(_MATERIALITY_PROMPT.format(digest=digest))
        gate = raw if isinstance(raw, MaterialityGate) else MaterialityGate.model_validate(raw)
    except Exception as exc:
        logger.warning("check_materiality LLM failed (%s) — failing open", exc)
        return {
            "has_material_news": True,
            "materiality_reason": "Materiality check unavailable; assuming material.",
            "headline_tags": [],
        }

    logger.info("Materiality gate: %s — %s", gate.has_material_news, gate.reason)
    return {
        "has_material_news": gate.has_material_news,
        "materiality_reason": gate.reason,
        "headline_tags": [t.model_dump() for t in gate.tags],
    }


def route_materiality(state: PipelineState) -> str:
    """Conditional edge: full analysis only when the gate says material."""
    return "orchestrate" if state.get("has_material_news") else "skip_news"


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
# Etapa 3 — orchestrator → Send(topic_worker x N) → aggregate_signals
# ---------------------------------------------------------------------------

def _analysis_to_dict(item: AnalyzedArticle) -> dict[str, object]:
    """Serializa un AnalyzedArticle a dict para el estado del grafo."""
    return {
        "title": item.article.title,
        "url": item.article.url,
        "topic": item.topic,
        "keywords": item.keywords,
        "entities": item.entities,
        "fx_relevance": item.fx_relevance,
        "fx_channel": item.fx_channel,
        "severity": item.severity,
        "bullish_cop": item.bullish_cop,
        "reasoning": item.reasoning,
    }


def orchestrate(state: PipelineState) -> PipelineState:
    """Agrupa artículos en clusters por tópico usando los tags del gate.

    Determinista, cero LLM: el gate ya pagó la etiqueta gruesa. Los tags
    no-materiales se descartan; los titulares sin tag van a "other".
    Si hay más de MAX_TOPIC_WORKERS clusters, los más pequeños se funden
    en "other" para acotar el costo del fan-out.
    """
    articles = state.get("raw_articles", [])
    tags = state.get("headline_tags", [])

    tag_by_index = {t["index"]: t for t in tags}
    clusters: dict[str, list[int]] = {}
    for i in range(min(len(articles), 30)):
        tag = tag_by_index.get(i)
        if tag is not None and not tag.get("material", True):
            continue  # el gate ya dijo que este titular no mueve el FX
        topic = tag["topic"] if tag is not None else "other"
        clusters.setdefault(topic, []).append(i)

    if not clusters:  # gate material pero sin tags utilizables → un solo cluster
        clusters = {"other": list(range(min(len(articles), 30)))}

    if len(clusters) > MAX_TOPIC_WORKERS:
        by_size = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        keep = dict(by_size[: MAX_TOPIC_WORKERS - 1])
        overflow = [i for _, idxs in by_size[MAX_TOPIC_WORKERS - 1 :] for i in idxs]
        keep.setdefault("other", []).extend(overflow)
        clusters = keep

    logger.info(
        "Orchestrator: %d clusters → %s",
        len(clusters),
        {t: len(idx) for t, idx in clusters.items()},
    )
    return {"clusters": clusters}


def fan_out_clusters(state: PipelineState) -> list[Send]:
    """Conditional edge dinámico: un Send (= un worker) por cluster de tópico."""
    articles = state.get("raw_articles", [])
    clusters = state.get("clusters", {})
    return [
        Send(
            "topic_worker",
            {
                "cluster_topic": topic,
                "cluster_articles": [articles[i] for i in indices],
            },
        )
        for topic, indices in clusters.items()
        if indices
    ]


def topic_worker(state: TopicWorkerState) -> PipelineState:
    """Worker por cluster: análisis profundo de SOLO sus artículos.

    Recibe el payload del `Send` (no el estado global). Prompt chaining
    interno vía NewsAnalyzer (extraer → clasificar → canal de transmisión
    → impacto). Su salida se fusiona al estado global por los reducers de
    worker_analyses / cluster_narratives.
    """
    topic = state.get("cluster_topic", "other")
    articles = state.get("cluster_articles", [])
    if not articles:
        return {"worker_analyses": [], "cluster_narratives": []}

    analysis = NewsAnalyzer().analyze(articles)
    logger.info("topic_worker[%s]: %d artículos analizados", topic, len(analysis.items))
    return {
        "worker_analyses": [_analysis_to_dict(item) for item in analysis.items],
        "cluster_narratives": [f"[{topic}] {analysis.narrative}"],
    }


def aggregate_signals(state: PipelineState) -> PipelineState:
    """Consolida los workers en una señal direccional — determinista, sin LLM."""
    analyzed = state.get("worker_analyses", [])

    analyses = []
    titles: dict[int, str] = {}
    for i, d in enumerate(analyzed):
        titles[i] = str(d.get("title", ""))
        analyses.append(
            ArticleAnalysis(
                index=i,
                topic=d.get("topic", "other"),
                keywords=list(d.get("keywords") or ["sin_clasificar"]),
                entities=list(d.get("entities") or []),
                fx_relevance=d.get("fx_relevance", "none"),
                fx_channel=d.get("fx_channel", "none"),
                severity=d.get("severity", "low"),
                bullish_cop=bool(d.get("bullish_cop", False)),
                reasoning=str(d.get("reasoning", ""))[:240],
            )
        )

    signal = aggregate_news_signal(analyses, titles)
    summary = "\n".join(state.get("cluster_narratives", [])) or "No analysis available."
    logger.info(
        "Señal de noticias: %s (score=%.2f, drivers=%d)",
        signal.direction,
        signal.score,
        len(signal.drivers),
    )
    return {
        "analyzed_articles": analyzed,
        "news_signal": signal.model_dump(),
        "news_summary": summary,
    }


# ---------------------------------------------------------------------------
# Node: analyze_news  (modo una-sola-llamada — Etapa 1; lo usan notebooks/tests)
# ---------------------------------------------------------------------------

def analyze_news(state: PipelineState) -> PipelineState:
    """Classify articles via NewsAnalyzer (structured output, tier 'fast')."""
    articles = state.get("raw_articles", [])
    if not articles:
        return {"analyzed_articles": [], "news_summary": "No news available."}

    analysis = NewsAnalyzer().analyze(articles[:20])
    analyzed = [_analysis_to_dict(item) for item in analysis.items]
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
