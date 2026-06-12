"""LangGraph node functions for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.types import Send

if TYPE_CHECKING:
    import pandas as pd

from cop_fx.agents.state import PipelineState, TopicWorkerState
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalyzer
from cop_fx.config.settings import get_settings
from cop_fx.contracts import (
    AdjudicatorVerdict,
    ArticleAnalysis,
    DirectionalCall,
    MarketSignal,
    MaterialityGate,
    NewsSignal,
    TimeSeriesSignal,
    aggregate_news_signal,
)
from cop_fx.data.article_body import attach_bodies
from cop_fx.data.fx_fetcher import FXFetcher
from cop_fx.data.market_fetcher import MARKET_SYMBOLS, fetch_yahoo_series
from cop_fx.data.news_fetcher import NewsFetcher
from cop_fx.llm import get_chat_model
from cop_fx.logger import get_logger
from cop_fx.paths import PROJECT_ROOT
from cop_fx.timeseries.evaluator import evaluate
from cop_fx.timeseries.models import (
    ARIMAForecaster,
    ForecastResult,
    ProphetForecaster,
    ensemble_forecast,
)
from cop_fx.tracking.predictions import PredictionStore

logger = get_logger(__name__)

# Knobs operativos — viven en config.yaml (raíz), no aquí. Estos alias de
# módulo existen para legibilidad de los nodos y de los tests.
_cfg = get_settings()
MAX_TOPIC_WORKERS = _cfg.max_topic_workers
TS_NEUTRAL_BAND_PCT = _cfg.ts_neutral_band_pct
MARKET_DEAD_BAND_PCT = _cfg.market_dead_band_pct
GATE_HEADLINES_CAP = _cfg.gate_headlines_cap
MIN_ANALYZABLE_CHARS = _cfg.min_analyzable_chars


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
# Node: fetch_market  (integración del estudio macro — notebooks/02)
# ---------------------------------------------------------------------------

def fetch_market(state: PipelineState) -> PipelineState:
    """Contexto de mercado determinista para el adjudicador.

    Aplica el hallazgo validado del estudio macro: el agregado bursátil
    colombiano (GXG) de AYER es el único predictor adelantado robusto del
    USD/COP (equity[t-1]→cop[t] ≈ -0.4). Bolsa arriba ⇒ COP se fortalece
    ⇒ dirección `down`. DXY y Brent viajan como contexto, sin voto.

    Es contexto OPCIONAL: si Yahoo falla, el pipeline sigue sin él.
    """
    try:
        rets: dict[str, float] = {}
        for name in ("equity", "dxy", "brent"):
            series = fetch_yahoo_series(MARKET_SYMBOLS[name], lookback_days=30)
            rets[name] = float(series["y"].iloc[-1] / series["y"].iloc[-2] - 1) * 100

        equity = rets["equity"]
        if equity > MARKET_DEAD_BAND_PCT:
            direction = "down"   # bolsa arriba ⇒ apetito por Colombia ⇒ USD/COP baja
        elif equity < -MARKET_DEAD_BAND_PCT:
            direction = "up"
        else:
            direction = "neutral"

        signal = MarketSignal(
            direction=direction,  # type: ignore[arg-type]
            equity_ret_1d_pct=round(equity, 3),
            dxy_ret_1d_pct=round(rets["dxy"], 3),
            brent_ret_1d_pct=round(rets["brent"], 3),
        )
        logger.info(
            "Market context: %s (equity %+.2f%%, dxy %+.2f%%, brent %+.2f%%)",
            signal.direction, equity, rets["dxy"], rets["brent"],
        )
        return {"market_signal": signal.model_dump()}
    except Exception as exc:
        logger.warning("fetch_market failed (%s) — el contexto es opcional", exc)
        return {"market_signal": {}}


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
  - foreign macro OUTSIDE the US (Europe, UK, Asia) matters only if it moves
    global risk appetite, oil or the dollar index — on its own it is usually
    NOT material for USD/COP;
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

    digest = "\n".join(f"[{i}] {a.title}" for i, a in enumerate(articles[:GATE_HEADLINES_CAP]))
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
    for i in range(min(len(articles), GATE_HEADLINES_CAP)):
        tag = tag_by_index.get(i)
        if tag is not None and not tag.get("material", True):
            continue  # el gate ya dijo que este titular no mueve el FX
        topic = tag["topic"] if tag is not None else "other"
        clusters.setdefault(topic, []).append(i)

    if not clusters:  # gate material pero sin tags utilizables → un solo cluster
        clusters = {"other": list(range(min(len(articles), GATE_HEADLINES_CAP)))}

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

    # El veredicto se hace sobre la NOTICIA COMPLETA: el gate ya filtró por
    # titular (barato); aquí — solo para los artículos materiales — se
    # descarga el cuerpo antes de analizar. Sin texto analizable (ni cuerpo
    # ni summary decente) el artículo NO aporta nada y se descarta.
    attach_bodies(articles)
    readable = [
        a for a in articles
        if len(getattr(a, "body", "") or a.summary) >= MIN_ANALYZABLE_CHARS
    ]
    if len(readable) < len(articles):
        logger.info(
            "topic_worker[%s]: %d artículos sin texto analizable descartados",
            topic, len(articles) - len(readable),
        )
    if not readable:
        return {"worker_analyses": [], "cluster_narratives": []}

    analysis = NewsAnalyzer().analyze(readable)
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

    pool = articles[:30]
    attach_bodies(pool)  # veredictos sobre la noticia completa
    readable = [
        a for a in pool
        if len(getattr(a, "body", "") or a.summary) >= MIN_ANALYZABLE_CHARS
    ][:20]
    if not readable:
        return {"analyzed_articles": [], "news_summary": "No analyzable news today."}
    analysis = NewsAnalyzer().analyze(readable)
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
        "ts_signal": compute_ts_signal(
            latest=float(df["y"].iloc[-1]),
            ensemble_df=ensemble,
            prophet_result=prophet_result,
            arima_result=arima_result,
        ).model_dump(),
    }
    if prophet_result:
        updates["prophet_result"] = prophet_result
    if arima_result:
        updates["arima_result"] = arima_result
    return updates


def compute_ts_signal(
    *,
    latest: float,
    ensemble_df: pd.DataFrame,
    prophet_result: ForecastResult | None,
    arima_result: ForecastResult | None,
) -> TimeSeriesSignal:
    """Señal direccional del forecast — determinista, cero LLM.

    Solo importa el SIGNO del Δ entre el yhat final del ensemble y el último
    valor real; dentro de la banda muerta la serie se abstiene. `models_agree`
    exige que Prophet y ARIMA apunten al mismo lado (con uno solo, False).
    """
    yhat_final = float(ensemble_df["yhat"].iloc[-1])
    delta_pct = (yhat_final - latest) / latest * 100

    if delta_pct > TS_NEUTRAL_BAND_PCT:
        direction = "up"
    elif delta_pct < -TS_NEUTRAL_BAND_PCT:
        direction = "down"
    else:
        direction = "neutral"

    def _sign(result: ForecastResult | None) -> int:
        if result is None:
            return 0
        delta = float(result.forecast["yhat"].iloc[-1]) - latest
        return 1 if delta > 0 else -1

    models_agree = (
        prophet_result is not None
        and arima_result is not None
        and _sign(prophet_result) == _sign(arima_result)
    )
    return TimeSeriesSignal(
        direction=direction,  # type: ignore[arg-type]
        yhat_delta_pct=round(delta_pct, 3),
        models_agree=models_agree,
    )


# ---------------------------------------------------------------------------
# Node: adjudicate  (Etapa 4 — la capa de racionalidad)
# ---------------------------------------------------------------------------

_ADJUDICATOR_PROMPT = """You are the adjudicator of a Colombian FX intelligence system.
You must reconcile two INDEPENDENT signals about the USD/COP direction for
the next {horizon} days and produce a single, honestly-calibrated verdict.

NEWS SIGNAL (aggregated from today's analyzed articles, weighted by
severity x FX-relevance; score > 0 means COP strengthens => USD/COP DOWN):
{news_signal}

Cluster narratives:
{narratives}

TIME-SERIES SIGNAL (Prophet+ARIMA ensemble, deterministic sign):
{ts_signal}

MARKET CONTEXT (deterministic, yesterday's moves). The Colombian equity
aggregate is the ONLY empirically validated leading indicator of USD/COP
(equity up yesterday => COP tends to strengthen => USD/COP DOWN); DXY and
Brent are background, not votes. Use this as evidence to break ties or
temper confidence — do not treat it as a third signal to echo:
{market_signal}

Current rate: 1 USD = {latest:,.2f} COP ({change:+.2f}% vs 30 days ago).

Rules of reasoning — in this order:
1. State whether the signals agree, diverge, or one abstains (partial),
   BEFORE choosing a direction.
2. If they diverge, explain which signal should dominate and WHY (e.g. a
   high-severity news shock can override a mild technical trend; a strong
   trend can override weak, low-relevance news).
3. devils_advocate: build the STRONGEST case against your chosen direction.
   If you cannot rebut it convincingly, lower your confidence.
4. Calibrate: divergence caps confidence at 0.5 (enforced by code);
   confidence below 0.35 turns the call neutral (enforced). Abstaining
   (neutral) is a valid, professional answer — never fake conviction.
5. rationale must cite the specific drivers/headlines and the forecast sign.
6. caveats: data gaps, single-source bias, stale articles, model disagreement.
7. Write rationale, devils_advocate and caveats in SPANISH.
"""


def _fallback_call(
    news: NewsSignal, ts: TimeSeriesSignal, horizon: int
) -> DirectionalCall:
    """Reconciliación determinista cuando el adjudicador LLM no está disponible."""
    if news.direction == ts.direction:
        reconciliation, direction = "agree", news.direction
        confidence = 0.6 if news.direction != "neutral" else 0.4
    elif "neutral" in (news.direction, ts.direction):
        reconciliation = "partial"
        direction = news.direction if ts.direction == "neutral" else ts.direction
        confidence = 0.45
    else:
        reconciliation, direction, confidence = "diverge", "neutral", 0.3

    return DirectionalCall(
        direction=direction,
        confidence=confidence,
        horizon_days=horizon,
        news_signal=news,
        ts_signal=ts,
        reconciliation=reconciliation,  # type: ignore[arg-type]
        rationale=(
            f"Fallback determinista: noticias={news.direction} (score {news.score}), "
            f"serie={ts.direction} ({ts.yhat_delta_pct:+.2f}%)."
        ),
        devils_advocate=(
            "Sin adjudicador LLM el contra-argumento no fue explorado; "
            "tratar este veredicto con cautela adicional."
        ),
        caveats=["Adjudicador LLM no disponible — reconciliación por reglas fijas."],
    )


def adjudicate(state: PipelineState) -> PipelineState:
    """Cruza la señal de noticias contra la de la serie → DirectionalCall.

    El LLM (tier 'judge') produce solo el veredicto (AdjudicatorVerdict);
    las señales y el horizonte los inyecta el sistema al componer el
    DirectionalCall, cuyos validadores acotan la confianza y fuerzan la
    abstención. El LLM juzga; el código gobierna.
    """
    horizon = state.get("horizon_days", get_settings().forecast_horizon_days)
    raw_news = state.get("news_signal")
    news = (
        NewsSignal.model_validate(raw_news)
        if raw_news
        else NewsSignal(direction="neutral", score=0.0, drivers=[])
    )
    raw_ts = state.get("ts_signal")
    ts = (
        TimeSeriesSignal.model_validate(raw_ts)
        if raw_ts
        else TimeSeriesSignal(direction="neutral", yhat_delta_pct=0.0, models_agree=False)
    )

    market = state.get("market_signal") or {}
    prompt = _ADJUDICATOR_PROMPT.format(
        horizon=horizon,
        news_signal=news.model_dump_json(),
        narratives=state.get("news_summary", "(no narratives)"),
        ts_signal=ts.model_dump_json(),
        market_signal=market or "(not available today)",
        latest=state.get("latest_rate", 0.0),
        change=state.get("rate_change_pct", 0.0),
    )

    try:
        llm = get_chat_model("judge", temperature=0.0).with_structured_output(
            AdjudicatorVerdict
        )
        raw = llm.invoke(prompt)
        verdict = (
            raw if isinstance(raw, AdjudicatorVerdict) else AdjudicatorVerdict.model_validate(raw)
        )
        call = DirectionalCall(
            horizon_days=horizon,
            news_signal=news,
            ts_signal=ts,
            **verdict.model_dump(),
        )
    except Exception as exc:
        logger.warning("adjudicate LLM failed (%s) — deterministic fallback", exc)
        call = _fallback_call(news, ts, horizon)

    logger.info(
        "DirectionalCall: %s (confianza=%.2f, reconciliación=%s)",
        call.direction,
        call.confidence,
        call.reconciliation,
    )
    return {"directional_call": call.model_dump()}


# ---------------------------------------------------------------------------
# Node: record_prediction  (Etapa 5 — cerrar el loop predicción → realidad)
# ---------------------------------------------------------------------------

def record_prediction(state: PipelineState) -> PipelineState:
    """Persiste el DirectionalCall del día y evalúa predicciones pendientes.

    La evaluación usa el fx_df fresco de ESTA corrida: cada día nuevo
    trae la TRM que permite calificar las predicciones cuyo horizonte
    ya venció. Así el backtesting real se acumula solo, sin jobs extra.
    """
    call_dict = state.get("directional_call")
    if not call_dict:
        return {}

    try:
        store = PredictionStore()
        call = DirectionalCall.model_validate(call_dict)
        latest = state.get("latest_rate")
        store.save(
            call,
            run_date=state.get("run_date", date.today().isoformat()),
            latest_rate=float(latest) if latest else None,
        )
        fx_df = state.get("fx_df")
        if fx_df is not None and not fx_df.empty:
            store.evaluate_pending(fx_df)
    except Exception as exc:
        logger.error("record_prediction failed: %s", exc)
        return {"errors": [f"record_prediction: {exc}"]}
    return {}


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

    call = state.get("directional_call")
    call_emoji = ""
    verdict_section = ""
    if call:
        labels = {
            "down": "⬇️ USD/COP BAJA (COP se fortalece)",
            "up": "⬆️ USD/COP SUBE (COP se debilita)",
            "neutral": "⏸️ NEUTRAL — el sistema se abstiene",
        }
        call_emoji = {"down": "⬇️", "up": "⬆️", "neutral": "⏸️"}[call["direction"]]
        drivers = "".join(f"\n- {d}" for d in call["news_signal"]["drivers"])
        caveats = "".join(f"\n- {c}" for c in call["caveats"])
        market = state.get("market_signal") or {}
        market_line = ""
        if market:
            market_line = (
                f"Mercado (ayer): bolsa CO {market['equity_ret_1d_pct']:+.2f}% · "
                f"DXY {market['dxy_ret_1d_pct']:+.2f}% · "
                f"Brent {market['brent_ret_1d_pct']:+.2f}% "
                f"→ sesgo {market['direction']}\n\n"
            )
        verdict_section = f"""## Directional Call — {call["horizon_days"]} días
**{labels[call["direction"]]}** · confianza **{call["confidence"]:.2f}** · reconciliación **{call["reconciliation"]}**

Señales: noticias = {call["news_signal"]["direction"]} (score {call["news_signal"]["score"]}) · \
serie = {call["ts_signal"]["direction"]} ({call["ts_signal"]["yhat_delta_pct"]:+.2f}%, \
modelos {"concuerdan" if call["ts_signal"]["models_agree"] else "difieren"})

{market_line}**Racional:** {call["rationale"]}

**Abogado del diablo:** {call["devils_advocate"]}

**Drivers:**{drivers or " (sin drivers de alta severidad)"}

**Caveats:**{caveats or " N/A"}

"""

    trend_emoji = "📈" if change > 0 else "📉"
    report = f"""# COP/USD Intelligence Report — {run_date}

## Current Rate
**1 USD = {latest:,.2f} COP** {trend_emoji} ({change:+.2f}% vs 30 days ago)

{verdict_section}## Market Narrative
{narrative}

## {get_settings().forecast_horizon_days}-Day Forecast
{forecast_table}

## Model Performance (back-test)
{metrics_section or "N/A"}

---
*Generated by cop-fx-intelligence at {datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}*
"""

    # Persist — una ruta relativa en settings se resuelve contra la raíz
    # del proyecto, nunca contra el cwd (cop_fx.paths).
    output_dir = Path(get_settings().report_output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = str(output_dir / f"report_{run_date}.md")
    Path(report_path).write_text(report, encoding="utf-8")

    # Craft tweet (≤280 chars)
    call_line = ""
    if call:
        call_line = (
            f"{call_emoji} Señal {call['horizon_days']}d: {call['direction'].upper()} "
            f"(confianza {call['confidence']:.0%})\n"
        )
    tweet = (
        f"COP/USD — {run_date}\n"
        f"💵 1 USD = {latest:,.0f} COP ({change:+.1f}% 30d)\n"
        f"{call_line}"
        f"{narrative[:100]}...\n"
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
