"""LangGraph node functions for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from langgraph.types import Send, interrupt

from cop_fx.agents.memory import MEMORY_KEY, MEMORY_NAMESPACE, build_track_record
from cop_fx.agents.state import PipelineState, TopicWorkerState  # noqa: TC001
from cop_fx.analysis.gold_store import persist_articles, persist_keyword_insights
from cop_fx.analysis.news_analyzer import AnalyzedArticle, NewsAnalyzer
from cop_fx.config.settings import get_settings
from cop_fx.contracts import (
    RELEVANCE_WEIGHT,
    SEVERITY_WEIGHT,
    AdjudicatorVerdict,
    ArticleAnalysis,
    Direction,
    DirectionalCall,
    DominantSignal,
    MarketSignal,
    MaterialityGate,
    NewsSignal,
    TimeSeriesSignal,
    TopStory,
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

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Knobs operativos — viven en config.yaml (raíz), no aquí. Estos alias de
# módulo existen para legibilidad de los nodos y de los tests.
_cfg = get_settings()
MAX_TOPIC_WORKERS = _cfg.max_topic_workers
TS_NEUTRAL_BAND_PCT = _cfg.ts_neutral_band_pct
MARKET_DEAD_BAND_PCT = _cfg.market_dead_band_pct
GATE_HEADLINES_CAP = _cfg.gate_headlines_cap
MIN_ANALYZABLE_CHARS = _cfg.min_analyzable_chars

_COLOMBIA_SCOPE_TERMS = {
    "banrep",
    "bogota",
    "bogotá",
    "colombia",
    "colombian",
    "colombiana",
    "colombiano",
    "cop",
    "dian",
    "ecopetrol",
    "hacienda",
    "minhacienda",
    "petro",
    "peso",
    "trm",
}
_USD_LEG_TERMS = {
    "cpi",
    "dollar index",
    "dxy",
    "fed",
    "federal reserve",
    "fomc",
    "inflacion ee. uu.",
    "inflación ee. uu.",
    "ppi",
    "tariff",
    "treasury",
    "u.s.",
    "us ",
    "usa",
    "wholesale prices",
}
_GLOBAL_TRANSMISSION_TERMS = {
    "brent",
    "capital flows",
    "commodities",
    "commodity",
    "credit rating",
    "emerging markets",
    "energy",
    "global risk",
    "oil",
    "petroleo",
    "petróleo",
    "risk appetite",
    "spread",
}


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
            direction = "down"  # bolsa arriba ⇒ apetito por Colombia ⇒ USD/COP baja
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
            signal.direction,
            equity,
            rets["dxy"],
            rets["brent"],
        )
        return {"market_signal": signal.model_dump()}
    except Exception as exc:
        logger.warning("fetch_market failed (%s) — el contexto es opcional", exc)
        return {"market_signal": {}}


# ---------------------------------------------------------------------------
# Node: load_memory  (Etapa 6 — memoria entre corridas)
# ---------------------------------------------------------------------------


def load_memory(state: PipelineState) -> PipelineState:
    """Hidrata el track-record histórico para que el adjudicador se calibre.

    Rama paralela desde START, SIN arista de salida (igual que fetch_market):
    escribe `prior_performance` en el estado y el `defer` del adjudicador lo
    espera. Lee la fuente durable (`predictions.db`) y, si el grafo se compiló
    con un `BaseStore`, publica el récord en la memoria de largo plazo
    namespaced — la interfaz store.put/get de LangGraph.
    """
    try:
        record = build_track_record(PredictionStore())
    except Exception as exc:
        logger.warning("load_memory: no se pudo leer el historial (%s)", exc)
        return {"prior_performance": {}}

    if record:
        try:
            from langgraph.config import get_store

            store = get_store()
            if store is not None:
                store.put(MEMORY_NAMESPACE, MEMORY_KEY, record)
        except Exception:  # el store es opcional (grafo compilado sin store)
            pass
        logger.info(
            "Memoria: %d llamadas evaluadas, acierto %s",
            record.get("n_decided", 0),
            record.get("hit_rate"),
        )
    return {"prior_performance": record}


# ---------------------------------------------------------------------------
# Node: check_materiality  (Etapa 2 — router / gate de costo)
# ---------------------------------------------------------------------------

_MATERIALITY_PROMPT = """You are the first-line materiality officer for a
Colombian FX desk. You are paid to reject noise before it consumes analyst
time. Be strict, independent, and mechanism-driven.

For EACH headline below, produce a tag with its 0-based index, its topic
(use the full taxonomy — sports and culture have their own categories),
and whether THAT headline could materially move the USD/COP exchange rate.
The asset is USD/COP, not generic global macro. A headline is material only
if it affects Colombia/COP directly, the USD leg through Fed/US macro, oil or
terms of trade, country risk, or capital flows into Colombia/EM assets.

Material means: a reasonable FX desk could update its 1-7 business day
USD/COP view because the headline implies new information about rates,
inflation, fiscal risk, oil/terms of trade, country risk, capital flows,
growth, or the global USD leg.

Think in transmission channels, including second-order ones:
  - monetary/fiscal policy, oil/commodities, political risk, trade, US macro
    are usually material;
  - public health crises transmit via growth and fiscal cost;
  - climate events (drought, El Niño, floods) via food inflation and energy;
  - strikes and social unrest via country risk;
  - foreign macro OUTSIDE the US (Europe, UK, Asia) matters only if it moves
    global risk appetite, oil or the dollar index — on its own it is usually
    NOT material for USD/COP;
  - English-language sources are acceptable only when the Colombia/COP or
    USD-leg transmission is explicit enough to explain in Spanish;
  - sports, entertainment and human-interest stories are NOT material.

Reject:
  - stale follow-ups with no new fact;
  - generic business optimism with no FX channel;
  - purely local crime/weather unless it can affect inflation, exports,
    energy, fiscal cost, or country risk;
  - foreign macro outside the US when it is not tied to DXY, oil or risk appetite.

Then set `has_material_news` = true if at least one headline is material,
and explain the overall verdict in `reason`. If evidence is borderline,
prefer `material=false` for that headline and mention the uncertainty.

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
        "source": item.article.source,
        "author": getattr(item.article, "author", ""),
        "summary": getattr(item.article, "summary", ""),
        "url": item.article.url,
        "published_at": getattr(item.article, "published_at", None),
        "topic": item.topic,
        "keywords": item.keywords,
        "entities": item.entities,
        "fx_relevance": item.fx_relevance,
        "fx_channel": item.fx_channel,
        "severity": item.severity,
        "bullish_cop": item.bullish_cop,
        "reasoning": item.reasoning,
    }


def _has_colombia_or_us_cop_scope(article: dict[str, Any]) -> bool:
    """Keep only news with Colombia scope or a defensible USD/COP transmission."""
    if article.get("fx_relevance") == "none" or article.get("fx_channel") == "none":
        return False

    text = " ".join(
        str(part or "")
        for part in (
            article.get("title"),
            article.get("source"),
            article.get("summary"),
            article.get("topic"),
            article.get("fx_channel"),
            article.get("reasoning"),
            " ".join(str(item) for item in article.get("keywords") or []),
            " ".join(str(item) for item in article.get("entities") or []),
        )
    ).lower()

    if any(term in text for term in _COLOMBIA_SCOPE_TERMS):
        return True
    if article.get("topic") == "us_global_macro" and any(term in text for term in _USD_LEG_TERMS):
        return True
    if article.get("fx_channel") in {"terms_of_trade", "capital_flows", "country_risk"}:
        return any(term in text for term in _GLOBAL_TRANSMISSION_TERMS)
    return False


def _apply_scope_guard(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade material articles that are global noise for a Colombia FX desk."""
    guarded: list[dict[str, Any]] = []
    downgraded = 0
    for article in articles:
        item = dict(article)
        if item.get("fx_relevance") != "none" and not _has_colombia_or_us_cop_scope(item):
            downgraded += 1
            item["fx_relevance"] = "none"
            item["fx_channel"] = "none"
            item["severity"] = "low"
            item["bullish_cop"] = False
            item["reasoning"] = (
                "Descartada: no muestra canal claro hacia Colombia, COP, USD global "
                "o términos de intercambio."
            )
        guarded.append(item)
    if downgraded:
        logger.info("Scope guard: %d artículos globales degradados a no materiales", downgraded)
    return guarded


def orchestrate(state: PipelineState) -> PipelineState:
    """Agrupa artículos en clusters por tópico usando los tags del gate.

    Determinista, cero LLM: el gate ya pagó la etiqueta gruesa. Los tags
    no-materiales se descartan; los titulares sin tag van a "other".
    Si hay más de MAX_TOPIC_WORKERS clusters, los más pequeños se funden
    en "other" para acotar el costo del fan-out.
    """
    # 🎓 PROYECTO — PISTA 3 (orchestrator-workers): hoy este orquestador solo
    # PARTE los artículos en clusters y todos los workers usan el MISMO prompt.
    # El reto es que el orquestador PLANEE: asignar una persona/prompt por tipo
    # de cluster (banca central, commodities, riesgo-país) en el payload del
    # Send, o lanzar un debate alcista vs bajista. Ver docs/proyecto_individual.md §5.
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
    # 🎓 PROYECTO — PISTA 1 (ReAct): hoy el analista clasifica de UN SOLO disparo
    # y ADIVINA la severidad. El reto es darle herramientas deterministas
    # (get_series sobre Brent/DXY/equity en market_fetcher.py, get_trm en
    # fx_fetcher.py) y dejar que razone+actúe para ANCLAR la severidad en datos
    # reales. La tool informa; la serie sigue dando el signo. Ver §5 Pista 1.
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
        a for a in articles if len(getattr(a, "body", "") or a.summary) >= MIN_ANALYZABLE_CHARS
    ]
    if len(readable) < len(articles):
        logger.info(
            "topic_worker[%s]: %d artículos sin texto analizable descartados",
            topic,
            len(articles) - len(readable),
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
    analyzed = _apply_scope_guard(state.get("worker_analyses", []))

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
    if state.get("persist_gold", False):
        try:
            persist_articles(analyzed)
            persist_keyword_insights(analyzed)
        except Exception as exc:
            logger.warning("No se pudo persistir la capa GOLD: %s", exc)
    return {
        "analyzed_articles": analyzed,
        "news_signal": signal.model_dump(),
        "news_summary": summary,
    }


# ---------------------------------------------------------------------------
# Node: pick_top_story  (el agente editor — ¿cuál es LA noticia del día?)
# ---------------------------------------------------------------------------

_TOP_STORY_PROMPT = """You are the front-page editor AND risk manager of a
Colombian FX desk. Your job is to choose the one story that would most
deserve a trader's attention before deciding USD/COP exposure.

From the analyzed candidates below, choose THE single most important story
of the day for the USD/COP direction over the next 1-7 business days.
Importance = horizon relevance x transmission channel x novelty x evidence
quality. A structural story only wins if it can plausibly reprice spot USD/COP
during this horizon.

Decision criteria, in order:
1. Transmission clarity: the story must have a named FX channel.
2. Horizon relevance: prefer stories that can move spot in 1-7 business days.
   Penalize announcements whose main effect is years away unless markets are
   likely to reprice credibility, TES, CDS, fiscal risk or expectations now.
3. Surprise/novelty: new information beats repetition of an already-known theme.
4. Persistence: a durable repricing driver beats intraday noise.
5. Scope: macro/fiscal/monetary/oil/global USD beats narrow sector color.
6. Evidence quality: prefer specific facts over vague commentary. If an article
   likely came from a short paywalled summary, be conservative.

If the highest-severity item is stale, routine, or duplicated, choose the
cleaner fresher driver instead. Do not choose a story merely because it is
dramatic; choose it because it can reprice USD/COP.

Write `spanish_title` as a concise Spanish desk headline faithful to the
chosen original title. Keep proper nouns, acronyms and tickers unchanged.
Write `why_it_matters` and `watch_next` in SPANISH only. Do not use English or
other languages except proper nouns, tickers and institution names.

CANDIDATES:
{candidates}
"""


def pick_top_story(state: PipelineState) -> PipelineState:
    """Agente editor: elige la noticia MAS importante del dia.

    Preseleccion determinista (severidad x relevancia, top 10) para no
    gastar tokens en ruido; el LLM solo elige y justifica; el sistema
    compone el registro con los datos reales del artículo elegido.
    Fallback determinista: el candidato de mayor peso.
    """
    analyzed = state.get("worker_analyses", [])
    candidates = [d for d in analyzed if d.get("fx_relevance") != "none"]
    candidates.sort(
        key=lambda d: (
            SEVERITY_WEIGHT.get(str(d.get("severity", "low")), 0)
            * RELEVANCE_WEIGHT.get(str(d.get("fx_relevance", "none")), 0)
        ),
        reverse=True,
    )
    candidates = candidates[:10]
    if not candidates:
        return {"top_story": {}}

    digest = "\n".join(
        f"[{i}] ({d.get('source', '?')} · {d['topic']} · {d['severity']}/{d['fx_relevance']}"
        f" · canal {d['fx_channel']}) {d['title']} — {d['reasoning']}"
        for i, d in enumerate(candidates)
    )

    def _compose(
        chosen: dict[str, Any],
        spanish_title: str,
        why: str,
        watch: str,
    ) -> dict[str, object]:
        return {
            "title": chosen["title"],
            "display_title": spanish_title,
            "source": chosen.get("source", ""),
            "url": chosen.get("url", ""),
            "topic": chosen["topic"],
            "fx_channel": chosen["fx_channel"],
            "severity": chosen["severity"],
            "why_it_matters": why,
            "watch_next": watch,
        }

    try:
        llm = get_chat_model("fast", temperature=0.0).with_structured_output(TopStory)
        raw = llm.invoke(_TOP_STORY_PROMPT.format(candidates=digest))
        ts = raw if isinstance(raw, TopStory) else TopStory.model_validate(raw)
        chosen = candidates[min(ts.chosen_index, len(candidates) - 1)]
        top = _compose(chosen, ts.spanish_title, ts.why_it_matters, ts.watch_next)
    except Exception as exc:
        logger.warning("pick_top_story LLM failed (%s) — fallback determinista", exc)
        chosen = candidates[0]
        top = _compose(
            chosen,
            str(chosen["title"]),
            f"Mayor peso severidad x relevancia del dia: {chosen['reasoning']}",
            "Seguimiento del tema en la próxima corrida.",
        )

    logger.info("Top story: %s (%s)", top["title"], top["source"])
    return {"top_story": top}


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
        a for a in pool if len(getattr(a, "body", "") or a.summary) >= MIN_ANALYZABLE_CHARS
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
            r = forecaster_cls().fit_predict(train_df, horizon_days=horizon)
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

    direction: Direction
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
        direction=direction,
        yhat_delta_pct=round(delta_pct, 3),
        models_agree=models_agree,
    )


# ---------------------------------------------------------------------------
# Node: adjudicate  (Etapa 4 — la capa de racionalidad)
# ---------------------------------------------------------------------------

_ADJUDICATOR_PROMPT = """You are the independent chair of a Colombian FX
investment committee. You are not a news summarizer and not a chart follower:
your job is to reconcile independent evidence and decide whether the desk
should call USD/COP up, down, or abstain for the next {horizon} days.

You must be adversarial toward every signal. News can overreact to headlines;
time-series models can extrapolate stale trends; market context can be a one-day
false signal. The professional answer is often neutral.

NEWS SIGNAL (aggregated from today's analyzed articles, weighted by
severity x FX-relevance; score > 0 means COP strengthens => USD/COP DOWN):
{news_signal}

Cluster narratives:
{narratives}

TIME-SERIES SIGNAL (Prophet+ARIMA ensemble, deterministic sign):
{ts_signal}

TOP STORY OF THE DAY (chosen by the desk-editor agent — give it extra
weight when judging which signal should dominate):
{top_story}

MARKET CONTEXT (deterministic, yesterday's moves). The Colombian equity
aggregate is the ONLY empirically validated leading indicator of USD/COP
(equity up yesterday => COP tends to strengthen => USD/COP DOWN); DXY and
Brent are background, not votes. Use this as evidence to break ties or
temper confidence — do not treat it as a third signal to echo:
{market_signal}

RECENT TRACK RECORD (your own past calls, already scored against the realized
TRM — this is the desk's memory across runs). Use it to calibrate, not to
predict: if recent decisive calls were wrong, or high-confidence calls failed,
demand stronger fresh evidence today and lean toward lower confidence or
neutral. Do NOT mechanically repeat or invert yesterday's direction:
{track_record}

Current rate: 1 USD = {latest:,.2f} COP ({change:+.2f}% vs 30 days ago).

Rules of reasoning — in this order:
1. First classify the relationship between news_signal and ts_signal:
   agree, diverge, or partial. Do this before choosing a direction.
2. Decide which signal dominates:
   - news dominates only if the top story or aggregate news is fresh, specific,
     high/medium relevance, and plausibly reprices USD/COP within the horizon.
   - timeseries dominates only if the forecast sign is meaningful and not
     contradicted by a stronger fresh shock.
   - market dominates only as a tie-breaker or when news/TS are weak; do not
     invent a third vote from DXY/Brent.
   - none dominates when evidence is stale, mixed, weak, or contradictory.
3. Direction MUST match dominant_signal. If dominant_signal=news, direction
   must equal news_signal.direction. If timeseries, direction must equal
   ts_signal.direction. If market, direction must equal market_context.direction.
   If none, direction must be neutral. The code will correct contradictions.
4. devils_advocate: build the strongest case against your chosen direction.
   If that argument is not clearly weaker, lower confidence or abstain.
5. Calibrate hard:
   - divergence caps confidence at 0.5 (enforced by code);
   - partial evidence should rarely exceed 0.6;
   - confidence below 0.35 turns the call neutral (enforced);
   - if Prophet and ARIMA disagree, penalize any timeseries-dominant verdict;
   - neutral is a valid, professional answer.
6. consistency_notes must explicitly state:
   - the dominant signal selected;
   - why the losing signal did not dominate;
   - whether direction matches the dominant signal;
   - any reason confidence was capped.
7. rationale must cite specific drivers/headlines and the forecast sign. Do not
   make claims that are not in the provided signals.
8. Write rationale, devils_advocate, consistency_notes and caveats in SPANISH
   only. Do not use English or other languages except proper nouns, tickers and
   institution names.
"""


def _fallback_call(news: NewsSignal, ts: TimeSeriesSignal, horizon: int) -> DirectionalCall:
    """Reconciliación determinista cuando el adjudicador LLM no está disponible."""
    if news.direction == ts.direction:
        reconciliation, direction = "agree", news.direction
        confidence = 0.6 if news.direction != "neutral" else 0.4
        dominant_signal = "news" if news.direction != "neutral" else "none"
    elif "neutral" in (news.direction, ts.direction):
        reconciliation = "partial"
        direction = news.direction if ts.direction == "neutral" else ts.direction
        confidence = 0.45
        dominant_signal = "news" if ts.direction == "neutral" else "timeseries"
    else:
        reconciliation, direction, confidence = "diverge", "neutral", 0.3
        dominant_signal = "none"

    return DirectionalCall(
        direction=direction,
        confidence=confidence,
        horizon_days=horizon,
        news_signal=news,
        ts_signal=ts,
        reconciliation=reconciliation,  # type: ignore[arg-type]
        dominant_signal=dominant_signal,  # type: ignore[arg-type]
        consistency_notes=["Fallback determinista sin adjudicador LLM."],
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


def _market_direction(raw_market: dict[str, object] | None) -> Direction:
    if not raw_market:
        return "neutral"
    value = raw_market.get("direction", "neutral")
    if value in {"down", "up", "neutral"}:
        return cast("Direction", value)
    return "neutral"


def _expected_reconciliation(
    news: NewsSignal, ts: TimeSeriesSignal
) -> Literal["agree", "diverge", "partial"]:
    if news.direction == ts.direction:
        return "agree"
    if "neutral" in (news.direction, ts.direction):
        return "partial"
    return "diverge"


def _direction_for_dominant_signal(
    dominant_signal: DominantSignal,
    news: NewsSignal,
    ts: TimeSeriesSignal,
    market_direction: Direction,
) -> Direction:
    if dominant_signal == "news":
        return news.direction
    if dominant_signal == "timeseries":
        return ts.direction
    if dominant_signal == "market":
        return market_direction
    return "neutral"


def _coherent_directional_call(
    *,
    verdict: AdjudicatorVerdict,
    news: NewsSignal,
    ts: TimeSeriesSignal,
    market: MarketSignal | None,
    horizon: int,
) -> DirectionalCall:
    """Build a DirectionalCall and fix structural contradictions deterministically."""
    expected_reconciliation = _expected_reconciliation(news, ts)
    market_direction = market.direction if market is not None else "neutral"
    dominant_signal = verdict.dominant_signal
    notes = list(verdict.consistency_notes)

    if expected_reconciliation != verdict.reconciliation:
        notes.append(
            "Reconciliacion corregida por codigo: "
            f"{verdict.reconciliation} -> {expected_reconciliation}."
        )

    expected_direction = _direction_for_dominant_signal(dominant_signal, news, ts, market_direction)
    direction = verdict.direction
    confidence = verdict.confidence

    if expected_direction == "neutral":
        if direction != "neutral":
            notes.append("Direccion corregida a neutral porque la senal dominante no decide.")
        direction = "neutral"
        confidence = min(confidence, 0.45)
        dominant_signal = "none"
    elif direction != expected_direction:
        notes.append(
            "Direccion corregida por coherencia con la senal dominante "
            f"{dominant_signal}: {direction} -> {expected_direction}."
        )
        direction = expected_direction
        confidence = min(confidence, 0.5)

    if expected_reconciliation == "agree" and direction != news.direction:
        notes.append("Direccion corregida porque noticias y serie concuerdan.")
        direction = news.direction
        dominant_signal = "news" if news.direction != "neutral" else "none"

    if (
        expected_reconciliation == "diverge"
        and dominant_signal == "timeseries"
        and not ts.models_agree
    ):
        notes.append("Confianza limitada: la serie domina aunque Prophet y ARIMA difieren.")
        confidence = min(confidence, 0.45)

    return DirectionalCall(
        direction=direction,
        confidence=confidence,
        horizon_days=horizon,
        news_signal=news,
        ts_signal=ts,
        market_signal=market,
        reconciliation=expected_reconciliation,
        dominant_signal=dominant_signal,
        consistency_notes=notes,
        rationale=verdict.rationale,
        devils_advocate=verdict.devils_advocate,
        caveats=verdict.caveats,
    )


def adjudicate(state: PipelineState) -> PipelineState:
    """Cruza la señal de noticias contra la de la serie → DirectionalCall.

    El LLM (tier 'judge') produce solo el veredicto (AdjudicatorVerdict);
    las señales y el horizonte los inyecta el sistema al componer el
    DirectionalCall, cuyos validadores acotan la confianza y fuerzan la
    abstención. El LLM juzga; el código gobierna.
    """
    # 🎓 PROYECTO — PISTA 2 (evaluator-optimizer): hoy esto es UN SOLO disparo y
    # las contradicciones las corrige el código en _build_directional_call. El
    # reto es cerrar el lazo: un nodo crítico que puntúe el veredicto contra una
    # rúbrica y, si reprueba, regenere (con revision_count + recursion_limit).
    # Solución de referencia trabajada en examples/evaluator_optimizer_reference.py.
    # Ver docs/proyecto_individual.md §5 Pista 2.
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
    market_model = MarketSignal.model_validate(market) if market else None
    top_story = state.get("top_story") or {}
    prior = state.get("prior_performance") or {}
    prompt = _ADJUDICATOR_PROMPT.format(
        horizon=horizon,
        news_signal=news.model_dump_json(),
        narratives=state.get("news_summary", "(no narratives)"),
        ts_signal=ts.model_dump_json(),
        top_story=top_story or "(no top story today)",
        market_signal=market or "(not available today)",
        track_record=prior.get("digest") or "(sin historial previo — primera(s) corrida(s))",
        latest=state.get("latest_rate", 0.0),
        change=state.get("rate_change_pct", 0.0),
    )

    try:
        llm = get_chat_model("judge", temperature=0.0).with_structured_output(AdjudicatorVerdict)
        raw = llm.invoke(prompt)
        verdict = (
            raw if isinstance(raw, AdjudicatorVerdict) else AdjudicatorVerdict.model_validate(raw)
        )
        call = _coherent_directional_call(
            verdict=verdict,
            news=news,
            ts=ts,
            market=market_model,
            horizon=horizon,
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
            top_story=state.get("top_story") or None,
            market_signal=state.get("market_signal") or None,
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
                f"| {row['ds'].strftime('%Y-%m-%d')} | {row['yhat']:.2f} "
                f"| {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f} |"
            )
        forecast_table = "| Date | Forecast | 95% CI |\n|------|----------|--------|\n" + "\n".join(
            rows
        )

    metrics_section = ""
    for m in state.get("eval_metrics", []):
        metrics_section += (
            f"- **{m.model_name.upper()}**: MAE={m.mae:.2f}, "
            f"RMSE={m.rmse:.2f}, MAPE={m.mape:.2f}%\n"
        )

    call = state.get("directional_call")
    verdict_section = ""
    if call:
        labels = {
            "down": "⬇️ USD/COP BAJA (COP se fortalece)",
            "up": "⬆️ USD/COP SUBE (COP se debilita)",
            "neutral": "⏸️ NEUTRAL — el sistema se abstiene",
        }
        drivers = "".join(f"\n- {d}" for d in call["news_signal"]["drivers"])
        caveats = "".join(f"\n- {c}" for c in call["caveats"])
        market = state.get("market_signal") or {}
        dominant = call.get("dominant_signal", "none")
        consistency_notes = "".join(f"\n- {note}" for note in call.get("consistency_notes", []))
        market_line = ""
        if market:
            market_line = (
                f"Mercado (ayer): bolsa CO {market['equity_ret_1d_pct']:+.2f}% · "
                f"DXY {market['dxy_ret_1d_pct']:+.2f}% · "
                f"Brent {market['brent_ret_1d_pct']:+.2f}% "
                f"→ sesgo {market['direction']}\n\n"
            )
        top = state.get("top_story") or {}
        top_story_section = ""
        if top:
            display_title = top.get("display_title") or top["title"]
            original_note = (
                f"\n\n*Titular original:* {top['title']}"
                if display_title != top["title"]
                else ""
            )
            top_story_section = (
                f"**📌 Noticia del día:** {display_title} ({top['source']})"
                f"{original_note}\n\n"
                f"*Por qué importa:* {top['why_it_matters']}\n\n"
                f"*Vigilar:* {top['watch_next']}\n\n"
            )
        verdict_section = f"""## Directional Call — {call["horizon_days"]} días
**{labels[call["direction"]]}** · confianza **{call["confidence"]:.2f}** · \
reconciliación **{call["reconciliation"]}** · domina **{dominant}**

Señales: noticias = {call["news_signal"]["direction"]} (score {call["news_signal"]["score"]}) · \
serie = {call["ts_signal"]["direction"]} ({call["ts_signal"]["yhat_delta_pct"]:+.2f}%, \
modelos {"concuerdan" if call["ts_signal"]["models_agree"] else "difieren"})

{market_line}{top_story_section}**Racional:** {call["rationale"]}

**Abogado del diablo:** {call["devils_advocate"]}

**Checks de coherencia:**{consistency_notes or " OK"}

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

    return {"report_markdown": report, "report_path": report_path}


# ---------------------------------------------------------------------------
# Node: human_review  (Etapa 6 — HITL con interrupt para revisar el veredicto)
# ---------------------------------------------------------------------------


def human_review(state: PipelineState) -> PipelineState:
    """Punto de control humano antes de finalizar la corrida (human-in-the-loop).

    Solo actúa con `hitl_enabled=True` (que exige un checkpointer). Llama a
    `interrupt(...)` con el veredicto del día: el grafo SE PAUSA y el payload
    vuelve al llamador (CLI/dashboard). La corrida se reanuda con
    `Command(resume={"approved": bool})` y ese valor es lo que devuelve
    `interrupt`. Sin HITL es un passthrough — el comportamiento por defecto del
    pipeline no cambia.
    """
    if not state.get("hitl_enabled", False):
        return {}

    call = state.get("directional_call") or {}
    decision = interrupt(
        {
            "type": "verdict_review",
            "direction": call.get("direction"),
            "confidence": call.get("confidence"),
            "report_path": state.get("report_path", ""),
            "question": "¿Aceptar este veredicto? Reanuda con {'approved': true|false}.",
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    logger.info("human_review: %s", "aprobado" if approved else "rechazado")
    return {"review_approved": approved}
