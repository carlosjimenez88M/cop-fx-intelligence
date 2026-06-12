"""LangGraph StateGraph wiring for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import date
from typing import cast

from langgraph.graph import END, START, StateGraph

from cop_fx.agents.nodes import (
    adjudicate,
    aggregate_signals,
    check_materiality,
    fan_out_clusters,
    fetch_fx,
    fetch_market,
    fetch_news,
    generate_report,
    orchestrate,
    pick_top_story,
    publish,
    record_prediction,
    route_materiality,
    run_forecast,
    skip_news,
    topic_worker,
)
from cop_fx.agents.state import PipelineState
from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph[PipelineState]:
    """Construct the intelligence pipeline graph.

    ::

        START ─┬─► fetch_fx ──► run_forecast (+ ts_signal) ─────────────┐
               └─► fetch_news ─► check_materiality ─(router)─┐          │
                                   │ material                ▼          │
                                   ▼                     skip_news      │
                              orchestrate                    │          │
                                   │ Send x N clusters       │          │
                                   ▼                         │          │
                          topic_worker (dinámicos)           │          │
                                   ▼                         ▼          ▼
                            aggregate_signals ─────────► adjudicate (defer)
                                                             │
                                              generate_report ─► publish ─► END

    Mecanismos de LangGraph trabajando juntos:
      - `Send` (orchestrator-workers): la cantidad de workers varía cada
        día con los clusters de noticias — un grafo estático no puede.
        Todos los Sends corren en el MISMO superstep; aggregate_signals
        se dispara una sola vez cuando todos terminan.
      - `defer=True` en adjudicate + UN SOLO trigger entrante (el final de
        la rama de noticias). run_forecast NO tiene arista hacia adjudicate
        a propósito: un trigger extra hace que el nodo deferred se ejecute
        una vez por trigger (verificado empíricamente — defer no retiene
        cuando lo único pendiente son tasks de Send). Con un solo trigger,
        defer se limita a esperar a que la rama FX termine de escribir
        ts_signal en el estado antes de emitir el veredicto.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("fetch_fx", fetch_fx)
    graph.add_node("fetch_market", fetch_market)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("check_materiality", check_materiality)
    graph.add_node("skip_news", skip_news)
    graph.add_node("orchestrate", orchestrate)
    graph.add_node("topic_worker", topic_worker)
    graph.add_node("aggregate_signals", aggregate_signals)
    graph.add_node("pick_top_story", pick_top_story)
    graph.add_node("run_forecast", run_forecast)
    graph.add_node("adjudicate", adjudicate, defer=True)
    graph.add_node("generate_report", generate_report)
    graph.add_node("record_prediction", record_prediction)
    graph.add_node("publish", publish)

    # Parallel fetch branches (fetch_market no tiene arista de salida a
    # propósito: escribe market_signal y el defer del adjudicador lo espera)
    graph.add_edge(START, "fetch_fx")
    graph.add_edge(START, "fetch_market")
    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_fx", "run_forecast")

    # News branch: materiality gate decides between fan-out and skip
    graph.add_edge("fetch_news", "check_materiality")
    graph.add_conditional_edges(
        "check_materiality",
        route_materiality,
        {"orchestrate": "orchestrate", "skip_news": "skip_news"},
    )

    # Etapa 3: fan-out dinámico — un worker por cluster de tópico
    graph.add_conditional_edges("orchestrate", fan_out_clusters, ["topic_worker"])
    graph.add_edge("topic_worker", "aggregate_signals")

    # El agente editor elige LA noticia del día con los análisis ya hechos
    graph.add_edge("aggregate_signals", "pick_top_story")

    # Etapa 4: convergencia en el adjudicador → reporte → publicación
    # (sin arista run_forecast→adjudicate: ver nota sobre defer en el docstring)
    graph.add_edge("pick_top_story", "adjudicate")
    graph.add_edge("skip_news", "adjudicate")
    graph.add_edge("adjudicate", "generate_report")
    graph.add_edge("generate_report", "record_prediction")
    graph.add_edge("record_prediction", "publish")
    graph.add_edge("publish", END)

    return graph


def run_pipeline(
    *,
    run_date: str | None = None,
    publish_enabled: bool = False,
) -> PipelineState:
    """Execute the full pipeline and return the final state."""
    settings = get_settings()
    initial_state: PipelineState = {
        "run_date": run_date or date.today().isoformat(),
        "horizon_days": settings.forecast_horizon_days,
        "publish_enabled": publish_enabled,
        "persist_gold": True,
        "errors": [],
        "raw_articles": [],
        "analyzed_articles": [],
    }

    graph = build_graph()
    compiled = graph.compile()
    final_state = cast("PipelineState", compiled.invoke(initial_state))

    if final_state.get("errors"):
        logger.warning("Pipeline completed with errors: %s", final_state["errors"])

    return final_state
