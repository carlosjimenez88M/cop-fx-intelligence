"""LangGraph StateGraph wiring for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from cop_fx.agents.nodes import (
    adjudicate,
    aggregate_signals,
    check_materiality,
    fan_out_clusters,
    fetch_fx,
    fetch_market,
    fetch_news,
    generate_report,
    human_review,
    load_memory,
    orchestrate,
    pick_top_story,
    record_prediction,
    route_materiality,
    run_forecast,
    skip_news,
    topic_worker,
)
from cop_fx.agents.persistence import get_checkpointer, get_store
from cop_fx.agents.state import PipelineState
from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore

logger = get_logger(__name__)


def build_graph() -> StateGraph[PipelineState]:
    """Construct the intelligence pipeline graph.

    ::

        START ─┬─► fetch_fx ──► run_forecast (+ ts_signal) ─────────────┐
               ├─► fetch_market (contexto, sin arista de salida)        │
               ├─► load_memory (track-record, sin arista de salida)     │
               └─► fetch_news ─► check_materiality ─(router)─┐          │
                                   │ material                ▼          │
                                   ▼                     skip_news      │
                              orchestrate                    │          │
                                   │ Send x N clusters       │          │
                                   ▼                         │          │
                          topic_worker (dinámicos)           │          │
                                   ▼                         ▼          ▼
                            aggregate_signals ─────────► adjudicate (defer)
                                   ▼ pick_top_story            │
                          generate_report ─► record_prediction │
                                   └─► human_review (HITL) ─► END

    Mecanismos de LangGraph trabajando juntos:
      - `Send` (orchestrator-workers): la cantidad de workers varía cada día
        con los clusters de noticias — un grafo estático no puede.
      - `defer=True` en adjudicate + UN SOLO trigger entrante (el final de la
        rama de noticias). fetch_market y load_memory son ramas paralelas SIN
        arista hacia adjudicate a propósito: escriben en el estado (market_signal,
        prior_performance) y el defer las espera, sin sumar un trigger extra que
        dispararía el nodo deferred más de una vez.
      - `interrupt` (HITL): el nodo human_review pausa antes de finalizar cuando
        hitl_enabled; se reanuda con Command(resume=...) sobre el mismo thread_id.
      - Checkpointer + Store: ver `compile_graph`.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("fetch_fx", fetch_fx)
    graph.add_node("fetch_market", fetch_market)
    graph.add_node("load_memory", load_memory)
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
    graph.add_node("human_review", human_review)

    # Ramas paralelas desde START. fetch_market y load_memory no tienen arista
    # de salida a propósito (escriben estado que el defer del adjudicador espera).
    graph.add_edge(START, "fetch_fx")
    graph.add_edge(START, "fetch_market")
    graph.add_edge(START, "load_memory")
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

    # Etapa 4: convergencia en el adjudicador → reporte → predicción → HITL → publicación
    # (sin arista run_forecast→adjudicate: ver nota sobre defer en el docstring)
    graph.add_edge("pick_top_story", "adjudicate")
    graph.add_edge("skip_news", "adjudicate")
    graph.add_edge("adjudicate", "generate_report")
    graph.add_edge("generate_report", "record_prediction")
    graph.add_edge("record_prediction", "human_review")
    graph.add_edge("human_review", END)

    return graph


def compile_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph[PipelineState]:
    """Compila el grafo con las dos memorias de LangGraph (Etapa 6).

      - `checkpointer`: memoria por hilo (`thread_id`). Obligatoria para `interrupt`
        y para reanudar en otro proceso. Si es None, el grafo corre sin estado
        persistente (comportamiento clásico del pipeline batch).
      - `store`: memoria de largo plazo entre hilos. El nodo `load_memory` publica
        ahí el track-record; por defecto se inyecta un `InMemoryStore` para que la
        interfaz exista en cada corrida (la fuente durable es `predictions.db`).
    """
    if store is None:
        store = get_store()
    return build_graph().compile(checkpointer=checkpointer, store=store)


def run_pipeline(
    *,
    run_date: str | None = None,
    hitl: bool = False,
    thread_id: str | None = None,
    persistent_checkpoint: bool = True,
) -> PipelineState:
    """Execute the full pipeline and return the final state.

    Con `hitl=True` el grafo se compila con un checkpointer (durable por defecto)
    y se invoca bajo un `thread_id`; si `human_review` lanza `interrupt`, la
    corrida se PAUSA y el estado devuelto trae `__interrupt__` — reanúdala con
    `resume_pipeline(thread_id=..., approved=...)`.
    """
    settings = get_settings()
    run_date = run_date or date.today().isoformat()
    initial_state: PipelineState = {
        "run_date": run_date,
        "horizon_days": settings.forecast_horizon_days,
        "persist_gold": True,
        "hitl_enabled": hitl,
        "errors": [],
        "raw_articles": [],
        "analyzed_articles": [],
    }

    checkpointer: BaseCheckpointSaver[Any] | None = None
    config: RunnableConfig | None = None
    if hitl:
        checkpointer = get_checkpointer(persistent=persistent_checkpoint)
        config = {"configurable": {"thread_id": thread_id or run_date}}

    compiled = compile_graph(checkpointer=checkpointer)
    result = cast("dict[str, Any]", compiled.invoke(initial_state, config=config))

    if result.get("__interrupt__"):
        logger.info("Pipeline paused at human_review (HITL) — awaiting approval")
        return cast("PipelineState", result)

    final_state = cast("PipelineState", result)
    if final_state.get("errors"):
        logger.warning("Pipeline completed with errors: %s", final_state["errors"])
    return final_state


def resume_pipeline(
    *,
    thread_id: str,
    approved: bool,
    persistent_checkpoint: bool = True,
) -> PipelineState:
    """Reanuda una corrida pausada en `human_review` con la decisión humana.

    Reconstruye el checkpointer durable, recupera el estado del `thread_id` y
    entrega `Command(resume={"approved": ...})` a `interrupt`.
    """
    checkpointer = get_checkpointer(persistent=persistent_checkpoint)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    resume_value: dict[str, Any] = {"approved": approved}

    compiled = compile_graph(checkpointer=checkpointer)
    command: Command[Any] = Command(resume=resume_value)
    final_state = cast("PipelineState", compiled.invoke(command, config=config))
    if final_state.get("errors"):
        logger.warning("Resumed pipeline completed with errors: %s", final_state["errors"])
    return final_state
