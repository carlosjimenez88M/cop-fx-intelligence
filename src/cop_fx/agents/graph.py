"""LangGraph StateGraph wiring for the COP/USD intelligence pipeline."""

from __future__ import annotations

from datetime import date

from langgraph.graph import END, START, StateGraph

from cop_fx.agents.nodes import (
    analyze_news,
    check_materiality,
    fetch_fx,
    fetch_news,
    generate_report,
    publish,
    route_materiality,
    run_forecast,
    skip_news,
)
from cop_fx.agents.state import PipelineState
from cop_fx.config.settings import get_settings
from cop_fx.logger import get_logger

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Construct the intelligence pipeline graph.

    ::

        START ─┬─► fetch_fx ──► run_forecast ─────────────────────┐
               └─► fetch_news ─► check_materiality ─(router)─┐    │
                                   │ material                │    │
                                   ▼                         ▼    ▼
                              analyze_news              skip_news │
                                   └───────────► generate_report ◄┘  (defer)
                                                       │
                                                    publish ─► END

    `generate_report` lleva ``defer=True``: las ramas tienen longitudes
    distintas (FX = 2 pasos, noticias = 3), y sin defer el nodo se
    dispararía una vez por rama en supersteps diferentes.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("fetch_fx", fetch_fx)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("check_materiality", check_materiality)
    graph.add_node("skip_news", skip_news)
    graph.add_node("analyze_news", analyze_news)
    graph.add_node("run_forecast", run_forecast)
    graph.add_node("generate_report", generate_report, defer=True)
    graph.add_node("publish", publish)

    # Parallel fetch branches
    graph.add_edge(START, "fetch_fx")
    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_fx", "run_forecast")

    # News branch: materiality gate decides between full analysis and skip
    graph.add_edge("fetch_news", "check_materiality")
    graph.add_conditional_edges(
        "check_materiality",
        route_materiality,
        {"analyze_news": "analyze_news", "skip_news": "skip_news"},
    )

    # Convergence
    graph.add_edge("analyze_news", "generate_report")
    graph.add_edge("skip_news", "generate_report")
    graph.add_edge("run_forecast", "generate_report")
    graph.add_edge("generate_report", "publish")
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
        "errors": [],
        "raw_articles": [],
        "analyzed_articles": [],
    }

    graph = build_graph()
    compiled = graph.compile()
    final_state: PipelineState = compiled.invoke(initial_state)

    if final_state.get("errors"):
        logger.warning("Pipeline completed with errors: %s", final_state["errors"])

    return final_state
