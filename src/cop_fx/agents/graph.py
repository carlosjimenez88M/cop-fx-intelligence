"""LangGraph StateGraph wiring for the COP/USD intelligence pipeline."""

from __future__ import annotations

from cop_fx.logger import get_logger
from datetime import date

from langgraph.graph import END, START, StateGraph

from cop_fx.agents.nodes import (
    analyze_news,
    fetch_fx,
    fetch_news,
    generate_report,
    publish,
    run_forecast,
)
from cop_fx.agents.state import PipelineState
from cop_fx.config.settings import get_settings

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """Construct and compile the intelligence pipeline graph."""
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("fetch_fx", fetch_fx)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("analyze_news", analyze_news)
    graph.add_node("run_forecast", run_forecast)
    graph.add_node("generate_report", generate_report)
    graph.add_node("publish", publish)

    # Edges: fetch fx + news in parallel (both start from START),
    # then converge at analyze_news → forecast → report → publish
    graph.add_edge(START, "fetch_fx")
    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_news", "analyze_news")
    graph.add_edge("fetch_fx", "run_forecast")
    graph.add_edge("analyze_news", "generate_report")
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
