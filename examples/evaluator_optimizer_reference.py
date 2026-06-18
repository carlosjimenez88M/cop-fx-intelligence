"""Solución de referencia — PISTA 2 del proyecto (evaluator-optimizer).

⚠️  Esto es un EJEMPLO TRABAJADO de "cómo debería quedar" el patrón. NO es parte
del pipeline de producción (`cop_fx.agents`). Su único objetivo es mostrar, de
forma autocontenida y ejecutable sin API keys, el lazo completo del patrón
evaluator-optimizer para que lo adaptes al nodo real `adjudicate`.

El patrón (Anthropic, *Building Effective Agents*): un **generador** propone una
respuesta y un **evaluador** la juzga contra una rúbrica explícita; si no pasa, el
evaluador devuelve feedback accionable y el generador REGENERA. Se itera hasta
pasar o hasta agotar un presupuesto de revisiones.

Lo que demuestra y que el `adjudicate` actual NO hace:
  - cierra el lazo en el GRAFO (arista condicional de vuelta), no en el código;
  - lleva `revision_count` en el estado + tope `MAX_REVISIONS` para no hacer
    bucle infinito (equivalente a fijar `recursion_limit`);
  - cuando se agotan las revisiones, **se abstiene** (decisión segura) en vez de
    publicar un veredicto incoherente.

Cómo correr el demo:

    uv run python -m examples.evaluator_optimizer_reference

Cómo lo adaptarías al proyecto: reemplaza `critique` determinista por un segundo
LLM/tier con una rúbrica en el prompt, y monta `generate → critique → (vuelve|END)`
dentro del grafo real respetando la trampa del `defer`/trigger único (ver
docs/proyecto_individual.md §4.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.graph.state import CompiledStateGraph

MAX_REVISIONS = 2


class Verdict(TypedDict):
    """Veredicto simplificado (espejo reducido de `DirectionalCall`)."""

    direction: str  # "up" | "down" | "neutral"
    confidence: float
    dominant_signal: str  # "news" | "timeseries" | "market" | "none"
    rationale: str


class LoopState(TypedDict, total=False):
    evidence: str
    verdict: Verdict
    critique: str
    revision_count: int
    approved: bool


class VerdictGenerator(Protocol):
    """Contrato del generador: dado el contexto y el feedback previo, propone un
    veredicto. En el proyecto, esto envuelve la llamada al LLM (tier judge)."""

    def __call__(self, evidence: str, feedback: str | None) -> Verdict: ...


def _make_generate(generator: VerdictGenerator) -> Callable[[LoopState], LoopState]:
    def generate(state: LoopState) -> LoopState:
        # El feedback del crítico (si lo hay) entra al generador: ESTO es lo que
        # convierte "reintentar" en "optimizar".
        verdict = generator(state["evidence"], state.get("critique"))
        return {
            "verdict": verdict,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    return generate


def critique(state: LoopState) -> LoopState:
    """Evalúa el veredicto contra una rúbrica EXPLÍCITA.

    Determinista aquí para que el ejemplo corra sin API; en el proyecto sería un
    segundo LLM/tier con la rúbrica en el prompt. Devuelve `approved` + feedback
    accionable que el generador usará para corregir.
    """
    v = state["verdict"]
    problems: list[str] = []
    if not 0.0 <= v["confidence"] <= 1.0:
        problems.append("confidence fuera de [0,1]")
    if v["dominant_signal"] == "none" and v["direction"] != "neutral":
        problems.append("dominant_signal=none exige direction=neutral")
    if v["dominant_signal"] != "none" and v["direction"] == "neutral":
        problems.append("hay señal dominante pero la dirección es neutral")
    if len(v["rationale"].split()) < 5:
        problems.append("rationale demasiado pobre para auditar la decisión")
    return {"approved": not problems, "critique": "; ".join(problems)}


def _route(state: LoopState) -> str:
    if state.get("approved"):
        return "done"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "abstain"  # presupuesto agotado → abstención segura
    return "generate"  # hay feedback y queda presupuesto → regenerar


def abstain(state: LoopState) -> LoopState:
    """Sin veredicto coherente tras las revisiones: el sistema se abstiene."""
    safe: Verdict = {
        "direction": "neutral",
        "confidence": 0.0,
        "dominant_signal": "none",
        "rationale": "Sin veredicto coherente tras las revisiones — abstención.",
    }
    return {"verdict": safe, "approved": True}


def build_evaluator_optimizer(generator: VerdictGenerator) -> CompiledStateGraph[LoopState]:
    """Compila el grafo generate → critique → (regenera | abstiene | END)."""
    graph: StateGraph[LoopState] = StateGraph(LoopState)
    graph.add_node("generate", _make_generate(generator))
    graph.add_node("critique", critique)
    graph.add_node("abstain", abstain)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        _route,
        {"generate": "generate", "abstain": "abstain", "done": END},
    )
    graph.add_edge("abstain", END)
    return graph.compile()


def _demo() -> None:
    # Generador de demo: la 1ª vez devuelve un veredicto INCOHERENTE
    # (confidence > 1 y dominant=none con dirección up); con feedback, corrige.
    state = {"n": 0}

    def generator(evidence: str, feedback: str | None) -> Verdict:
        state["n"] += 1
        if state["n"] == 1:
            return {
                "direction": "up",
                "confidence": 1.4,
                "dominant_signal": "none",
                "rationale": "sube",
            }
        return {
            "direction": "neutral",
            "confidence": 0.3,
            "dominant_signal": "none",
            "rationale": "Señales mixtas; ninguna domina con claridad hoy.",
        }

    graph = build_evaluator_optimizer(generator)
    final = graph.invoke({"evidence": "noticias mixtas, serie plana"})
    print(f"revisiones: {final['revision_count']}")
    print(f"aprobado:   {final['approved']}")
    print(f"veredicto:  {final['verdict']}")


if __name__ == "__main__":
    _demo()
