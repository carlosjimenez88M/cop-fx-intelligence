"""Etapa 6 — memoria entre corridas para el adjudicador.

El sistema ya guarda cada veredicto y lo auto-califica cuando vence el horizonte
(`tracking/predictions.py`). Eso es un archivo histórico durable: la memoria a
largo plazo del desk. Este módulo lo lee y lo destila en un **digest** que el
adjudicador recibe como contexto, para que se calibre contra su propio récord
en vez de empezar cada día desde cero ("tus últimas llamadas alcistas fallaron;
exige más evidencia antes de repetir").

Diseño:
  - `build_track_record(store)` es una función PURA sobre un `PredictionStore`
    ya construido — fácil de testear y agnóstica de LangGraph.
  - El nodo `load_memory` (en `agents/nodes.py`) la invoca y, si el grafo se
    compiló con un `BaseStore`, publica el récord en el namespace de memoria
    a largo plazo — demostrando la interfaz store.put/get de LangGraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cop_fx.tracking.predictions import PredictionStore

# Namespace de memoria a largo plazo (convención store: namespace → key → value).
MEMORY_NAMESPACE = ("cop_fx", "adjudicator")
MEMORY_KEY = "track_record"

_HIT_MARK = {1: "acertó", 0: "falló"}


def build_track_record(store: PredictionStore, *, limit: int = 5) -> dict[str, Any]:
    """Destila el historial de predicciones en un dict listo para el prompt.

    Devuelve ``{}`` si no hay historial o si la lectura falla — la memoria es
    contexto opcional, nunca un punto de quiebre del pipeline.
    """
    try:
        recent = store.recent_calls(limit=limit)
        if not isinstance(recent, list) or not recent:
            return {}
    except Exception:
        return {}

    evaluated = [r for r in recent if r.get("final_hit") in (0, 1)]
    decided = [r for r in evaluated if r.get("direction") != "neutral"]
    hits = sum(int(r["final_hit"]) for r in decided)
    hit_rate = round(hits / len(decided), 2) if decided else None

    # Señal de sobre-confianza: llamadas decididas con confianza alta que fallaron.
    overconfident = [
        r for r in decided if r.get("final_hit") == 0 and float(r.get("confidence") or 0) >= 0.6
    ]

    lines: list[str] = []
    for r in recent:
        hit = r.get("final_hit")
        outcome = (
            f" → real {r.get('actual_direction')} ({_HIT_MARK.get(int(hit))})"
            if hit in (0, 1)
            else " → aún sin evaluar"
        )
        lines.append(
            f"{r.get('run_date')}: {r.get('direction')} "
            f"(conf {float(r.get('confidence') or 0):.2f}, domina {r.get('dominant_signal')})"
            f"{outcome}"
        )

    digest_parts = [
        f"{len(decided)} llamadas direccionales evaluadas; "
        + (f"acierto {hit_rate:.0%}." if hit_rate is not None else "aún sin acierto medible."),
    ]
    if overconfident:
        digest_parts.append(
            f"Atención: {len(overconfident)} llamada(s) de confianza alta (>=0.6) "
            "fallaron — modera la confianza si la evidencia de hoy se parece."
        )
    digest_parts.append("Historial reciente:\n" + "\n".join(lines))

    return {
        "n_decided": len(decided),
        "hit_rate": hit_rate,
        "n_overconfident_misses": len(overconfident),
        "recent": recent,
        "digest": "\n".join(digest_parts),
    }
