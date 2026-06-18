"""Tests del ejemplo trabajado de la Pista 2 (evaluator-optimizer)."""

from __future__ import annotations

import pytest
from examples.evaluator_optimizer_reference import Verdict, build_evaluator_optimizer


@pytest.mark.unit()
def test_loop_corrects_an_incoherent_verdict() -> None:
    """1er veredicto incoherente → el crítico lo rechaza → regenera y pasa."""
    calls = {"n": 0}

    def generator(evidence: str, feedback: str | None) -> Verdict:
        calls["n"] += 1
        if calls["n"] == 1:
            # incoherente: confidence fuera de rango + dominant=none con dirección
            return {
                "direction": "up",
                "confidence": 1.4,
                "dominant_signal": "none",
                "rationale": "x",
            }
        # el feedback del crítico debe haber llegado
        assert feedback
        return {
            "direction": "neutral",
            "confidence": 0.3,
            "dominant_signal": "none",
            "rationale": "Señales mixtas; ninguna domina con claridad.",
        }

    final = build_evaluator_optimizer(generator).invoke({"evidence": "mixto"})

    assert final["approved"] is True
    assert final["revision_count"] == 2  # una corrección
    assert final["verdict"]["direction"] == "neutral"


@pytest.mark.unit()
def test_loop_abstains_when_budget_exhausted() -> None:
    """Generador que SIEMPRE devuelve incoherencia → abstención al agotar el presupuesto."""

    def bad_generator(evidence: str, feedback: str | None) -> Verdict:
        return {
            "direction": "up",
            "confidence": 9.9,  # nunca coherente
            "dominant_signal": "none",
            "rationale": "x",
        }

    final = build_evaluator_optimizer(bad_generator).invoke({"evidence": "mixto"})

    assert final["approved"] is True  # cerró por abstención, no por bucle infinito
    assert final["verdict"]["direction"] == "neutral"
    assert final["verdict"]["dominant_signal"] == "none"
    assert "abstención" in final["verdict"]["rationale"].lower()
