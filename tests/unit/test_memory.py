"""Unit tests for cross-run memory (Etapa 6)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cop_fx.agents.memory import build_track_record
from cop_fx.agents.nodes import load_memory


class _FakeStore:
    """Stub mínimo de PredictionStore: solo expone recent_calls."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def recent_calls(self, *, limit: int = 5) -> list[dict[str, Any]]:
        return self._rows[:limit]


def _row(
    run_date: str,
    direction: str,
    confidence: float,
    actual: str | None,
    hit: int | None,
) -> dict[str, Any]:
    return {
        "run_date": run_date,
        "direction": direction,
        "confidence": confidence,
        "reconciliation": "agree",
        "dominant_signal": "news",
        "actual_direction": actual,
        "final_hit": hit,
    }


@pytest.mark.unit()
def test_build_track_record_empty_is_falsy() -> None:
    assert build_track_record(_FakeStore([])) == {}


@pytest.mark.unit()
def test_build_track_record_computes_hit_rate_and_overconfidence() -> None:
    rows = [
        _row("2026-06-10", "up", 0.70, "down", 0),  # alta confianza y FALLÓ
        _row("2026-06-09", "down", 0.55, "down", 1),  # acertó
        _row("2026-06-08", "neutral", 0.30, "up", None),  # abstención: no cuenta
    ]
    rec = build_track_record(_FakeStore(rows))

    assert rec["n_decided"] == 2  # up + down; la abstención no entra
    assert rec["hit_rate"] == 0.5  # 1 de 2
    assert rec["n_overconfident_misses"] == 1  # el up 0.70 que falló
    assert "acierto 50%" in rec["digest"]
    assert "confianza alta" in rec["digest"]


@pytest.mark.unit()
def test_build_track_record_survives_bad_store() -> None:
    broken = MagicMock()
    broken.recent_calls.side_effect = RuntimeError("db locked")
    assert build_track_record(broken) == {}


@pytest.mark.unit()
def test_load_memory_node_populates_prior_performance() -> None:
    fake = _FakeStore([_row("2026-06-12", "down", 0.6, "down", 1)])
    with patch("cop_fx.agents.nodes.PredictionStore", return_value=fake):
        result = load_memory({"run_date": "2026-06-12"})  # type: ignore[arg-type]

    assert "prior_performance" in result
    assert result["prior_performance"]["n_decided"] == 1
    assert result["prior_performance"]["hit_rate"] == 1.0


@pytest.mark.unit()
def test_load_memory_node_is_resilient_to_store_failure() -> None:
    with patch("cop_fx.agents.nodes.PredictionStore", side_effect=RuntimeError("no db")):
        result = load_memory({"run_date": "2026-06-12"})  # type: ignore[arg-type]
    assert result["prior_performance"] == {}
