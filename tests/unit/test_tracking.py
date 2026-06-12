"""Unit tests for Etapa 5: PredictionStore + directional backtest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from cop_fx.contracts import DirectionalCall, NewsSignal, TimeSeriesSignal
from cop_fx.tracking import PredictionStore, directional_backtest

if TYPE_CHECKING:
    from pathlib import Path


def _call(direction: str = "up", confidence: float = 0.6) -> DirectionalCall:
    return DirectionalCall(
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        horizon_days=5,
        news_signal=NewsSignal(direction="up", score=-1.5, drivers=["riesgo país"]),
        ts_signal=TimeSeriesSignal(direction="up", yhat_delta_pct=0.5, models_agree=True),
        reconciliation="agree",
        dominant_signal="news",
        rationale="r",
        devils_advocate="El petróleo podría repuntar y revertir la presión sobre el peso.",
        caveats=[],
    )


def _fx_series(start: str, days: int, start_rate: float, daily_change: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rates = [start_rate * (1 + daily_change) ** i for i in range(days)]
    return pd.DataFrame({"ds": dates, "y": rates})


@pytest.mark.unit()
def test_store_roundtrip_and_upsert(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "p.db")
    store.save(_call("up", 0.6), run_date="2026-06-01", latest_rate=4000.0)
    store.save(_call("down", 0.4), run_date="2026-06-01", latest_rate=4000.0)  # upsert

    df = store.all()
    assert len(df) == 1
    assert df.iloc[0]["direction"] == "down"
    assert df.iloc[0]["confidence"] == 0.4


@pytest.mark.unit()
def test_evaluate_pending_scores_hits(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "p.db")
    # Predicción "up" el 01-jun con tasa 4000; horizonte 5 días
    store.save(_call("up"), run_date="2026-06-01", latest_rate=4000.0)
    # La serie efectivamente sube ~0.2%/día → en el horizonte: up
    fx = _fx_series("2026-06-01", 10, 4000.0, 0.002)

    assert store.evaluate_pending(fx) == 1
    row = store.all().iloc[0]
    assert row["actual_direction"] == "up"
    assert row["hit"] == 1
    assert row["final_hit"] == 1
    assert row["news_hit"] == 1
    assert row["ts_hit"] == 1

    # Re-evaluar no duplica trabajo
    assert store.evaluate_pending(fx) == 0


@pytest.mark.unit()
def test_evaluate_neutral_prediction_has_null_hit(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "p.db")
    store.save(_call("neutral", 0.4), run_date="2026-06-01", latest_rate=4000.0)
    fx = _fx_series("2026-06-01", 10, 4000.0, 0.002)
    store.evaluate_pending(fx)

    row = store.all().iloc[0]
    assert row["actual_direction"] == "up"
    assert pd.isna(row["hit"])  # la abstención no se premia ni castiga


@pytest.mark.unit()
def test_evaluate_skips_unreached_horizon(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "p.db")
    store.save(_call("up"), run_date="2026-06-08", latest_rate=4000.0)
    fx = _fx_series("2026-06-01", 9, 4000.0, 0.002)  # solo llega al 09-jun

    assert store.evaluate_pending(fx) == 0
    assert store.all().iloc[0]["evaluated_at"] is None


@pytest.mark.unit()
def test_metrics_shapes(tmp_path: Path) -> None:
    store = PredictionStore(tmp_path / "p.db")
    store.save(_call("up", 0.7), run_date="2026-06-01", latest_rate=4000.0)
    store.save(_call("down", 0.5), run_date="2026-06-02", latest_rate=4008.0)
    fx = _fx_series("2026-06-01", 12, 4000.0, 0.002)  # siempre sube
    store.evaluate_pending(fx)

    m = store.metrics()
    assert m["n_evaluated"] == 2
    assert m["n_decided"] == 2
    assert m["hit_rate"] == 0.5  # up acertó, down falló
    assert m["final_hit_rate"] == 0.5
    assert m["news_hit_rate"] == 1.0
    assert m["ts_hit_rate"] == 1.0
    assert "confusion" in m and "hit_rate_by_confidence" in m


@pytest.mark.unit()
def test_directional_backtest_beats_nothing_on_trend() -> None:
    # Tendencia alcista clara + ruido leve: ARIMA y momentum deberían superar 50%
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    rates = 4000 + np.arange(200) * 3.0 + rng.normal(0, 2, 200)
    df = pd.DataFrame({"ds": dates, "y": rates})

    detail, summary = directional_backtest(df, horizon_days=5, n_origins=15, min_history=120)

    assert summary["n_origins"] == 15
    assert set(detail.columns) >= {"origin", "actual", "arima", "momentum", "always_up"}
    # En tendencia pura, always_up acierta ~100% y ARIMA debería seguirla
    assert summary["always_up"]["hit_rate"] > 0.9
    assert summary["arima"]["hit_rate"] is not None
