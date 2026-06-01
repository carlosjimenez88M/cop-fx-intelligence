"""Unit tests for timeseries forecasting models and evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cop_fx.timeseries.evaluator import EvalMetrics, evaluate, walk_forward_eval
from cop_fx.timeseries.models import (
    ARIMAForecaster,
    ForecastResult,
    ProphetForecaster,
    ensemble_forecast,
)


@pytest.fixture()
def synthetic_fx_df() -> pd.DataFrame:
    """Daily COP/USD-like data with slight upward trend + noise."""
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    values = 4000.0 + np.arange(n) * 0.5 + rng.normal(0, 20, n)
    return pd.DataFrame({"ds": dates, "y": values})


# ── Prophet ────────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_prophet_returns_correct_horizon(synthetic_fx_df: pd.DataFrame) -> None:
    forecaster = ProphetForecaster()
    result = forecaster.fit_predict(synthetic_fx_df, horizon_days=5)
    assert isinstance(result, ForecastResult)
    assert result.model_name == "prophet"
    assert len(result.forecast) == 5
    assert set(result.forecast.columns) >= {"ds", "yhat", "yhat_lower", "yhat_upper"}


@pytest.mark.unit()
def test_prophet_forecast_values_are_positive(synthetic_fx_df: pd.DataFrame) -> None:
    result = ProphetForecaster().fit_predict(synthetic_fx_df, horizon_days=3)
    assert (result.forecast["yhat"] > 0).all()


# ── ARIMA ──────────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_arima_returns_correct_horizon(synthetic_fx_df: pd.DataFrame) -> None:
    result = ARIMAForecaster().fit_predict(synthetic_fx_df, horizon_days=5)
    assert result.model_name == "arima"
    assert len(result.forecast) == 5


@pytest.mark.unit()
def test_arima_forecast_confidence_interval_ordering(synthetic_fx_df: pd.DataFrame) -> None:
    result = ARIMAForecaster().fit_predict(synthetic_fx_df, horizon_days=3)
    assert (result.forecast["yhat_lower"] <= result.forecast["yhat"]).all()
    assert (result.forecast["yhat"] <= result.forecast["yhat_upper"]).all()


# ── Ensemble ───────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_ensemble_averages_predictions(synthetic_fx_df: pd.DataFrame) -> None:
    r1 = ProphetForecaster().fit_predict(synthetic_fx_df, horizon_days=3)
    r2 = ARIMAForecaster().fit_predict(synthetic_fx_df, horizon_days=3)
    ens = ensemble_forecast([r1, r2])
    assert len(ens) == 3
    # ensemble yhat should be between the two model yhats
    low = pd.concat([r1.forecast["yhat"], r2.forecast["yhat"]], axis=1).min(axis=1)
    high = pd.concat([r1.forecast["yhat"], r2.forecast["yhat"]], axis=1).max(axis=1)
    assert (ens["yhat"] >= low - 1e-6).all()
    assert (ens["yhat"] <= high + 1e-6).all()


@pytest.mark.unit()
def test_ensemble_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match="No forecast results"):
        ensemble_forecast([])


# ── Evaluator ──────────────────────────────────────────────────────────────

@pytest.mark.unit()
def test_evaluate_perfect_forecast(synthetic_fx_df: pd.DataFrame) -> None:
    n = len(synthetic_fx_df)
    test_df = synthetic_fx_df.tail(5).copy()
    # craft a "perfect" forecast whose yhat == y
    perfect_forecast_df = test_df.rename(columns={"y": "yhat"}).copy()
    perfect_forecast_df["yhat_lower"] = perfect_forecast_df["yhat"] - 10
    perfect_forecast_df["yhat_upper"] = perfect_forecast_df["yhat"] + 10

    result = ForecastResult(
        model_name="perfect",
        horizon_days=5,
        forecast=perfect_forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        train_df=synthetic_fx_df.head(n - 5),
    )
    metrics = evaluate(result, test_df)
    assert metrics.mae < 1e-6
    assert metrics.rmse < 1e-6
    assert metrics.mape < 1e-4


@pytest.mark.unit()
def test_evaluate_no_overlap_raises() -> None:
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=3, freq="D"),
            "yhat": [1.0, 2.0, 3.0],
            "yhat_lower": [0.9, 1.9, 2.9],
            "yhat_upper": [1.1, 2.1, 3.1],
        }
    )
    actuals = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=3, freq="D"), "y": [1.0, 2.0, 3.0]}
    )
    result = ForecastResult(
        model_name="test", horizon_days=3, forecast=df, train_df=pd.DataFrame()
    )
    with pytest.raises(ValueError, match="No overlapping dates"):
        evaluate(result, actuals)


@pytest.mark.unit()
def test_walk_forward_eval_returns_metrics(synthetic_fx_df: pd.DataFrame) -> None:
    metrics = walk_forward_eval(synthetic_fx_df, ARIMAForecaster(), horizon_days=3, n_splits=2)
    assert len(metrics) == 2
    for m in metrics:
        assert isinstance(m, EvalMetrics)
        assert m.mae >= 0
