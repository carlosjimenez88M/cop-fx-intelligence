"""Unit tests for the time-series diagnostics module (ACF/PACF, ADF/KPSS, AIC)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cop_fx.timeseries.diagnostics import (
    correlogram,
    residual_diagnostics,
    select_arima_order,
    stationarity_tests,
    suggest_d,
)


@pytest.fixture()
def random_walk() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(4000 + rng.normal(0, 10, 300).cumsum())


@pytest.fixture()
def white_noise() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 300))


@pytest.fixture()
def ar1_series() -> pd.Series:
    """AR(1) fuerte: y_t = 0.8·y_{t-1} + ε — PACF debe cortar en el rezago 1."""
    rng = np.random.default_rng(7)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.8 * y[t - 1] + rng.normal(0, 1)
    return pd.Series(y)


@pytest.mark.unit()
def test_random_walk_is_not_stationary(random_walk: pd.Series) -> None:
    report = stationarity_tests(random_walk)
    assert report.is_stationary is False


@pytest.mark.unit()
def test_white_noise_is_stationary(white_noise: pd.Series) -> None:
    report = stationarity_tests(white_noise)
    assert report.is_stationary is True


@pytest.mark.unit()
def test_suggest_d_for_random_walk(random_walk: pd.Series) -> None:
    # Una diferencia convierte el random walk en ruido blanco
    assert suggest_d(random_walk) == 1


@pytest.mark.unit()
def test_correlogram_detects_ar1_memory(ar1_series: pd.Series) -> None:
    report = correlogram(ar1_series, nlags=10)
    # AR(1): el rezago 1 domina la PACF; la ACF decae lento (varios significativos)
    assert 1 in report.significant_pacf_lags
    assert len(report.significant_acf_lags) >= 3
    assert report.conf_band > 0


@pytest.mark.unit()
def test_correlogram_white_noise_mostly_silent(white_noise: pd.Series) -> None:
    report = correlogram(white_noise, nlags=20)
    # Al 95%, por azar se espera ~1 falso positivo en 20 rezagos
    assert len(report.significant_acf_lags) <= 2


@pytest.mark.unit()
def test_select_arima_order_returns_valid_tuple(ar1_series: pd.Series) -> None:
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(ar1_series)), "y": ar1_series})
    best, table = select_arima_order(df, max_p=2, max_q=2)
    assert len(best) == 3
    assert table["aic"].is_monotonic_increasing
    # Para un AR(1) puro, el orden ganador debe incluir componente AR
    assert best[0] >= 1


@pytest.mark.unit()
def test_residual_diagnostics_white_noise_after_fit(ar1_series: pd.Series) -> None:
    df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(ar1_series)), "y": ar1_series})
    report = residual_diagnostics(df, order=(1, 0, 0))
    # Un AR(1) bien ajustado deja residuales ≈ ruido blanco
    assert report.residuals_autocorrelated is False
    assert report.ljung_box_pvalue > 0.05
