"""Unit tests del módulo direccional macro y del composite de riesgo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cop_fx.data import market_fetcher
from cop_fx.timeseries import directional as D


def _panel(n: int = 400, *, seed: int = 0) -> pd.DataFrame:
    """Panel sintético donde el COP futuro depende del retorno de un driver."""
    rng = np.random.default_rng(seed)
    ds = pd.bdate_range("2022-01-01", periods=n)
    driver = np.cumprod(1 + rng.normal(0, 0.01, n))
    # cop sube cuando el driver bajó hace H días → señal aprendible
    shock = np.r_[np.zeros(5), -np.diff(driver, 5)]
    cop = np.cumprod(1 + 0.5 * shock + rng.normal(0, 0.003, n))
    return pd.DataFrame({"ds": ds, "cop": cop * 4000, "equity": driver, "dxy": driver[::-1]})


@pytest.mark.unit()
def test_make_features_no_future_leak() -> None:
    panel = _panel()
    x, y, cols = D.make_features(panel, horizon=5)
    assert len(x) == len(y)
    assert set(y.unique()) <= {0.0, 1.0}
    assert all(c.endswith(("_r1", "_r5", "_r10")) for c in cols)
    # X no debe contener NaN (se dropean las filas iniciales)
    assert not x.isna().to_numpy().any()


@pytest.mark.unit()
def test_logistic_learns_signal() -> None:
    panel = _panel(500)
    res = D.walk_forward_accuracy(
        panel, horizon=5, min_train=200, n_eval=120, refit_every=30, include_gru=False
    )
    acc = dict(zip(res["modelo"], res["accuracy"], strict=False))
    # Con señal sintética, la logística debe superar a always_up claramente.
    assert acc["logistic"] >= acc["always_up"]
    assert set(res["modelo"]) == {"always_up", "momentum", "logistic", "gboost"}


@pytest.mark.unit()
def test_macro_directional_signal_shape() -> None:
    panel = _panel(400)
    sig = D.macro_directional_signal(panel=panel, horizon=5)
    assert sig is not None
    assert sig.direction in {"up", "down", "neutral"}
    assert 0.0 <= sig.prob_up <= 1.0
    assert sig.n_train > 0


@pytest.mark.unit()
def test_macro_directional_signal_insufficient_data() -> None:
    tiny = _panel(50)
    assert D.macro_directional_signal(panel=tiny) is None


@pytest.mark.unit()
def test_risk_context_composite_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bolsa local subiendo fuerte ⇒ composite negativo ⇒ presión a la baja del USD/COP."""

    def fake_series(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        n = 30
        base = np.linspace(100, 100, n)
        if symbol == "GXG":  # equity sube hoy (último retorno positivo grande)
            base = base.copy()
            base[-1] = 105.0
        return pd.DataFrame({"ds": pd.bdate_range("2024-01-01", periods=n), "y": base})

    monkeypatch.setattr(market_fetcher, "fetch_yahoo_series", fake_series)
    ctx = market_fetcher.compute_risk_context()

    assert ctx  # no vacío
    assert ctx["n_drivers"] == len(market_fetcher.RISK_BASKET)
    # equity tiene signo negativo en el canasto ⇒ su subida empuja el composite a negativo
    assert float(ctx["composite"]) < 0


@pytest.mark.unit()
def test_risk_context_empty_when_all_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        raise ConnectionError("yahoo down")

    monkeypatch.setattr(market_fetcher, "fetch_yahoo_series", boom)
    assert market_fetcher.compute_risk_context() == {}
