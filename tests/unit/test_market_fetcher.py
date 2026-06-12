"""Unit tests for the exogenous market series fetcher."""

from __future__ import annotations

import pandas as pd
import pytest

from cop_fx.data import market_fetcher


def _series(start: str, days: int, base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"ds": pd.date_range(start, periods=days, freq="D"), "y": [base + i for i in range(days)]}
    )


@pytest.mark.unit()
def test_panel_aligns_on_common_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    fx = _series("2026-01-01", 10, 4000.0)
    fake = {
        "BZ=F": _series("2026-01-03", 10, 70.0),       # empieza 2 días después
        "DX-Y.NYB": _series("2026-01-01", 8, 100.0),   # termina antes
        "GXG": _series("2026-01-01", 10, 30.0),
    }
    monkeypatch.setattr(
        market_fetcher, "fetch_yahoo_series", lambda symbol, lookback_days=365: fake[symbol]
    )

    panel = market_fetcher.fetch_market_panel(fx)

    assert list(panel.columns) == ["ds", "cop", "brent", "dxy", "equity"]
    # inner join: solo fechas comunes (03..08 ene = 6 días)
    assert len(panel) == 6
    assert panel["ds"].is_monotonic_increasing


@pytest.mark.unit()
def test_panel_degrades_gracefully_when_a_series_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _series("2026-01-01", 10, 4000.0)

    def _fetch(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        if symbol == "BZ=F":
            raise ConnectionError("yahoo down")
        return _series("2026-01-01", 10, 100.0)

    monkeypatch.setattr(market_fetcher, "fetch_yahoo_series", _fetch)
    panel = market_fetcher.fetch_market_panel(fx)

    assert "brent" not in panel.columns       # la caída no tumba el panel
    assert {"cop", "dxy", "equity"} <= set(panel.columns)
    assert len(panel) == 10
