"""Servicio de forecast: ajusta Prophet + ARIMA y deriva el signo del ensemble.

El ajuste es CPU-bound (statsmodels/Prophet); corre en threadpool para no
bloquear el loop. La fuente de la serie es el CSV cacheado si existe; si no,
cae al ``FXFetcher`` en vivo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from fastapi.concurrency import run_in_threadpool

from cop_fx.api.errors import DataUnavailable
from cop_fx.api.schemas import (
    Direction,
    ForecastPoint,
    ForecastResponse,
    ModelDelta,
)
from cop_fx.data.fx_fetcher import FXFetcher
from cop_fx.paths import DATA_DIR
from cop_fx.timeseries.models import (
    ARIMAForecaster,
    ProphetForecaster,
    ensemble_forecast,
)

if TYPE_CHECKING:
    from cop_fx.config.settings import Settings

_FX_CSV = DATA_DIR / "cop_usd.csv"


class ForecastService:
    """Calcula el forecast del USD/COP y su dirección implícita."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def compute(self, horizon_days: int | None = None) -> ForecastResponse:
        horizon = horizon_days or self._settings.forecast_horizon_days
        return await run_in_threadpool(self._compute_sync, horizon)

    # ------------------------------------------------------------------

    def _load_fx(self) -> pd.DataFrame:
        if _FX_CSV.exists():
            return pd.read_csv(_FX_CSV, parse_dates=["ds"]).sort_values("ds")
        return FXFetcher().fetch()

    def _compute_sync(self, horizon: int) -> ForecastResponse:
        fx = self._load_fx()
        if fx.empty:
            raise DataUnavailable("No hay serie de TRM disponible para el forecast")

        prophet = ProphetForecaster().fit_predict(fx, horizon_days=horizon)
        arima = ARIMAForecaster().fit_predict(fx, horizon_days=horizon)
        ensemble = ensemble_forecast([prophet, arima])
        latest = float(fx["y"].iloc[-1])

        models = [
            ModelDelta(
                model=name,
                yhat_final=(yhat := float(result.forecast["yhat"].iloc[-1])),
                delta_pct=round((yhat - latest) / latest * 100, 3),
            )
            for name, result in (("Prophet", prophet), ("ARIMA", arima))
        ]
        yhat_ensemble = float(ensemble["yhat"].iloc[-1])
        ensemble_delta = round((yhat_ensemble - latest) / latest * 100, 3)
        models.append(
            ModelDelta(model="Ensemble", yhat_final=yhat_ensemble, delta_pct=ensemble_delta)
        )

        points = [
            ForecastPoint(ds=pd.Timestamp(ds).to_pydatetime(), yhat=float(yhat))
            for ds, yhat in zip(ensemble["ds"], ensemble["yhat"], strict=False)
        ]
        return ForecastResponse(
            latest_rate=latest,
            horizon_days=horizon,
            direction=self._direction(ensemble_delta),
            models=models,
            ensemble=points,
        )

    def _direction(self, delta_pct: float) -> Direction:
        band = self._settings.ts_neutral_band_pct
        if delta_pct <= -band:
            return "down"
        if delta_pct >= band:
            return "up"
        return "neutral"
