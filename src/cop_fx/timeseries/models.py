"""Prophet and ARIMA wrappers for COP/USD forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from cop_fx.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    model_name: str
    horizon_days: int
    forecast: pd.DataFrame  # columns: ds, yhat, yhat_lower, yhat_upper
    train_df: pd.DataFrame


class ProphetForecaster:
    """Thin wrapper around Facebook Prophet."""

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "multiplicative",
    ) -> None:
        self._params = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_mode": seasonality_mode,
            "daily_seasonality": False,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
        }

    def fit_predict(self, df: pd.DataFrame, horizon_days: int = 7) -> ForecastResult:
        """Fit on ``df`` (columns: ds, y) and predict ``horizon_days`` ahead."""
        model = Prophet(**self._params)
        model.add_country_holidays(country_name="CO")
        model.fit(df)

        future = model.make_future_dataframe(periods=horizon_days, freq="B")  # business days
        forecast = model.predict(future)
        forecast_tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)

        logger.info(
            "Prophet forecast: last known=%.2f, day+%d=%.2f",
            df["y"].iloc[-1],
            horizon_days,
            forecast_tail["yhat"].iloc[-1],
        )
        return ForecastResult(
            model_name="prophet",
            horizon_days=horizon_days,
            forecast=forecast_tail.reset_index(drop=True),
            train_df=df,
        )


class ARIMAForecaster:
    """Wrapper around statsmodels ARIMA with auto-order fallback."""

    # Orden por defecto respaldado por evidencia (notebooks/series_de_tiempo.ipynb):
    # gana por BIC, deja residuales ruido-blanco (Ljung-Box p=1.0) y supera al
    # ganador por AIC (2,1,3) en hit-rate direccional out-of-sample (52% vs 35%).
    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)) -> None:
        self._order = order

    def fit_predict(self, df: pd.DataFrame, horizon_days: int = 7) -> ForecastResult:
        series = df.set_index("ds")["y"].asfreq("B").ffill()

        try:
            result = ARIMA(series, order=self._order).fit()
        except Exception as exc:
            logger.warning("ARIMA(%s) failed (%s), retrying with (1,1,1)", self._order, exc)
            result = ARIMA(series, order=(1, 1, 1)).fit()

        forecast_obj = result.get_forecast(steps=horizon_days)
        forecast_mean = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)

        forecast_df = pd.DataFrame(
            {
                "ds": forecast_mean.index,
                "yhat": forecast_mean.values,
                "yhat_lower": conf_int.iloc[:, 0].values,
                "yhat_upper": conf_int.iloc[:, 1].values,
            }
        )

        logger.info(
            "ARIMA forecast: last known=%.2f, day+%d=%.2f",
            series.iloc[-1],
            horizon_days,
            forecast_df["yhat"].iloc[-1],
        )
        return ForecastResult(
            model_name="arima",
            horizon_days=horizon_days,
            forecast=forecast_df.reset_index(drop=True),
            train_df=df,
        )


def ensemble_forecast(results: list[ForecastResult]) -> pd.DataFrame:
    """Simple average ensemble across multiple ForecastResult objects."""
    if not results:
        raise ValueError("No forecast results to ensemble")

    base = results[0].forecast[["ds"]].copy()
    yhats = np.column_stack([r.forecast["yhat"].to_numpy(dtype=float) for r in results])
    lower = np.column_stack([r.forecast["yhat_lower"].to_numpy(dtype=float) for r in results])
    upper = np.column_stack([r.forecast["yhat_upper"].to_numpy(dtype=float) for r in results])

    base["yhat"] = yhats.mean(axis=1)
    base["yhat_lower"] = lower.min(axis=1)
    base["yhat_upper"] = upper.max(axis=1)
    return base
