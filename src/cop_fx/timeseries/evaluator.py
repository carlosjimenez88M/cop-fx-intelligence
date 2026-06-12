"""Forecast evaluation metrics: MAE, RMSE, MAPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from cop_fx.timeseries.models import ForecastResult


class Forecaster(Protocol):
    def fit_predict(self, df: pd.DataFrame, horizon_days: int = 7) -> ForecastResult:
        """Fit a model and return a forecast."""
        ...


@dataclass
class EvalMetrics:
    model_name: str
    mae: float
    rmse: float
    mape: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"{self.model_name} — MAE={self.mae:.2f}  RMSE={self.rmse:.2f}  "
            f"MAPE={self.mape:.2f}%  (n={self.n_samples})"
        )


def evaluate(result: ForecastResult, actuals: pd.DataFrame) -> EvalMetrics:
    """Compare in-sample or out-of-sample predictions against actuals.

    ``actuals`` must have columns [ds, y].
    """
    merged = result.forecast.merge(actuals, on="ds", how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping dates between forecast ({result.model_name}) and actuals")

    y_true = merged["y"].to_numpy(dtype=float)
    y_pred = merged["yhat"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # avoid division by zero
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-9))) * 100)

    return EvalMetrics(
        model_name=result.model_name,
        mae=mae,
        rmse=rmse,
        mape=mape,
        n_samples=len(merged),
    )


def walk_forward_eval(
    df: pd.DataFrame,
    forecaster: Forecaster,
    horizon_days: int = 7,
    n_splits: int = 4,
) -> list[EvalMetrics]:
    """Rolling-origin cross-validation."""
    n = len(df)
    split_size = n // (n_splits + 1)
    metrics_list: list[EvalMetrics] = []

    for i in range(1, n_splits + 1):
        cutoff = split_size * i
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff : cutoff + horizon_days]
        if test.empty:
            break
        result = forecaster.fit_predict(train, horizon_days=len(test))
        metrics_list.append(evaluate(result, test))

    return metrics_list
