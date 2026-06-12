"""Backtest direccional de la señal de serie de tiempo — sin LLM, sin costo.

Responde la pregunta de honestidad intelectual del proyecto: ¿la señal
técnica le gana a baselines triviales? Si "momentum" (repetir el último
movimiento) acierta igual, la serie no aporta y todo el peso recae en las
noticias. La parte de noticias no se puede backtestear sin archivo histórico
de análisis — eso lo irá acumulando la tabla `predictions` día a día.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cop_fx.logger import get_logger
from cop_fx.timeseries.models import ARIMAForecaster

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

NEUTRAL_BAND_PCT = 0.10


def _direction(change_pct: float, band: float = NEUTRAL_BAND_PCT) -> str:
    if change_pct > band:
        return "up"
    if change_pct < -band:
        return "down"
    return "neutral"


def directional_backtest(
    df: pd.DataFrame,
    *,
    horizon_days: int = 5,
    n_origins: int = 40,
    min_history: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rolling-origin: en cada día t, predice la dirección a t+horizon.

    Estrategias comparadas (todas deterministas):
      - arima:    signo del yhat de ARIMA(2,1,2) vs el último real
      - momentum: repetir la dirección del último movimiento diario
      - always_up: el dólar siempre sube (baseline ingenuo)

    Devuelve (detalle por origen, resumen por estrategia).
    """
    import pandas as pd

    series = df.sort_values("ds").reset_index(drop=True)
    last_origin = len(series) - horizon_days - 1
    first_origin = max(min_history, last_origin - n_origins + 1)
    if first_origin >= last_origin:
        raise ValueError(
            f"Serie demasiado corta: {len(series)} filas para "
            f"min_history={min_history} + horizon={horizon_days}"
        )

    rows: list[dict[str, Any]] = []
    for t in range(first_origin, last_origin + 1):
        train = series.iloc[: t + 1]
        latest = float(train["y"].iloc[-1])
        prev = float(train["y"].iloc[-2])
        actual = float(series["y"].iloc[t + horizon_days])
        actual_dir = _direction((actual - latest) / latest * 100)

        try:
            result = ARIMAForecaster().fit_predict(train, horizon_days=horizon_days)
            yhat = float(result.forecast["yhat"].iloc[-1])
            arima_dir = _direction((yhat - latest) / latest * 100)
        except Exception as exc:
            logger.warning("ARIMA failed at origin %d: %s", t, exc)
            arima_dir = "neutral"

        rows.append(
            {
                "origin": series["ds"].iloc[t],
                "actual": actual_dir,
                "arima": arima_dir,
                "momentum": _direction((latest - prev) / prev * 100),
                "always_up": "up",
            }
        )

    detail = pd.DataFrame(rows)
    summary: dict[str, Any] = {"n_origins": len(detail), "horizon_days": horizon_days}
    for strategy in ("arima", "momentum", "always_up"):
        decided = detail[detail[strategy] != "neutral"]
        summary[strategy] = {
            "hit_rate": (
                float((decided[strategy] == decided["actual"]).mean())
                if len(decided)
                else None
            ),
            "n_decided": len(decided),
            "n_abstained": int((detail[strategy] == "neutral").sum()),
        }
    return detail, summary
