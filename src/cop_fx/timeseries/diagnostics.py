"""Diagnóstico clásico de series de tiempo (Box-Jenkins) para el USD/COP.

Hasta ahora el orden ARIMA(2,1,2) era un acto de fe. Este módulo lo somete
al proceso correcto:

  1. Estacionariedad — ADF (H0: raíz unitaria) + KPSS (H0: estacionaria).
     Juntos sugieren el orden de diferenciación `d`.
  2. ACF / PACF — qué rezagos tienen memoria real (fuera de las bandas de
     confianza). PACF que corta en p sugiere AR(p); ACF que corta en q
     sugiere MA(q).
  3. Selección por AIC — grid pequeño de (p, d, q) y que gane la evidencia.
  4. Ljung-Box sobre residuales — si quedan autocorrelacionados, el modelo
     dejó estructura sin capturar.

Todo determinista, cero LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import pacf as sm_pacf

from cop_fx.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Estacionariedad
# ---------------------------------------------------------------------------

@dataclass
class StationarityReport:
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    is_stationary: bool
    verdict: str


def stationarity_tests(series: pd.Series) -> StationarityReport:
    """ADF + KPSS sobre la misma serie — sus hipótesis nulas son opuestas.

    ADF rechaza (p<0.05) ⇒ evidencia de estacionariedad.
    KPSS rechaza (p<0.05) ⇒ evidencia de NO estacionariedad.
    Concluir solo cuando ambos apuntan al mismo lado; si se contradicen,
    el veredicto es "ambiguo" y conviene diferenciar por prudencia.
    """
    clean = series.dropna().to_numpy()
    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")

    adf_says_stationary = adf_p < ALPHA
    kpss_says_stationary = kpss_p >= ALPHA
    if adf_says_stationary and kpss_says_stationary:
        verdict, stationary = "estacionaria (ADF y KPSS concuerdan)", True
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict, stationary = "NO estacionaria (ADF y KPSS concuerdan)", False
    else:
        verdict, stationary = "ambigua (ADF y KPSS se contradicen) — diferenciar por prudencia", False

    return StationarityReport(
        adf_stat=round(float(adf_stat), 4),
        adf_pvalue=round(float(adf_p), 4),
        kpss_stat=round(float(kpss_stat), 4),
        kpss_pvalue=round(float(kpss_p), 4),
        is_stationary=stationary,
        verdict=verdict,
    )


def suggest_d(series: pd.Series, max_d: int = 2) -> int:
    """Diferenciar hasta que ADF+KPSS declaren estacionariedad (máx. max_d)."""
    current = series.dropna()
    for d in range(max_d + 1):
        if stationarity_tests(current).is_stationary:
            return d
        current = current.diff().dropna()
    return max_d


# ---------------------------------------------------------------------------
# 2. ACF / PACF
# ---------------------------------------------------------------------------

@dataclass
class CorrelogramReport:
    acf: np.ndarray
    pacf: np.ndarray
    conf_band: float                       # banda ±1.96/√n
    significant_acf_lags: list[int] = field(default_factory=list)
    significant_pacf_lags: list[int] = field(default_factory=list)


def correlogram(series: pd.Series, nlags: int = 20) -> CorrelogramReport:
    """ACF y PACF con los rezagos que salen de la banda de confianza al 95%."""
    clean = series.dropna().to_numpy()
    acf_vals = np.asarray(sm_acf(clean, nlags=nlags, fft=True))
    pacf_vals = np.asarray(sm_pacf(clean, nlags=nlags, method="ywm"))
    band = 1.96 / np.sqrt(len(clean))

    return CorrelogramReport(
        acf=acf_vals,
        pacf=pacf_vals,
        conf_band=round(float(band), 4),
        significant_acf_lags=[i for i in range(1, nlags + 1) if abs(acf_vals[i]) > band],
        significant_pacf_lags=[i for i in range(1, nlags + 1) if abs(pacf_vals[i]) > band],
    )


# ---------------------------------------------------------------------------
# 3. Selección de orden por AIC
# ---------------------------------------------------------------------------

def select_arima_order(
    df: pd.DataFrame,
    *,
    max_p: int = 3,
    max_q: int = 3,
    d: int | None = None,
) -> tuple[tuple[int, int, int], pd.DataFrame]:
    """Grid pequeño de (p, d, q) sobre la serie; gana el menor AIC.

    Devuelve (orden_ganador, tabla completa ordenada por AIC).
    `d` se infiere con ADF+KPSS si no se pasa.
    """
    import pandas as pd

    y = df["y"].reset_index(drop=True)
    d_used = suggest_d(y) if d is None else d

    rows: list[dict[str, object]] = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fitted = ARIMA(y, order=(p, d_used, q)).fit()
                rows.append({"order": (p, d_used, q), "aic": round(float(fitted.aic), 2),
                             "bic": round(float(fitted.bic), 2)})
            except Exception as exc:
                logger.warning("ARIMA(%d,%d,%d) no convergió: %s", p, d_used, q, exc)

    if not rows:
        raise RuntimeError("Ningún orden ARIMA convergió")
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = table.iloc[0]["order"]
    logger.info("Orden ARIMA seleccionado por AIC: %s", best)
    return best, table


# ---------------------------------------------------------------------------
# 4. Diagnóstico de residuales
# ---------------------------------------------------------------------------

@dataclass
class ResidualReport:
    order: tuple[int, int, int]
    ljung_box_pvalue: float
    residuals_autocorrelated: bool
    aic: float
    verdict: str


def residual_diagnostics(
    df: pd.DataFrame, order: tuple[int, int, int], lags: int = 10
) -> ResidualReport:
    """Ljung-Box sobre los residuales: H0 = ruido blanco (lo deseable)."""
    fitted = ARIMA(df["y"].reset_index(drop=True), order=order).fit()
    lb = acorr_ljungbox(fitted.resid, lags=[lags], return_df=True)
    pvalue = float(lb["lb_pvalue"].iloc[0])
    autocorrelated = pvalue < ALPHA

    return ResidualReport(
        order=order,
        ljung_box_pvalue=round(pvalue, 4),
        residuals_autocorrelated=autocorrelated,
        aic=round(float(fitted.aic), 2),
        verdict=(
            "residuales AUTOCORRELACIONADOS — el modelo deja estructura sin capturar"
            if autocorrelated
            else "residuales ≈ ruido blanco — el modelo capturó la estructura lineal"
        ),
    )
