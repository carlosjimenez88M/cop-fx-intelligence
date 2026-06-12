"""Series macro exógenas para el análisis multi-señal (arquitectura.md §2).

El USD/COP es mitad Colombia, mitad dólar global. Estas series capturan
los drivers fundamentales como UNA serie numérica cada uno — gratis, sin
LLM y sin scraping:

  - brent  (BZ=F)     : Colombia exporta petróleo → términos de intercambio
  - dxy    (DX-Y.NYB) : el lado dólar de la ecuación
  - equity (GXG)      : ETF MSCI Colombia — el agregado de la bolsa local
                        (proxy de COLCAP; apetito de riesgo país)

Acciones INDIVIDUALES quedan fuera a propósito: para un problema
direccional agregado aportan ruido > señal (ver arquitectura.md §2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from cop_fx.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

MARKET_SYMBOLS: dict[str, str] = {
    "brent": "BZ=F",
    "dxy": "DX-Y.NYB",
    "equity": "GXG",
}

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (cop-fx-intelligence)"}


def fetch_yahoo_series(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    """Serie diaria [ds, y] desde la chart API pública de Yahoo Finance."""
    import pandas as pd

    params = {"range": f"{max(lookback_days, 30)}d", "interval": "1d"}
    with httpx.Client(timeout=30, headers=_HEADERS) as client:
        resp = client.get(_CHART_URL.format(symbol=symbol), params=params)
        resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "y": pd.to_numeric(
                pd.Series(result["indicators"]["quote"][0]["close"]), errors="coerce"
            ),
        }
    )
    return df.dropna().drop_duplicates(subset="ds").reset_index(drop=True)


def fetch_market_panel(
    fx_df: pd.DataFrame, *, lookback_days: int = 365
) -> pd.DataFrame:
    """Panel alineado por fecha: cop + brent + dxy + equity (inner join).

    Las series que fallen se omiten con warning — el panel degrada con
    gracia en vez de tumbar el análisis.
    """
    panel = fx_df.rename(columns={"y": "cop"})[["ds", "cop"]].copy()
    for name, symbol in MARKET_SYMBOLS.items():
        try:
            series = fetch_yahoo_series(symbol, lookback_days)
            panel = panel.merge(
                series.rename(columns={"y": name}), on="ds", how="inner"
            )
            logger.info("Serie %s (%s): %d filas", name, symbol, len(series))
        except Exception as exc:
            logger.warning("Serie %s (%s) falló: %s — se omite", name, symbol, exc)
    return panel.sort_values("ds").reset_index(drop=True)
