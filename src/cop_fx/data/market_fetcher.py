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


def fetch_market_panel(fx_df: pd.DataFrame, *, lookback_days: int = 365) -> pd.DataFrame:
    """Panel alineado por fecha: cop + brent + dxy + equity (inner join).

    Las series que fallen se omiten con warning — el panel degrada con
    gracia en vez de tumbar el análisis.
    """
    panel = fx_df.rename(columns={"y": "cop"})[["ds", "cop"]].copy()
    for name, symbol in MARKET_SYMBOLS.items():
        try:
            series = fetch_yahoo_series(symbol, lookback_days)
            panel = panel.merge(series.rename(columns={"y": name}), on="ds", how="inner")
            logger.info("Serie %s (%s): %d filas", name, symbol, len(series))
        except Exception as exc:
            logger.warning("Serie %s (%s) falló: %s — se omite", name, symbol, exc)
    return panel.sort_values("ds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contexto de riesgo multi-driver (coincidente, NO forecast)
# ---------------------------------------------------------------------------
#
# El estudio macro (notebooks/02) mostró dos cosas:
#   1. El USD/COP co-mueve fuerte con un canasto de activos de riesgo EM:
#      bolsa local (-), pares EM CLP/MXN/BRL (+), DXY (+), cobre/petróleo (-).
#   2. Ese co-movimiento es CONTEMPORÁNEO; el poder PREDICTIVO a 1-5 días es
#      débil (gboost/GRU no superan de forma estable al baseline trivial).
#
# Por eso el contexto se construye como un composite de RIESGO coincidente, no
# como un modelo que pretenda adivinar el futuro: leemos el movimiento de ayer
# del canasto y lo orientamos hacia "presión sobre el USD/COP". Es evidencia
# para el adjudicador, con banda muerta para abstenerse cuando el canasto es
# ambiguo. Cada driver pondera por su |correlación semanal| con el COP.

# (símbolo, signo hacia USD/COP-arriba, peso = |corr semanal|)
RISK_BASKET: tuple[tuple[str, str, float], ...] = (
    ("equity", "GXG", -0.68),
    ("usdclp", "CLP=X", 0.48),
    ("usdmxn", "MXN=X", 0.45),
    ("usdbrl", "BRL=X", 0.43),
    ("cobre", "HG=F", -0.39),
    ("dxy", "DX-Y.NYB", 0.35),
    ("brent", "BZ=F", -0.23),
)


def compute_risk_context(*, lookback_days: int = 40) -> dict[str, object]:
    """Composite de riesgo coincidente orientado al USD/COP.

    Devuelve un dict con la dirección (banda muerta sobre el composite),
    el score y los retornos de ayer de los componentes clave. Cada driver
    aporta su retorno de 1 día normalizado por su volatilidad reciente
    (z-score), de modo que la bolsa (~1%/día) y un par EM (~0.3%/día) pesen
    de forma comparable. Solo lectura; si todos los símbolos fallan, devuelve
    ``{}`` y el llamador degrada con elegancia.
    """
    rets_1d: dict[str, float] = {}
    contributions: list[float] = []
    weights: list[float] = []
    for name, symbol, signed_weight in RISK_BASKET:
        try:
            series = fetch_yahoo_series(symbol, lookback_days)
            daily = series["y"].pct_change().dropna()
            ret_1d = float(daily.iloc[-1]) * 100
            vol = float(daily.tail(20).std()) or float("nan")
            z = (float(daily.iloc[-1]) / vol) if vol == vol and vol > 0 else 0.0
        except Exception as exc:  # un símbolo caído no tumba el contexto
            logger.warning("risk_context: %s (%s) falló: %s", name, symbol, exc)
            continue
        rets_1d[name] = round(ret_1d, 3)
        sign = 1.0 if signed_weight >= 0 else -1.0
        weight = abs(signed_weight)
        contributions.append(sign * weight * z)
        weights.append(weight)

    if not weights:
        return {}

    composite = sum(contributions) / sum(weights)
    em_peers = [rets_1d[k] for k in ("usdclp", "usdmxn", "usdbrl") if k in rets_1d]
    em_avg = sum(em_peers) / len(em_peers) if em_peers else 0.0

    return {
        "composite": round(composite, 3),
        "equity_ret_1d_pct": rets_1d.get("equity", 0.0),
        "dxy_ret_1d_pct": rets_1d.get("dxy", 0.0),
        "brent_ret_1d_pct": rets_1d.get("brent", 0.0),
        "em_peers_ret_1d_pct": round(em_avg, 3),
        "n_drivers": len(weights),
    }
